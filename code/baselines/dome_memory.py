"""Neo4j-backed memory module for DOME, faithful to the original
`pipline/MEM.py` flow but driven by a caller-supplied chat function (an
OpenAI-compatible endpoint) and a local sentence-transformer model.

Pipeline preserved (same as original MEM.py):
  set_initial(li, title)
    - For each text in li, run the original KGC prompt (`prompt_KG.PROMPT_TEMPLATE['KGC']`)
      to extract triples (head, relation, tail).
    - Embed unique entities with sentence-transformers/all-MiniLM-L6-v2.
    - Persist triples into Neo4j with Cypher MERGE (Entity{name,title}, time).
    - Persist (entity, time_step) list + entity embeddings to a per-title pickle.

  set_history(text, title, step)
    - Same as set_initial but extends, with init=False and the given time_step.

  find_relevant_info(current_outline, step, title)
    - Build an outline-side KG via KGC prompt -> embeddings.
    - Cosine-match outline entities against history entity embeddings (>=0.7).
    - Step<2: collect 1-hop neighbors only.  Step>=2: also find shortest paths
      between matched entities and verbalize them via the `get_path` prompt.
    - Run the `evaluate` prompt to score each (outline, neighbor-as-sentence)
      pair; keep top-K (=9 in the paper, we keep 9 if >=16 triples else all).
    - Mine schema-grouped temporal patterns (s1..s4) and verbalize via
      schema1..schema4 prompts.
    - Concatenate path_prompt + neighbor_prompt + deep_info as the returned
      'history' string -- the same output as the upstream `find_relevant_info`.

The per-title isolation lets multiple samples run concurrently against the
same Neo4j: every node carries a `title` property, every Cypher query filters
on it, and per-title pickles live in `<store_root>/<title>.pkl`.
"""
from __future__ import annotations

import os, re, pickle, itertools, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from typing import List, Tuple
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DOME_DIR = _os.path.join(
    _HERE, "Generating-Long-form-Story-Using-Dynamic-Hierarchical-Outlining-with-Memory-Enhancement",
    "pipline")
sys.path.insert(0, _DOME_DIR)
from prompt_KG import PROMPT_TEMPLATE  # noqa: E402

NEO4J_URI = os.environ.get("DOME_NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("DOME_NEO4J_USER", "")
NEO4J_PASS = os.environ.get("DOME_NEO4J_PASS", "")
EMBED_MODEL = os.environ.get(
    "DOME_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
STORE_ROOT = os.environ.get(
    "DOME_STORE_ROOT", ".dome_kg_store")

os.makedirs(STORE_ROOT, exist_ok=True)

_driver = None
_driver_lock = threading.Lock()

def get_driver():
    global _driver
    with _driver_lock:
        if _driver is None:
            auth = (NEO4J_USER, NEO4J_PASS) if NEO4J_USER else None
            _driver = GraphDatabase.driver(NEO4J_URI, auth=auth)
    return _driver

_embedder = None
_embedder_lock = threading.Lock()

def get_embedder():
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = SentenceTransformer(EMBED_MODEL,
                cache_folder=os.environ.get("DOME_HF_CACHE"),
                local_files_only=True)
    return _embedder


# ---------- helpers ----------------------------------------------------------
def _render(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _embed(strings):
    if not strings:
        return np.zeros((0, 384), dtype=np.float32)
    return get_embedder().encode(list(strings), normalize_embeddings=True,
                                 show_progress_bar=False)


def parse_string_to_tuple(s: str):
    """Verbatim-equivalent of MEM.py::parse_string_to_tuple."""
    s1 = s.split(".")
    if len(s1) == 1:
        return None
    body = ".".join(s1[1:]).strip()
    body = body.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    parts = [p.strip().replace("_", " ") for p in body.split(",")]
    if len(parts) < 3:
        return None
    if len(parts) == 3:
        return tuple(parts)
    return tuple(parts[:2] + [", ".join(parts[2:])])


def data_process(text: str) -> pd.DataFrame:
    rows = []
    for li in text.strip().split("\n"):
        if not li.strip():
            continue
        t = parse_string_to_tuple(li)
        if t and len(t) == 3 and all(t):
            rows.append(t)
    return pd.DataFrame(rows, columns=["head", "relation", "tail"]).dropna()


# ---------- KG extraction (uses the model via the supplied chat fn) --------
def get_inputkg(text: str, llm_chat) -> pd.DataFrame:
    p = _render(PROMPT_TEMPLATE["KGC"], text=text)
    res = llm_chat(p)
    return data_process(res)


# ---------- Neo4j ops --------------------------------------------------------
def _safe_rel(name: str) -> str:
    """Neo4j relationship type cannot contain spaces or special chars."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name) or "REL"


def storeneo4j(df: pd.DataFrame, title: str, time_step: int):
    drv = get_driver()
    with drv.session() as session:
        for _, row in df.iterrows():
            h, r, t = row["head"], row["relation"], row["tail"]
            rel = _safe_rel(r)
            session.run(
                "MERGE (h:Entity {name:$h, title:$title}) "
                "MERGE (t:Entity {name:$t, title:$title}) "
                f"MERGE (h)-[rr:`{rel}` {{title:$title}}]->(t) "
                "ON CREATE SET rr.time=$time SET rr.relation=$relname",
                h=h, t=t, title=title, time=time_step, relname=r)


def find_shortest_path(start, end, candidate_list, title) -> Tuple[List[str], object]:
    drv = get_driver()
    with drv.session() as session:
        rec = session.run(
            "MATCH (s:Entity {name:$s, title:$title}), "
            "(e:Entity {name:$e, title:$title}) "
            "MATCH p = (s)-[*..5]->(e) "
            "WITH p, length(p) AS len ORDER BY len ASC LIMIT 1 RETURN p",
            s=start, e=end, title=title)
        result = list(rec)
    if not result:
        return [], []
    paths, exist_entity = [], []
    for r in result:
        path = r["p"]
        ents = [n["name"] for n in path.nodes]
        rels = [getattr(rel, "type", "") for rel in path.relationships]
        s_ents = [e.replace("_", " ") for e in ents]
        s_rels = [r.replace("_", " ") for r in rels]
        path_str = ""
        for i, e in enumerate(s_ents):
            if e in candidate_list:
                exist_entity = e
            path_str += e
            if i < len(s_rels):
                path_str += "->" + s_rels[i] + "->"
        paths.append(path_str)
    if len(paths) > 5:
        paths = sorted(paths, key=len)[:5]
    return paths, exist_entity


def get_entity_neighbors(name: str, title: str) -> List[List[str]]:
    drv = get_driver()
    q = ("MATCH (e:Entity {name:$name, title:$title})-[r]->(n:Entity {title:$title}) "
         "RETURN type(r) AS rt, collect(n.name) AS nb")
    with drv.session() as session:
        rec = list(session.run(q, name=name, title=title))
    out = []
    for r in rec:
        rt = r["rt"].replace("_", " ")
        nb = ",".join([x.replace("_", " ") for x in r["nb"]])
        out.append([name.replace("_", " "), rt, nb])
    return out


# ---------- pkl per-title ---------------------------------------------------
def _pkl_path(title: str):
    return os.path.join(STORE_ROOT, f"{title}.pkl")


def set_history_entity_embeddings(df: pd.DataFrame, title: str, time_step=1, init=True):
    if df.empty:
        return
    ents = list(dict.fromkeys(df["head"].tolist() + df["tail"].tolist()))
    embs = _embed(ents)
    enlist_time = [(e, time_step) for e in ents]
    storeneo4j(df, title, time_step)
    p = _pkl_path(title)
    if init or not os.path.exists(p):
        d = {"entity": enlist_time, "embedding": embs}
    else:
        with open(p, "rb") as f:
            d = pickle.load(f)
        d["entity"] = d["entity"] + enlist_time
        d["embedding"] = np.concatenate([d["embedding"], embs], axis=0)
    with open(p, "wb") as f:
        pickle.dump(d, f)


def get_history_entity_embeddings(title: str):
    p = _pkl_path(title)
    if not os.path.exists(p):
        return [], np.zeros((0, 384), dtype=np.float32)
    with open(p, "rb") as f:
        d = pickle.load(f)
    return d["entity"], d["embedding"]


# ---------- relevance --------------------------------------------------------
def find_sim_entity(enembedding, enlist, input_emb, input_list, threshold=0.7):
    if len(enembedding) == 0 or len(input_list) == 0:
        return []
    sim = cosine_similarity(input_emb, enembedding)  # (Q, K)
    matches = []
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            if sim[i, j] >= threshold:
                name = enlist[j][0].replace("_", " ")
                if name not in matches:
                    matches.append(name)
    return matches


def combine_lists(*lists):
    combos = list(itertools.product(*lists))
    out = []
    for combo in combos:
        new = []
        for sub in combo:
            if isinstance(sub, list):
                new += sub
            else:
                new.append(sub)
        out.append(new)
    return out


def find_path(match_kg, title):
    if len(match_kg) <= 1:
        return {}
    start = match_kg[0]
    candidate = list(match_kg[1:])
    result_paths = []
    while True:
        flag, paths_list = 0, []
        while candidate:
            end = candidate[0]; candidate.remove(end)
            paths, exist = find_shortest_path(start, end, candidate, title)
            if not paths:
                flag = 1
                if not candidate:
                    flag = 0; break
                start = candidate[0]; candidate.remove(start); break
            else:
                pl = [p.split("->") for p in paths]
                if pl: paths_list.append(pl)
            if exist and isinstance(exist, str) and exist in candidate:
                try: candidate.remove(exist)
                except ValueError: pass
            start = end
        rp = combine_lists(*paths_list) if paths_list else []
        if rp: result_paths.extend(rp)
        if flag != 1: break
    return result_paths[:5]


def find_neighbor(match_kg, title):
    out = []
    for m in match_kg:
        out.extend(get_entity_neighbors(m.replace("_", " "), title))
    return out


def get_prompt_path(result_path, llm_chat):
    if not result_path: return ""
    s = "\n".join(["->".join(p) if isinstance(p, list) else str(p)
                   for p in result_path])
    return llm_chat(_render(PROMPT_TEMPLATE["get_path"], Path=s))


def get_prompt_neighbor(neighbor_list, llm_chat):
    if not neighbor_list: return ""
    s = "\n".join(["->".join(n) for n in neighbor_list])
    return llm_chat(_render(PROMPT_TEMPLATE["get_neighbor"], neighbor=s))


def evaluator(outline, triples, llm_chat, top_k=9):
    """Convert each triple to a sentence (graph2text), then if many, score
    relevance against the outline (evaluate prompt) and keep top_k."""
    # Conversion and scoring of each triple are independent. The upstream
    # implementation processes them serially, which makes a single retrieval
    # very slow in wall-clock time. Here each triple still uses exactly the
    # same original prompt; requests are only issued in parallel, without
    # merging, sampling, or changing the final order or the top-k algorithm.
    workers = max(
        1,
        min(int(os.environ.get("DOME_MEMORY_WORKERS", "4")), len(triples)),
    )

    def graph_to_text(tr):
        try:
            sen = llm_chat(_render(PROMPT_TEMPLATE["graph2text"],
                                   triple=str(tuple(tr))))
        except Exception:
            sen = ""
        return sen.strip().split("\n")[0] if sen else ""

    if triples:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            sentences = list(executor.map(graph_to_text, triples))
    else:
        sentences = []
    if len(triples) < 16:
        return sentences, list(triples)

    def score_sentence(sen):
        if not sen:
            return 0
        try:
            r = llm_chat(_render(PROMPT_TEMPLATE["evaluate"],
                                 outline=outline, triplesentence=sen))
            m = re.search(r"Score:\s*(\d+)", r)
            return int(m.group(1)) if m else 0
        except Exception:
            return 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        scores = list(executor.map(score_sentence, sentences))
    order = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    new_sen = [sentences[i] for i, _ in order]
    new_tri = [triples[i] for i, _ in order]
    return new_sen, new_tri


def addtime(triples, title):
    enlist, _ = get_history_entity_embeddings(title)
    if not enlist: return []
    name_to_steps = {}
    for name, step in enlist:
        name_to_steps.setdefault(name, []).append(step)
    out = []
    for tr in triples:
        head, _, tail = tr[0], tr[1], tr[2]
        ts = name_to_steps.get(tail, name_to_steps.get(head, [1]))
        t = max(1, ts[0] if ts else 1)
        rec = list(tr[:3]) + [t]
        if rec not in out: out.append(rec)
    return out


def _group_by_idx(quads, idxs):
    grouped = {}
    for q in quads:
        key = tuple(q[i] for i in idxs)
        grouped.setdefault(key, []).append(q)
    res, keep = [], []
    for v in grouped.values():
        (keep if len(v) == 1 else res).append(v[0] if len(v) == 1 else v)
    return res, keep


def _trans_schema(quad_groups, schema_key, llm_chat):
    out = ""
    for grp in quad_groups:
        if not isinstance(grp, list) or len(grp) <= 1: continue
        try:
            r = llm_chat(_render(PROMPT_TEMPLATE[schema_key], inlist=str(grp)))
            out += (r.strip() + "\n")
        except Exception:
            continue
    return out


def info_refine(quads, llm_chat):
    if not quads: return ""
    res = ""
    s1, t1 = _group_by_idx(quads, [0, 1, 2])
    if s1: res += _trans_schema(s1, "schema2", llm_chat) + "\n"
    if t1:
        s2, t2 = _group_by_idx(t1, [0, 1])
        if s2: res += _trans_schema(s2, "schema1", llm_chat) + "\n"
        if t2:
            s3, t3 = _group_by_idx(t2, [0, 2])
            if s3: res += _trans_schema(s3, "schema4", llm_chat) + "\n"
            if t3:
                s4, _ = _group_by_idx(t3, [1, 2])
                if s4: res += _trans_schema(s4, "schema3", llm_chat) + "\n"
    return res


# ---------- public API mirroring MEM.py -------------------------------------
def set_initial(li, title, llm_chat):
    """li = [setting, character, outline]; build initial KG for the title."""
    df_all = pd.DataFrame(columns=["head", "relation", "tail"])
    for txt in li:
        if not txt: continue
        df = get_inputkg(txt, llm_chat)
        if not df.empty:
            df_all = pd.concat([df_all, df], ignore_index=True)
    df_all = df_all.dropna()
    set_history_entity_embeddings(df_all, title, time_step=0, init=True)


def set_history(text, title, step, llm_chat):
    df = get_inputkg(text, llm_chat)
    set_history_entity_embeddings(df, title, time_step=step, init=False)


def find_relevant_info(text, step, title, llm_chat):
    """Returns a natural-language `history` string relevant to the outline."""
    enlist, enembedding = get_history_entity_embeddings(title)
    if not enlist:
        return ""
    in_df = get_inputkg(text, llm_chat)
    in_ents = list(dict.fromkeys(in_df["head"].tolist() + in_df["tail"].tolist()))
    in_emb = _embed(in_ents)
    match = find_sim_entity(enembedding, enlist, in_emb, in_ents)
    if not match:
        return ""
    if step < 2:
        path_prompt = ""
        neighbor_list = find_neighbor(match, title)
    else:
        rp = find_path(match, title)
        neighbor_list = find_neighbor(match, title)
        path_prompt = get_prompt_path(rp, llm_chat) or ""
    sentences, kept = evaluator(text, neighbor_list, llm_chat)
    quads = addtime(kept, title)
    deep = info_refine(quads, llm_chat)
    neighbor_prompt = "\n".join([s for s in sentences if s])
    return (path_prompt + "\n" + neighbor_prompt + "\n" + (deep or "")).strip()


def reset_title(title):
    """Wipe state for a single title (Neo4j subgraph + pkl)."""
    drv = get_driver()
    with drv.session() as session:
        session.run("MATCH (n:Entity {title:$title}) DETACH DELETE n", title=title)
    p = _pkl_path(title)
    if os.path.exists(p): os.remove(p)
