#!/usr/bin/env python3
"""Check every quotation and count in the case-study appendix against the files in this package.

Usage: python code/analysis/case_study.py [package_root]

Sources, all shipped:
  results/stories/<arm>.jsonl              chapters, and for NstAgent the per-chapter tool trace
  results/case_study/rollsum_summaries.jsonl   RollSum's rolling summary after every chapter
  results/scores/constory/<arm>.csv        the judge's reported errors

NstAgent's intermediate states are not stored separately: this script rebuilds them by replaying the
``update`` calls in each chapter's tool trace through the agent's own update function
(``LongStoryAgent._tool_narrative_ops_update`` in code/agent/agent_core.py), checks that the replay ends
in exactly the stored ``final_state``, and writes the state after every chapter to
results/case_study/nstagent_states.jsonl. The report goes to results/case_study/case_checks.md.
Chapters are numbered from 0, as in the paper.
"""
import ast
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "code/agent"))
import agent_core as core  # noqa: E402
from generation_common import count_words  # noqa: E402

csv.field_size_limit(sys.maxsize)
OUT = ROOT / "results/case_study"
SUBTYPES = [
    "characterization_memory_contradictions", "characterization_knowledge_contradictions",
    "characterization_skill_power_fluctuations", "characterization_forgotten_abilities",
    "factual_detail_appearance_mismatches", "factual_detail_nomenclature_confusions",
    "factual_detail_quantitative_mismatches", "narrative_style_perspective_confusions",
    "narrative_style_tone_inconsistencies", "narrative_style_style_shifts",
    "timeline_plot_absolute_time_contradictions", "timeline_plot_duration_timeline_contradictions",
    "timeline_plot_simultaneity_contradictions", "timeline_plot_causeless_effects",
    "timeline_plot_causal_logic_violations", "timeline_plot_abandoned_plot_elements",
    "world_building_core_rules_violations", "world_building_social_norms_violations",
    "world_building_geographical_contradictions",
]


def norm(text):
    """Compare quotations modulo typography: curly quotes, dash style, Markdown emphasis, whitespace."""
    text = str(text).replace("**", "").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("—", "---").replace("–", "-")
    return " ".join(text.split())


def record(arm, story_id):
    for line in open(ROOT / f"results/stories/{arm}.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if row["id"] == story_id:
            return row
    raise KeyError((arm, story_id))


def chapters(row):
    return [c["content"] if isinstance(c, dict) else c for c in row["story"]]


def replay(row):
    """State after each chapter, rebuilt from the tool trace with the agent's own update function."""
    state, after = core.StoryState(), []
    for chapter in row["story"]:
        cid = int(chapter["id"])
        for call in chapter.get("tool_trace") or []:
            if call.get("name") != "update" or str(call.get("ok")) != "True":
                continue
            args = call["args"]
            args = ast.literal_eval(args) if isinstance(args, str) else args
            core.LongStoryAgent._tool_narrative_ops_update(args, state, cid)
        after.append(json.loads(json.dumps(state.to_dict())))
    return after


def judged_errors(arm, story_id):
    with open(ROOT / f"results/scores/constory/{arm}.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("id") == str(story_id):
                errors = {k: json.loads(r.get(k) or "[]") for k in SUBTYPES}
                return sum(map(len, errors.values())), errors, r["target_chapter_count"]
    raise KeyError((arm, story_id))


summaries = {r["id"]: r["summary"] for r in map(json.loads, open(OUT / "rollsum_summaries.jsonl", encoding="utf-8"))}
lines, failures, states_out = [], 0, []


def check(label, ok, detail=""):
    global failures
    failures += not ok
    lines.append(f"| {'✓' if ok else '✗'} | {label} | {detail} |")


def contains(text, *fragments):
    return all(norm(f) in norm(text) for f in fragments)


def where(texts, fragment):
    return [i for i, t in enumerate(texts) if norm(fragment) in norm(t)]


def case(title, rs_arm, nst_arm, sid):
    rs, nst = record(rs_arm, sid), record(nst_arm, sid)
    states = replay(nst)
    check("replayed NstAgent state equals the stored final_state", states[-1] == nst["final_state"],
          f"{len(states)} chapters")
    states_out.append({"arm": nst_arm, "id": sid, "state_after_chapter": states})
    return rs, nst, chapters(rs), chapters(nst), summaries[sid], states


def header(title):
    lines.extend(["", f"## {title}", "", "| | Claim | Evidence |", "|---|---|---|"])


# ---------------------------------------------------------------- Case 1
header("Case 1: Repeated events (DeepSeek-V4-Flash, 20K words, id 1338)")
rs, nst, R, A, S, states = case("1", "deepseek/rollsum_20k", "deepseek/nstagent_20k", 1338)
check("RollSum Ch. 2: the stone sings for Nerys", contains(R[2], "He drew it out. The runes shifted like sand in a current, and the stone sang", "a hum that answered something deep in the water, in the coral, in her."))
check("RollSum Ch. 3: the stone sings for Cyra", contains(R[3], "The stone sang", "a thin, high note, a string that had been waiting to be plucked."))
check("RollSum summary before Ch. 8 says it has never sung", contains(S[7], "The stone has never sung for him; it only pulses warmly.", "The prophecy-stone hums in the braided voices of the found souls"))
check("RollSum summary before Ch. 8 is 2,644 words", count_words(S[7]) == 2644, f"{count_words(S[7]):,} words")
check("RollSum Ch. 8 declares a first time", contains(R[8], "The prophecy-stone at Kaelen's hip pulsed, warm as a coal, and for the first time since the Ember-Scar it began to sing."))
check("RollSum Ch. 7: antagonists vow to wait at every facet", contains(R[7], "at every facet"))
past = {e["key"]: e for e in states[-1]["past_events"]}
e1, e2 = past.get("nerys_joins_kaelen", {}), past.get("cyra_joins_kaelen", {})
check("NstAgent past event nerys_joins_kaelen", contains(e1.get("description", ""), "On the drowned Tidal spire, Kaelen found Nerys", "The prophecy-stone sang in her presence"), f"added in Ch. {e1.get('chapter_id')}")
check("NstAgent past event cyra_joins_kaelen", contains(e2.get("description", ""), "At the Gyre Bastion", "the prophecy-stone sang for Cyra, the disgraced wind-rider."), f"added in Ch. {e2.get('chapter_id')}")
# A resolved requirement is removed from the state, so it resolves in the first chapter it is absent.
reqs = {}
for ch, st_ in enumerate(states):
    keys = {r["key"] for r in st_["future_requirements"]}
    for r in st_["future_requirements"]:
        reqs.setdefault(r["key"], {"text": r.get("description", ""), "added": ch})
    for key, info in reqs.items():
        if key not in keys and "resolved" not in info:
            info["resolved"] = ch
req = reqs.get("prophecy_stone_responds_to_champions", {})
check("Requirement prophecy_stone_responds_to_champions: added Ch. 1, resolved Ch. 2",
      req.get("added") == 1 and req.get("resolved") == 2
      and contains(req.get("text", ""), "must be shown to respond to the presence of other unbounded souls"),
      f"added {req.get('added')}, resolved {req.get('resolved')}")
check("All sixteen requirements resolved by the final chapter",
      len(reqs) == 16 and not states[-1]["future_requirements"],
      f"{sum('resolved' in r for r in reqs.values())}/{len(reqs)} resolved")
n_nst, _, w_nst = judged_errors("deepseek/nstagent_20k", 1338)
n_rs, err_rs, w_rs = judged_errors("deepseek/rollsum_20k", 1338)
check("Judge: 1 contradiction for NstAgent (last 7) vs 10 for RollSum (last 8)", (n_nst, n_rs, w_nst, w_rs) == (1, 10, "7", "8"), f"NstAgent {n_nst} in last {w_nst}; RollSum {n_rs} in last {w_rs}")
check("Judge flags the Ch. 7 vow as an abandoned plot element",
      any("every facet" in json.dumps(e) for e in err_rs["timeline_plot_abandoned_plot_elements"]))

# ---------------------------------------------------------------- Case 2
header("Case 2: Identity facts across 40 chapters (GPT-5.6 Luna, 100K words, id 1591)")
rs, nst, R, A, S, states = case("2", "luna/rollsum_100k", "luna/nstagent_100k", 1591)
check("RollSum Ch. 13: Vale born 1961", contains(R[13], "I was born in 1961."))
check("RollSum summary before Ch. 36: Vale approximately 1963", contains(S[35], "Vale, born approximately 1963, spent summers at a farmhouse with her younger sister Miriam"))
check("RollSum Ch. 38: Vale born 6 February 1963", contains(R[38], "Irena Vale. Born 6 February 1963."))
check("RollSum Ch. 2: Daniel born March 14, 1994", contains(R[2], "Daniel Reyes, born March 14, 1994, Newark, New Jersey."))
check("RollSum summary before Ch. 36: Daniel 17 October 1994", contains(S[35], "Daniel, born 17 October 1994, directly viewed the tape"))
check("RollSum Ch. 36: Daniel October seventeenth", contains(R[36], "You're Daniel Mateo Reyes. Born October seventeenth, nineteen ninety-four."))
check("RollSum Mara: July 1987 in Ch. 6, June 1988 from Ch. 7", contains(R[6], "July seventeenth, 1987") and contains(R[7], "June fourteenth, 1988"),
      f"'June fourteenth, 1988' in chapters {where(R, 'June fourteenth, 1988')}")
check("RollSum summary carries Mara born 1988", contains(S[35], "Mara, born 1988") and contains(S[-1], "1988"))
check("RollSum summary before Ch. 36 is 4,278 words", count_words(S[35]) == 4278, f"{count_words(S[35]):,} words")
check("NstAgent Ch. 24: Vale born 1969", contains(A[24], "I had been born in 1969, so the date was possible."))


def snapshot(st_, name):
    return " ".join(c.get("description", "") for c in st_["character_states"] if name in str(c.get("name", "")))


vale = [ch for ch in range(len(states)) if "1969" in snapshot(states[ch], "Vale")]
check("NstAgent Vale snapshot says 1969 after every chapter from 28 to 38", all(ch in vale for ch in range(28, 39)), f"chapters {vale[0]}-{vale[-1]}" if vale else "")
check("Vale snapshot quote", any(contains(snapshot(states[ch], "Vale"), "Vale remains active, exhausted, and factually oriented as a pre-cutoff director born in 1969.") for ch in range(28, 39)))
owen = [ch for ch in range(len(states)) if "born in 1993" in snapshot(states[ch], "Owen").lower()]
check("NstAgent Owen snapshot says 1993 from Ch. 12 to the end", owen and owen[0] == 12 and owen == list(range(12, len(states))), f"chapters {owen[0]}-{owen[-1]}" if owen else "")
check("Owen snapshot quote", any(contains(snapshot(st_, "Owen"), "barred from the restricted media room because he was born in 1993, after the January 1, 1991 vulnerability cutoff") for st_ in states))
daniel = [ch for ch, t in enumerate(A) if re.search(r"(1993|ninety-three)", t) and "Daniel" in t]
check("NstAgent Daniel born 1993 in Chapters 0, 1, 7, 28, 29", all(ch in daniel for ch in (0, 1, 7, 28, 29)), f"chapters mentioning Daniel and 1993: {daniel}")
n_nst, _, w_nst = judged_errors("luna/nstagent_100k", 1591)
n_rs, _, w_rs = judged_errors("luna/rollsum_100k", 1591)
check("Judge: 25 for RollSum vs 10 for NstAgent in the last four chapters", (n_rs, n_nst, w_rs, w_nst) == (25, 10, "4", "4"), f"RollSum {n_rs}, NstAgent {n_nst}")

# ---------------------------------------------------------------- Case 3
header("Case 3: When the state is wrong (GPT-5.6 Luna, 10K words, id 1454)")
rs, nst, R, A, S, states = case("3", "luna/rollsum_10k", "luna/nstagent_10k", 1454)
check("RollSum Ch. 0: Bishop Creel", contains(R[0], 'Behind him, Bishop Creel had said, "Continue."'),
      "the paper sets the inner quotation in single quotes")
check("RollSum Ch. 8: Bishop Creel", contains(R[8], "For one breath it was Bishop Creel, broad and pale beneath his red skullcap."))
creel = where(R, "Bishop Creel")
check("RollSum names Bishop Creel in Ch. 0 and four later chapters, and never renames him", creel[0] == 0 and len(creel) == 5 and not any(re.search(r"Bishop (?!Creel)[A-Z]", t) for t in R), f"chapters {creel}")
check("RollSum Mara is nine in Chapters 0 and 4", contains(R[0], "Mara had been nine, narrow-shouldered and solemn") and contains(R[4], "nine years old, wrists bound, candles guttering whenever she breathed."))
check("NstAgent Ch. 0: Bishop Armitage", contains(A[0], "Bishop Armitage ordered Elias to conduct the exorcism, though he had been ordained only two years and had never performed one alone."))
check("NstAgent Ch. 4: Bishop Haldane", contains(A[4], "Bishop Haldane said the demon would imitate her, and that hesitation would damn her."))
arm_, hal = where(A, "Armitage"), where(A, "Haldane")
check("Armitage only in Ch. 0; Haldane in five of the remaining nine chapters", arm_ == [0] and len([c for c in hal if c > 0]) == 5 and min(hal) == 4, f"Armitage {arm_}, Haldane {hal}")
check("NstAgent state records a past event naming Haldane", any(contains(e["description"], "Elias confessed that Mara had asked him to stop during her exorcism", "because he feared disobeying Bishop Haldane more than losing her.") for e in states[-1]["past_events"]))
check("NstAgent Mara nine in Ch. 0, eight in Ch. 6", contains(A[0], "Mara Venn had been nine when they brought her to Saint Bartholomew's rectory.") and contains(A[6], "Mara stepped through, eight years old, her nightdress dark with sweat."))
n_nst, err_nst, _ = judged_errors("luna/nstagent_10k", 1454)
n_rs, _, _ = judged_errors("luna/rollsum_10k", 1454)
check("Judge: 15 for NstAgent vs 6 for RollSum", (n_nst, n_rs) == (15, 6), f"NstAgent {n_nst}, RollSum {n_rs}")
check("Judge flags the rename as a memory contradiction and a nomenclature confusion",
      all(any("Haldane" in json.dumps(e) for e in err_nst[k]) for k in ("characterization_memory_contradictions", "factual_detail_nomenclature_confusions")))

report = ["# Case-study checks", "",
          f"Generated by `code/analysis/case_study.py`: {failures} failed check(s). Quotations are compared after",
          "normalizing curly quotes, dash style, Markdown emphasis, and whitespace; chapters are numbered from 0."] + lines + [""]
(OUT / "case_checks.md").write_text("\n".join(report), encoding="utf-8")
with open(OUT / "nstagent_states.jsonl", "w", encoding="utf-8") as f:
    for row in states_out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("\n".join(report))
sys.exit(1 if failures else 0)
