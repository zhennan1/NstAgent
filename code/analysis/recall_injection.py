#!/usr/bin/env python3
"""Build and score a controlled ConStory contradiction-recall experiment.

The calibration set uses all 20 LongStoryAgent stories at each of 10k, 20k,
50k, and 100k. Per length, it creates:

* 15 positive examples: three for each ConStory category.
* 5 consistent controls: one for each category.

Positive examples are balanced across near (same chapter) and far
(widely-separated chapters) contradictions over the four length tiers.
Every injected pair contains a unique, natural-looking entity token so judge
outputs can be matched without relying on fuzzy semantic grading.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DATASETS: Mapping[str, str] = {
    "longstory_10k": "stories/longstory/longstory_deepseek_10k_20.jsonl",
    "longstory_20k": "stories/longstory/longstory_deepseek_20k_20.jsonl",
    "longstory_50k": "stories/longstory/longstory_deepseek_50k_20.jsonl",
    "longstory_100k": "stories/longstory/longstory_deepseek_100k_20.jsonl",
}

CATEGORIES: Mapping[str, Mapping[str, Any]] = {
    "characterization": {
        "subtype": "knowledge_contradictions",
        "columns": [
            "characterization_memory_contradictions",
            "characterization_knowledge_contradictions",
            "characterization_skill_power_fluctuations",
            "characterization_forgotten_abilities",
        ],
    },
    "factual_detail": {
        "subtype": "appearance_mismatches",
        "columns": [
            "factual_detail_appearance_mismatches",
            "factual_detail_nomenclature_confusions",
            "factual_detail_quantitative_mismatches",
        ],
    },
    "narrative_style": {
        "subtype": "style_shifts",
        "columns": [
            "narrative_style_perspective_confusions",
            "narrative_style_tone_inconsistencies",
            "narrative_style_style_shifts",
        ],
    },
    "timeline_plot": {
        "subtype": "duration_timeline_contradictions",
        "columns": [
            "timeline_plot_absolute_time_contradictions",
            "timeline_plot_duration_timeline_contradictions",
            "timeline_plot_simultaneity_contradictions",
            "timeline_plot_causeless_effects",
            "timeline_plot_causal_logic_violations",
            "timeline_plot_abandoned_plot_elements",
        ],
    },
    "world_building": {
        "subtype": "core_rules_violations",
        "columns": [
            "world_building_core_rules_violations",
            "world_building_social_norms_violations",
            "world_building_geographical_contradictions",
        ],
    },
}

ALL_ERROR_COLUMNS: List[str] = [
    column
    for category in CATEGORIES.values()
    for column in category["columns"]
]


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stable_token(dataset: str, story_id: Any, category: str, label: str) -> str:
    raw = f"{dataset}|{story_id}|{category}|{label}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:8].upper()
    return f"QX-{digest}"


def pair_text(
    category: str,
    token: str,
    positive: bool,
    language: str,
    subject: str,
) -> Tuple[str, str]:
    """Return two explicit statements: contradictory if positive, consistent otherwise."""
    if language == "zh":
        name = subject or "故事主角"
        pairs = {
            "characterization": (
                f"在查阅“奈拉文卷宗 {token}”时，{name}明确承认自己从未学习过奈拉文字，一个符号也无法辨认。",
                (
                    f"后来没有接受课程、翻译、魔法或任何解释，{name}却流利地逐字译出了整页“奈拉文卷宗 {token}”，并声称自己从童年起就一直精通这种文字。"
                    if positive
                    else f"后来面对同一页“奈拉文卷宗 {token}”，{name}仍完全无法阅读，只能请受过训练的译者逐字解释。"
                ),
            ),
            "factual_detail": (
                (
                    f"在与所有情节物件隔离的密封展柜中，“测绘令牌 {token}”清晰可见的物理轮廓是不可变形的刚性正六边形，工匠档案确认其外观从制造后从未改变。"
                ),
                (
                    f"稍后同一密封展柜里的“测绘令牌 {token}”清晰可见的物理轮廓却是不可变形的完美圆形，档案同样坚称其外观从制造后从未改变。"
                    if positive
                    else f"稍后再次检查同一密封展柜里的“测绘令牌 {token}”，它清晰可见的物理轮廓依然是不可变形的刚性正六边形，与工匠档案完全一致。"
                ),
            ),
            "narrative_style": (
                (
                    f"在标记 {token} 的段落里，雨丝像银线般贴着窗面滑落，夜色安静地托住每一次呼吸，叙述仍保持小说一贯的抒情笔触。"
                    if positive
                    else f"房间边缘的一块普通黄铜铭牌刻着参考编号 {token}，除此之外没有任何特殊之处。"
                ),
                (
                    f"没有梦境、文件、引文、界面或场景转换作为理由，叙述在标记 {token} 处突然变成：SYSTEM_DIAGNOSTIC；SUBJECT={name}；MODE=VECTOR_SCAN；EXECUTE=7-B；RETURN_CODE=0；END_FIELDS；随后又立刻恢复原来的小说文体。"
                    if positive
                    else f"稍后那块普通黄铜铭牌仍然只刻着同一个参考编号 {token}，没有发生任何变化。"
                ),
            ),
            "timeline_plot": (
                f"与所有情节事件隔离的密封计时器 {token} 经过独立认证：在固定条件下完成一次不间断循环恰好需要四小时。",
                (
                    f"同一密封计时器 {token} 的同一次不间断循环随后被权威记录为恰好七小时，记录确认固定条件完全相同且没有暂停、延误或重置。"
                    if positive
                    else f"后来核验的权威记录再次确认：同一密封计时器 {token} 在相同固定条件下完成同一次不间断循环恰好需要四小时。"
                ),
            ),
            "world_building": (
                f"单独存放在密封试验库中的“玻璃钥匙 {token}”遵循一条绝对世界法则：只要接触一滴水就会立刻化为灰烬，而且不存在能够阻止反应的护盾、法术或容器。",
                (
                    f"没有护盾、法术或容器保护，同一把“玻璃钥匙 {token}”在试验库的水槽中裸露浸泡整整一年后依然完好无损，表面甚至没有留下痕迹。"
                    if positive
                    else f"为遵守这条法则，试验员始终把同一把“玻璃钥匙 {token}”存放在完全干燥的真空盒中，从未让它接触任何水分。"
                ),
            ),
        }
    else:
        name = subject or "the protagonist"
        pairs = {
            "characterization": (
                f"While examining the Neral-script dossier {token}, {name} explicitly admitted that they had never studied the Neral script and could not recognize a single symbol.",
                (
                    f"Later, without any lesson, translator, magic, or explanation, {name} fluently translated every line of the same Neral-script dossier {token} and claimed to have read the script since childhood."
                    if positive
                    else f"Later, when shown the same Neral-script dossier {token}, {name} still could not read it and asked a trained translator to explain every symbol."
                ),
            ),
            "factual_detail": (
                f"Inside a sealed display case isolated from every plot object, the clearly visible physical outline of the Surveyor's Token {token} was a rigid, immutable hexagon, and the maker's record said its appearance had never changed.",
                (
                    f"Later, the clearly visible physical outline of the very same Surveyor's Token {token} in that sealed case was a rigid, immutable perfect circle, while the record again insisted its appearance had never changed."
                    if positive
                    else f"Later inspection confirmed that the clearly visible physical outline of the very same Surveyor's Token {token} in that sealed case remained a rigid, immutable hexagon, exactly matching the maker's record."
                ),
            ),
            "narrative_style": (
                (
                    f"In the passage marked {token}, rain traced silver threads down the glass while the night quietly cradled every breath, preserving the novel's established lyrical prose."
                    if positive
                    else f"An ordinary brass plaque at the edge of the room bore the reference mark {token} and had no other unusual property."
                ),
                (
                    f"With no dream, document, quotation, interface, or scene transition to justify it, the narration at marker {token} abruptly became: SYSTEM_DIAGNOSTIC; SUBJECT={name}; MODE=VECTOR_SCAN; EXECUTE=7-B; RETURN_CODE=0; END_FIELDS; it then immediately resumed the prior fictional prose."
                    if positive
                    else f"Later, the same ordinary brass plaque still bore the same reference mark {token} and had not changed in any way."
                ),
            ),
            "timeline_plot": (
                f"The sealed chronometer {token}, isolated from every plot event, was independently certified to require exactly four hours for one uninterrupted cycle under fixed conditions.",
                (
                    f"An equally authoritative record later declared that the same uninterrupted cycle of the sealed chronometer {token} took exactly seven hours under identical fixed conditions, with no pause, delay, or reset."
                    if positive
                    else f"A later authoritative record confirmed that the same uninterrupted cycle of the sealed chronometer {token} took exactly four hours under identical fixed conditions."
                ),
            ),
            "world_building": (
                f"The Vitreous Key {token}, stored alone in a sealed test vault, obeyed an absolute law of the world: one drop of water turned it to ash instantly, and no ward, spell, or container could prevent the reaction.",
                (
                    f"With no ward, spell, or container protecting it, the same bare Vitreous Key {token} remained perfectly intact after a full year submerged in the test vault's water tank, without even a mark."
                    if positive
                    else f"To obey that law, the testers kept the same Vitreous Key {token} inside a perfectly dry vacuum case and never allowed it to touch water."
                ),
            ),
        }
    return pairs[category]


def injection_positions(chapter_count: int, distance: str) -> Tuple[int, int]:
    if chapter_count < 3:
        raise ValueError(f"need at least three chapters, got {chapter_count}")
    if distance == "near":
        middle = min(chapter_count - 2, max(1, chapter_count // 2))
        return middle, middle
    if distance != "far":
        raise ValueError(f"unknown distance: {distance}")
    first = min(chapter_count - 3, max(1, round((chapter_count - 1) * 0.15)))
    second = min(chapter_count - 2, max(first + 1, round((chapter_count - 1) * 0.85)))
    return first, second


def inject_record(
    record: dict,
    dataset: str,
    category: str,
    label: str,
    distance: str,
) -> Tuple[dict, dict]:
    result = copy.deepcopy(record)
    chapters = result.get("story")
    if not isinstance(chapters, list) or not all(isinstance(ch, dict) for ch in chapters):
        raise ValueError(f"{dataset}/{record.get('id')}: story is not a chapter list")

    token = stable_token(dataset, record["id"], category, label)
    original_flat = "\n".join(str(ch.get("content", "")) for ch in chapters)
    if token.lower() in original_flat.lower():
        raise ValueError(f"token collision in source story: {token}")

    character_states = result.get("final_state", {}).get("character_states", [])
    subject = ""
    if character_states and isinstance(character_states[0], dict):
        subject = str(character_states[0].get("name", "")).strip()
    if not subject:
        subject = "故事主角" if record.get("language") == "zh" else "the protagonist"

    positive = label == "positive"
    quote_a, quote_b = pair_text(
        category,
        token,
        positive=positive,
        language=record.get("language", "en"),
        subject=subject,
    )
    chapter_a, chapter_b = injection_positions(len(chapters), distance)

    if chapter_a == chapter_b:
        original = str(chapters[chapter_a].get("content", ""))
        chapters[chapter_a]["content"] = f"{quote_a}\n\n{original}\n\n{quote_b}"
    else:
        original_a = str(chapters[chapter_a].get("content", ""))
        original_b = str(chapters[chapter_b].get("content", ""))
        chapters[chapter_a]["content"] = f"{quote_a}\n\n{original_a}"
        chapters[chapter_b]["content"] = f"{original_b}\n\n{quote_b}"

    subtype = CATEGORIES[category]["subtype"]
    metadata = {
        "dataset": dataset,
        "story_id": record["id"],
        "language": record.get("language", ""),
        "subject": subject,
        "label": label,
        "target_category": category,
        "target_subtype": subtype,
        "target_column": f"{category}_{subtype}",
        "distance": distance,
        "chapter_count": len(chapters),
        "chapter_a": chapter_a,
        "chapter_b": chapter_b,
        "chapter_gap": chapter_b - chapter_a,
        "canary": token,
        "quote_a": quote_a,
        "quote_b": quote_b,
    }
    result["_recall_injection"] = metadata
    return result, metadata


def build_slots(length_index: int) -> List[Tuple[str, str, str]]:
    slots: List[Tuple[str, str, str]] = []
    for category_index, category in enumerate(CATEGORIES):
        control_distance = "near" if (length_index + category_index) % 2 == 0 else "far"
        slots.append((category, "control", control_distance))
        for positive_index in range(3):
            distance = (
                "near"
                if (length_index + category_index + positive_index) % 2 == 0
                else "far"
            )
            slots.append((category, "positive", distance))
    return slots


def generate(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[dict] = []

    for length_index, (dataset, relative_path) in enumerate(DATASETS.items()):
        source_path = repo_root / relative_path
        records = read_jsonl(source_path)
        if len(records) != 20:
            raise ValueError(f"{dataset}: expected 20 records, got {len(records)}")
        if len({str(row["id"]) for row in records}) != 20:
            raise ValueError(f"{dataset}: duplicate story IDs")

        rng = random.Random(args.seed + length_index * 1009)
        shuffled_records = sorted(records, key=lambda row: str(row["id"]))
        rng.shuffle(shuffled_records)
        slots = build_slots(length_index)
        rng.shuffle(slots)

        injected_rows: List[dict] = []
        for record, (category, label, distance) in zip(shuffled_records, slots):
            injected, metadata = inject_record(
                record,
                dataset=dataset,
                category=category,
                label=label,
                distance=distance,
            )
            injected_rows.append(injected)
            metadata["source_file"] = str(source_path)
            metadata["injected_file"] = str(output_dir / f"{dataset}.jsonl")
            manifest_rows.append(metadata)

        output_path = output_dir / f"{dataset}.jsonl"
        write_jsonl(output_path, injected_rows)
        counts = Counter(
            (row["label"], row["target_category"], row["distance"])
            for row in manifest_rows
            if row["dataset"] == dataset
        )
        print(f"{dataset}: wrote {len(injected_rows)} records -> {output_path}")
        print(f"  positive={sum(v for k, v in counts.items() if k[0] == 'positive')}")
        print(f"  control={sum(v for k, v in counts.items() if k[0] == 'control')}")

    manifest_path = output_dir / "injection_manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)

    positives = [row for row in manifest_rows if row["label"] == "positive"]
    controls = [row for row in manifest_rows if row["label"] == "control"]
    validation = {
        "seed": args.seed,
        "total": len(manifest_rows),
        "positive": len(positives),
        "control": len(controls),
        "positive_by_length": Counter(row["dataset"] for row in positives),
        "positive_by_category": Counter(row["target_category"] for row in positives),
        "positive_by_distance": Counter(row["distance"] for row in positives),
        "control_by_category": Counter(row["target_category"] for row in controls),
        "control_by_distance": Counter(row["distance"] for row in controls),
    }
    serializable = {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in validation.items()
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(serializable, ensure_ascii=False, indent=2))


def parse_run_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be JUDGE=/path/to/result_dir")
    judge, raw_path = value.split("=", 1)
    if not judge.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run must be JUDGE=/path/to/result_dir")
    return judge.strip(), Path(raw_path).resolve()


def cell_failed(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() == "PARSE_FAILED"


def contains_token(value: Any, token: str) -> bool:
    if value is None:
        return False
    return token.casefold() in str(value).casefold()


def wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - spread, center + spread


def result_rows(path: Path) -> Dict[str, dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate IDs in result: {path}")
    return by_id


def classify_one(judge: str, manifest: dict, result: dict) -> Tuple[dict, List[dict]]:
    category = manifest["target_category"]
    subtype_column = manifest["target_column"]
    category_columns = list(CATEGORIES[category]["columns"])
    token = manifest["canary"]

    hit_columns = [
        column
        for column in ALL_ERROR_COLUMNS
        if column in result and contains_token(result[column], token)
    ]
    hit_category_columns = [column for column in category_columns if column in hit_columns]
    category_available = all(
        column in result and not cell_failed(result[column])
        for column in category_columns
    )
    story_complete = result.get("evaluation_status") == "completed"

    detection = {
        "judge": judge,
        **manifest,
        "evaluation_status": result.get("evaluation_status", ""),
        "criteria_completed": int(result.get("criteria_completed") or 0),
        "target_category_available": category_available,
        "detected_target_subtype": subtype_column in hit_columns,
        "detected_target_category": bool(hit_category_columns),
        "detected_any_category": bool(hit_columns),
        "misclassified_category": bool(hit_columns) and not bool(hit_category_columns),
        "hit_columns": json.dumps(hit_columns, ensure_ascii=False),
    }
    matched = [
        {
            "judge": judge,
            "dataset": manifest["dataset"],
            "story_id": manifest["story_id"],
            "label": manifest["label"],
            "target_category": category,
            "target_subtype": manifest["target_subtype"],
            "distance": manifest["distance"],
            "canary": token,
            "matched_column": column,
            "matched_cell": result[column],
        }
        for column in hit_columns
    ]
    return detection, matched


def group_rows(rows: Sequence[dict]) -> List[Tuple[str, str, List[dict]]]:
    definitions = [
        ("overall", lambda row: "all"),
        ("length", lambda row: row["dataset"]),
        ("distance", lambda row: row["distance"]),
        ("category", lambda row: row["target_category"]),
        (
            "length_distance",
            lambda row: f"{row['dataset']}|{row['distance']}",
        ),
        (
            "length_category",
            lambda row: f"{row['dataset']}|{row['target_category']}",
        ),
    ]
    grouped: List[Tuple[str, str, List[dict]]] = []
    for group_type, getter in definitions:
        buckets: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            buckets[getter(row)].append(row)
        for group_value, bucket in sorted(buckets.items()):
            grouped.append((group_type, group_value, bucket))
    return grouped


def summarize(detections: Sequence[dict]) -> List[dict]:
    summaries: List[dict] = []
    judges = sorted({row["judge"] for row in detections})
    for judge in judges:
        judge_rows = [row for row in detections if row["judge"] == judge]
        for label in ("positive", "control"):
            label_rows = [row for row in judge_rows if row["label"] == label]
            for group_type, group_value, bucket in group_rows(label_rows):
                n = len(bucket)
                available = sum(bool(row["target_category_available"]) for row in bucket)
                complete = sum(row["evaluation_status"] == "completed" for row in bucket)
                subtype_hits = sum(bool(row["detected_target_subtype"]) for row in bucket)
                category_hits = sum(bool(row["detected_target_category"]) for row in bucket)
                any_hits = sum(bool(row["detected_any_category"]) for row in bucket)
                misclassified = sum(bool(row["misclassified_category"]) for row in bucket)
                subtype_low, subtype_high = wilson(subtype_hits, n)
                category_low, category_high = wilson(category_hits, n)
                any_low, any_high = wilson(any_hits, n)
                conditional_category_low, conditional_category_high = wilson(
                    category_hits,
                    available,
                )
                summaries.append(
                    {
                        "judge": judge,
                        "label": label,
                        "group_type": group_type,
                        "group_value": group_value,
                        "n": n,
                        "complete_results": complete,
                        "target_category_available": available,
                        "subtype_hits": subtype_hits,
                        "subtype_rate": subtype_hits / n if n else math.nan,
                        "subtype_ci95_low": subtype_low,
                        "subtype_ci95_high": subtype_high,
                        "category_hits": category_hits,
                        "category_rate": category_hits / n if n else math.nan,
                        "category_ci95_low": category_low,
                        "category_ci95_high": category_high,
                        "category_rate_when_available": (
                            category_hits / available if available else math.nan
                        ),
                        "category_available_ci95_low": conditional_category_low,
                        "category_available_ci95_high": conditional_category_high,
                        "any_hits": any_hits,
                        "any_rate": any_hits / n if n else math.nan,
                        "any_ci95_low": any_low,
                        "any_ci95_high": any_high,
                        "misclassified_category": misclassified,
                    }
                )
    return summaries


def score(args: argparse.Namespace) -> None:
    manifest_rows = read_jsonl(Path(args.manifest).resolve())
    manifest_by_dataset: Dict[str, List[dict]] = defaultdict(list)
    for row in manifest_rows:
        manifest_by_dataset[row["dataset"]].append(row)

    detections: List[dict] = []
    matches: List[dict] = []
    for judge, result_dir in args.run:
        for dataset in DATASETS:
            result_path = result_dir / f"{dataset}.csv"
            if not result_path.exists():
                raise FileNotFoundError(result_path)
            by_id = result_rows(result_path)
            for manifest in manifest_by_dataset[dataset]:
                key = str(manifest["story_id"])
                if key not in by_id:
                    raise ValueError(f"{judge}/{dataset}: missing story {key}")
                detection, matched = classify_one(judge, manifest, by_id[key])
                detections.append(detection)
                matches.extend(matched)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize(detections)
    write_csv(output_dir / "detections.csv", detections)
    write_csv(output_dir / "recall_summary.csv", summaries)
    write_jsonl(output_dir / "matched_errors.jsonl", matches)
    (output_dir / "recall_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(detections)} detection records -> {output_dir}")
    for row in summaries:
        if row["group_type"] != "overall":
            continue
        metric = "recall" if row["label"] == "positive" else "false-positive"
        print(
            f"{row['judge']} {metric}: n={row['n']}, "
            f"subtype={row['subtype_rate']:.3f}, "
            f"category={row['category_rate']:.3f}, "
            f"any={row['any_rate']:.3f}, "
            f"complete={row['complete_results']}/{row['n']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--repo-root", required=True)
    generate_parser.add_argument("--output-dir", required=True)
    generate_parser.add_argument("--seed", type=int, default=20260725)
    generate_parser.set_defaults(func=generate)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--manifest", required=True)
    score_parser.add_argument("--run", action="append", type=parse_run_arg, required=True)
    score_parser.add_argument("--output-dir", required=True)
    score_parser.set_defaults(func=score)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
