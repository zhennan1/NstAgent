# Judge recall of the ConStory judge (DeepSeek-V4-Pro) under controlled error injection

Generated: 2026-09-29 05:58 CST; evaluated 144/144 (120 positives, 24 consistent controls).

**Question**: a lower CED at 100K could reflect the judge's recall decaying over a ~90K-word prefix rather than better stories.
**Method**: a pair of explicit, self-contained statements is injected into real NstAgent stories; the later statement **always lands inside the terminal window** (whole story at 10K; last 7/5/4 chapters at 20K/50K/100K), and the earlier one is placed at one of three prefix depths. The judge and prompt are identical to the main evaluation.
Each injection carries a unique marker `QX-XXXXXXXX` that occurs exactly twice in its story, so detection is an exact string match.

## 1. By length (main result)

| Length | Target-subtype recall | Target-category recall | Detected in any category | Controls reported as target type | Controls in any category |
|---|---|---|---|---|---|
| 10K | 28/30 = 93.3% [79, 98] | 29/30 = 96.7% [83, 99] | 30/30 = 100.0% [89, 100] | 0/6 = 0.0% [0, 39] | 3/6 = 50.0% [19, 81] |
| 20K | 30/30 = 100.0% [89, 100] | 30/30 = 100.0% [89, 100] | 30/30 = 100.0% [89, 100] | 2/6 = 33.3% [10, 70] | 6/6 = 100.0% [61, 100] |
| 50K | 29/30 = 96.7% [83, 99] | 30/30 = 100.0% [89, 100] | 30/30 = 100.0% [89, 100] | 2/6 = 33.3% [10, 70] | 5/6 = 83.3% [44, 97] |
| 100K | 30/30 = 100.0% [89, 100] | 30/30 = 100.0% [89, 100] | 30/30 = 100.0% [89, 100] | 2/6 = 33.3% [10, 70] | 5/6 = 83.3% [44, 97] |

## 2. By prefix depth (within each length)

Depth is the number of words between the earlier and the later statement: `near` is just before the window, `far` is at the start of the story.

| Length | Depth | Mean gap (words) | Target-category recall | Detected in any category |
|---|---|---:|---|---|
| 10K | near | 0 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |
| 10K | mid | 4,649 | 9/10 = 90.0% [60, 98] | 10/10 = 100.0% [72, 100] |
| 10K | far | 9,318 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |
| 20K | near | 4,444 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |
| 20K | mid | 10,700 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |
| 20K | far | 15,329 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |
| 50K | near | 4,487 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |
| 50K | mid | 27,192 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |
| 50K | far | 46,753 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |
| 100K | near | 5,374 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |
| 100K | mid | 55,279 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |
| 100K | far | 100,978 | 10/10 = 100.0% [72, 100] | 10/10 = 100.0% [72, 100] |

## 3. Sources of control false triggers

Controls inject two **mutually consistent** statements, so any reported error is a false trigger. However, an inserted paragraph reads as foreign text, which the judge reports as `style_shifts`, and an inserted object that never returns is reported as `abandoned_plot_elements`. These two channels measure the injection method, not contradiction detection.

| Criterion | Controls | Positives |
|---|---|---|
| Reported in any category | 19/24 = 79.2% [60, 91] | 120/120 = 100.0% [97, 100] |
| **Reported as the target contradiction type** | 6/24 = 25.0% [12, 45] | 119/120 = 99.2% [95, 100] |
| Excluding the style-shift and abandoned-plot channels | 5/24 = 20.8% [9, 40] | 115/120 = 95.8% [91, 98] |

Most frequent subtypes among controls: `narrative_style_style_shifts` 18, `timeline_plot_abandoned_plot_elements` 10, `world_building_core_rules_violations` 3.
Positives are reported under 3.98 subtypes on average, controls under 1.50: positives are reported repeatedly across subtypes (taxonomy drift).

## 4. Trend tests

- **Target-category recall**: vs. length Kendall τ=+0.112 (p=0.18); vs. log10(gap words) τ=+0.056 (p=0.462).
- **Detected in any category**: all positives have the same outcome (all detected); no trend test needed.
- **10K+20K vs. 50K+100K** (target-category recall): 59/60 vs. 60/60, Fisher p=1.

## 5. Scope and limitations

- The injected statements are explicit and self-contained, so they are easier to detect than naturally occurring contradictions; this recall is an **upper bound**.
- The unique marker may act as a retrieval anchor and make distant injections easier to detect.
- The later statement always lies inside the terminal window, so a miss can only be due to the judge failing to recall the earlier statement from the prefix, not to the protocol excluding the error.
- The two control statements are mutually consistent and estimate the false-trigger rate; "reported as the target contradiction type" must be distinguished from "reported as a style shift because of insertion traces".
