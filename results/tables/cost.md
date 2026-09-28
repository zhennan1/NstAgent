# Generation cost per story on DeepSeek-V4-Flash

Recomputed by `code/analysis/cost_table.py` from the per-call usage logs in `results/usage/deepseek/`.
Sample: the first five prompts of `data/prompts_en.jsonl` (ids 1994, 1996, 1993, 1712, 1814), generated
with the settings of the main runs and one story per job. Each cell is the mean per completed story; where
some stories did not complete, the mean over all finished stories follows in parentheses. Output tokens
include reasoning tokens. Cost uses off-peak prices in USD per million tokens: 0.22 for uncached input, 0.007 for cached input, 0.66 for output.

| Method | Length | Complete/finished | Calls | Input tokens | Cached input | Output tokens | Reasoning tokens | Minutes | Cost (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 10K | 5/5 | 2 | 8,586 | 0 | 32,466 | 12,971 | 4.0 | 0.02 |
| StoryWriter | 10K | 4/5 | 26 (39) | 108,018 (239,969) | 62,336 (134,605) | 91,228 (102,824) | 49,197 (55,936) | 10.6 (12.3) | 0.07 (0.09) |
| DOME | 10K | 5/5 | 3,501 | 1,719,146 | 154 | 580,429 | 76,904 | 43.4 | 0.76 |
| RollSum | 10K | 5/5 | 40 | 209,725 | 86,733 | 252,195 | 216,659 | 23.7 | 0.19 |
| RollSum | 20K | 5/5 | 56 | 386,520 | 162,560 | 472,955 | 396,240 | 47.8 | 0.36 |
| RollSum | 50K | 3/5 | 93 (95) | 867,189 (1,008,624) | 383,147 (513,434) | 775,659 (888,992) | 569,363 (596,733) | 82.2 (90.9) | 0.62 (0.70) |
| RollSum | 100K | 3/5 | 171 (145) | 2,314,500 (2,033,908) | 1,162,240 (1,075,661) | 1,710,584 (1,555,023) | 1,181,995 (1,028,126) | 175.8 (158.1) | 1.39 (1.24) |
| NstAgent | 10K | 5/5 | 68 | 711,689 | 412,979 | 159,084 | 109,680 | 19.0 | 0.17 |
| NstAgent | 20K | 5/5 | 104 | 1,466,085 | 882,790 | 273,029 | 169,788 | 32.6 | 0.31 |
| NstAgent | 50K | 5/5 | 161 | 3,174,102 | 1,856,614 | 507,352 | 322,233 | 61.1 | 0.64 |
| NstAgent | 100K | 5/5 | 259 | 7,499,367 | 4,495,155 | 953,223 | 580,844 | 112.8 | 1.32 |

Notes:

- DOME keeps its released configuration: format conversion, the five-act and hierarchical outlines, and the
  knowledge-graph memory calls run with reasoning disabled, and chapter writing with reasoning enabled.
  Every other method uses the provider's default, reasoning enabled.
- Five stories per arm, so these are rough estimates. The price per 10K words written by NstAgent is the
  cost divided by the target length in units of 10K words.
