# Paper numbers: dispersion, paired tests, fixed subset, and global-subtype split

Generated: 2026-09-29 05:58 CST. Recomputed from the stored per-story results; nothing is re-judged.
CED is CED-original = E/W×10⁴. SE is the standard error of the mean; [ ] after a paired difference is the 95% bootstrap CI,
p is from a paired t-test, and `Holm` is the Holm-corrected p within one metric (across variants or lengths).

## 1. 10K: mean ± SE and paired tests against NstAgent

### DeepSeek-V4-Flash

| Method | Subtype CED | Instance CED | Writing Quality |
|---|---|---|---|
| NstAgent | 4.541 ± 0.190 (SD 1.899, n=100) | 6.392 ± 0.328 (SD 3.282, n=100) | 8.998 ± 0.059 (SD 0.595, n=100) |
| RollSum | 5.981 ± 0.181 (SD 1.813, n=100) | 9.171 ± 0.432 (SD 4.320, n=100) | 8.946 ± 0.053 (SD 0.529, n=100) |
| Direct | 6.225 ± 0.249 (SD 2.477, n=99) | 9.293 ± 0.512 (SD 5.098, n=99) | 8.705 ± 0.073 (SD 0.728, n=99) |
| StoryWriter | 7.395 ± 0.301 (SD 2.976, n=98) | 11.803 ± 0.651 (SD 6.444, n=98) | 7.353 ± 0.123 (SD 1.215, n=98) |
| DOME (20) | 9.607 ± 0.615 (SD 2.752, n=20) | 16.765 ± 1.434 (SD 6.412, n=20) | 4.870 ± 0.247 (SD 1.106, n=20) |

| Pair (NstAgent − baseline) | Metric | Difference [95% CI] | p (t) | Holm |
|---|---|---|---|---|
| NstAgent − RollSum | sub | -1.440 [-1.86, -1.03] | 1.1e-09 | 2.1e-09 |
| NstAgent − RollSum | ins | -2.779 [-3.65, -1.95] | 9.2e-09 | 1.8e-08 |
| NstAgent − RollSum | wb | +0.052 [-0.05, +0.15] | 0.31 | 0.31 |
| NstAgent − Direct | sub | -1.684 [-2.24, -1.13] | 3.8e-08 | 3.8e-08 |
| NstAgent − Direct | ins | -2.927 [-3.99, -1.89] | 2.8e-07 | 2.8e-07 |
| NstAgent − Direct | wb | +0.299 [+0.15, +0.44] | 0.00013 | 0.00026 |
| NstAgent − StoryWriter | sub | -2.852 [-3.44, -2.28] | 2.2e-15 | 6.7e-15 |
| NstAgent − StoryWriter | ins | -5.397 [-6.69, -4.17] | 7.8e-13 | 2.3e-12 |
| NstAgent − StoryWriter | wb | +1.639 [+1.42, +1.87] | 8.1e-26 | 2.4e-25 |

### GPT-5.6 Luna

| Method | Subtype CED | Instance CED | Writing Quality |
|---|---|---|---|
| NstAgent | 4.852 ± 0.193 (SD 1.931, n=100) | 6.826 ± 0.357 (SD 3.567, n=100) | 9.258 ± 0.040 (SD 0.396, n=100) |
| RollSum | 4.638 ± 0.222 (SD 2.223, n=100) | 6.653 ± 0.361 (SD 3.613, n=100) | 9.228 ± 0.043 (SD 0.428, n=100) |
| Direct | 5.428 ± 0.248 (SD 2.445, n=97) | 7.961 ± 0.434 (SD 4.276, n=97) | 9.233 ± 0.049 (SD 0.484, n=97) |
| StoryWriter | 5.163 ± 0.232 (SD 2.321, n=100) | 7.679 ± 0.464 (SD 4.639, n=100) | 8.408 ± 0.109 (SD 1.089, n=100) |
| DOME (20) | 6.861 ± 0.649 (SD 2.901, n=20) | 11.419 ± 1.545 (SD 6.909, n=20) | 6.950 ± 0.315 (SD 1.410, n=20) |

| Pair (NstAgent − baseline) | Metric | Difference [95% CI] | p (t) | Holm |
|---|---|---|---|---|
| NstAgent − RollSum | sub | +0.214 [-0.30, +0.70] | 0.4 | 0.49 |
| NstAgent − RollSum | ins | +0.173 [-0.69, +1.01] | 0.69 | 0.69 |
| NstAgent − RollSum | wb | +0.030 [-0.05, +0.11] | 0.44 | 0.89 |
| NstAgent − Direct | sub | -0.651 [-1.18, -0.13] | 0.017 | 0.051 |
| NstAgent − Direct | ins | -1.282 [-2.21, -0.33] | 0.008 | 0.024 |
| NstAgent − Direct | wb | +0.031 [-0.05, +0.12] | 0.48 | 0.89 |
| NstAgent − StoryWriter | sub | -0.311 [-0.83, +0.20] | 0.24 | 0.49 |
| NstAgent − StoryWriter | ins | -0.854 [-1.89, +0.11] | 0.1 | 0.21 |
| NstAgent − StoryWriter | wb | +0.850 [+0.66, +1.06] | 2.3e-13 | 6.9e-13 |

## 2. 10K–100K: mean ± SE, paired tests, and global-subtype split

### DeepSeek-V4-Flash

| Length | Method | Subtype CED | Instance CED | Writing Quality | Global subtype share | Local-only Instance CED |
|---|---|---|---|---|---|---|
| 10K | NstAgent | 4.541 ± 0.190 (SD 1.899, n=100) | 6.392 ± 0.328 (SD 3.282, n=100) | 8.998 ± 0.059 (SD 0.595, n=100) | 0.327 (5.1%) | 6.065 |
| 10K | RollSum | 5.981 ± 0.181 (SD 1.813, n=100) | 9.171 ± 0.432 (SD 4.320, n=100) | 8.946 ± 0.053 (SD 0.529, n=100) | 0.683 (7.5%) | 8.488 |
| 20K | NstAgent | 5.382 ± 0.197 (SD 1.972, n=100) | 8.350 ± 0.396 (SD 3.962, n=100) | 9.140 ± 0.047 (SD 0.472, n=100) | 0.533 (6.4%) | 7.818 |
| 20K | RollSum (last 8) | 6.761 ± 0.220 (SD 2.199, n=100) | 10.917 ± 0.470 (SD 4.699, n=100) | 8.986 ± 0.061 (SD 0.608, n=100) | 0.703 (6.4%) | 10.214 |
| 50K | NstAgent | 5.793 ± 0.213 (SD 2.132, n=100) | 8.906 ± 0.401 (SD 4.010, n=100) | 9.240 ± 0.048 (SD 0.482, n=100) | 0.766 (8.6%) | 8.140 |
| 50K | RollSum | 7.162 ± 0.230 (SD 2.255, n=96) | 11.458 ± 0.459 (SD 4.496, n=96) | 9.027 ± 0.052 (SD 0.513, n=96) | 1.211 (10.6%) | 10.247 |
| 100K | NstAgent | 4.859 ± 0.264 (SD 1.869, n=50) | 7.239 ± 0.493 (SD 3.488, n=50) | 9.204 ± 0.058 (SD 0.413, n=50) | 0.852 (11.8%) | 6.387 |
| 100K | RollSum | 7.239 ± 0.288 (SD 2.037, n=50) | 11.820 ± 0.657 (SD 4.647, n=50) | 8.964 ± 0.093 (SD 0.659, n=50) | 1.183 (10.0%) | 10.636 |

| Length | Metric | NstAgent − RollSum [95% CI] | p (t) | p (Wilcoxon) | Holm |
|---|---|---|---|---|---|
| 10K | sub | -1.440 [-1.86, -1.03] | 1.1e-09 | 1.5e-08 | 4.3e-09 | 
| 10K | ins | -2.779 [-3.65, -1.95] | 9.2e-09 | 6.9e-09 | 3.7e-08 | 
| 10K | ins_local | -2.423 [-3.28, -1.61] | 1.8e-07 | 1.3e-07 | 7e-07 | 
| 10K | wb | +0.052 [-0.05, +0.15] | 0.31 | 0.095 | 0.31 | 
| 20K | sub | -1.378 [-1.83, -0.94] | 4.1e-08 | 1.4e-07 | 1.2e-07 | 
| 20K | ins | -2.567 [-3.52, -1.63] | 7.9e-07 | 1.5e-06 | 2.4e-06 | 
| 20K | ins_local | -2.396 [-3.31, -1.48] | 1.6e-06 | 1.5e-06 | 4.7e-06 | 
| 20K | wb | +0.154 [+0.05, +0.26] | 0.0053 | 0.0016 | 0.011 | 
| 50K | sub | -1.429 [-1.91, -0.96] | 6.6e-08 | 5.7e-07 | 1.3e-07 | 
| 50K | ins | -2.594 [-3.58, -1.61] | 1.6e-06 | 7.9e-06 | 3.3e-06 | 
| 50K | ins_local | -2.151 [-3.10, -1.22] | 2.6e-05 | 5.9e-05 | 2.9e-05 | 
| 50K | wb | +0.215 [+0.14, +0.29] | 2.4e-07 | 2.2e-07 | 9.5e-07 | 
| 100K | sub | -2.381 [-3.20, -1.59] | 4.4e-07 | 5.2e-07 | 4.4e-07 | 
| 100K | ins | -4.581 [-6.37, -2.85] | 4.7e-06 | 2.2e-06 | 4.7e-06 | 
| 100K | ins_local | -4.250 [-6.01, -2.54] | 1.5e-05 | 8.4e-06 | 2.9e-05 | 
| 100K | wb | +0.240 [+0.09, +0.40] | 0.0036 | 0.0045 | 0.011 | 

### GPT-5.6 Luna

| Length | Method | Subtype CED | Instance CED | Writing Quality | Global subtype share | Local-only Instance CED |
|---|---|---|---|---|---|---|
| 10K | NstAgent | 4.852 ± 0.193 (SD 1.931, n=100) | 6.826 ± 0.357 (SD 3.567, n=100) | 9.258 ± 0.040 (SD 0.396, n=100) | 1.022 (15.0%) | 5.804 |
| 10K | RollSum | 4.638 ± 0.222 (SD 2.223, n=100) | 6.653 ± 0.361 (SD 3.613, n=100) | 9.228 ± 0.043 (SD 0.428, n=100) | 0.858 (12.9%) | 5.794 |
| 20K | NstAgent | 5.193 ± 0.216 (SD 2.158, n=100) | 7.698 ± 0.364 (SD 3.636, n=100) | 9.216 ± 0.040 (SD 0.399, n=100) | 1.057 (13.7%) | 6.642 |
| 20K | RollSum | 6.057 ± 0.259 (SD 2.588, n=100) | 8.972 ± 0.443 (SD 4.426, n=100) | 9.166 ± 0.045 (SD 0.446, n=100) | 1.194 (13.3%) | 7.778 |
| 50K | NstAgent | 5.271 ± 0.207 (SD 2.071, n=100) | 7.695 ± 0.346 (SD 3.464, n=100) | 9.210 ± 0.039 (SD 0.387, n=100) | 1.549 (20.1%) | 6.146 |
| 50K | RollSum | 6.014 ± 0.221 (SD 2.210, n=100) | 9.095 ± 0.430 (SD 4.299, n=100) | 9.196 ± 0.042 (SD 0.421, n=100) | 1.668 (18.3%) | 7.427 |
| 100K | NstAgent | 4.887 ± 0.333 (SD 2.357, n=50) | 7.218 ± 0.522 (SD 3.689, n=50) | 9.244 ± 0.074 (SD 0.526, n=50) | 1.904 (26.4%) | 5.314 |
| 100K | RollSum | 5.765 ± 0.320 (SD 2.265, n=50) | 9.369 ± 0.710 (SD 5.021, n=50) | 9.064 ± 0.068 (SD 0.479, n=50) | 1.939 (20.7%) | 7.430 |

| Length | Metric | NstAgent − RollSum [95% CI] | p (t) | p (Wilcoxon) | Holm |
|---|---|---|---|---|---|
| 10K | sub | +0.214 [-0.30, +0.70] | 0.4 | 0.36 | 0.4 | 
| 10K | ins | +0.173 [-0.69, +1.01] | 0.69 | 0.59 | 0.69 | 
| 10K | ins_local | +0.009 [-0.85, +0.84] | 0.98 | 0.88 | 0.98 | 
| 10K | wb | +0.030 [-0.05, +0.11] | 0.44 | 0.43 | 0.89 | 
| 20K | sub | -0.864 [-1.38, -0.35] | 0.0018 | 0.0031 | 0.007 | 
| 20K | ins | -1.274 [-2.11, -0.44] | 0.0039 | 0.0049 | 0.012 | 
| 20K | ins_local | -1.137 [-1.93, -0.34] | 0.0076 | 0.013 | 0.02 | 
| 20K | wb | +0.050 [-0.01, +0.11] | 0.13 | 0.12 | 0.38 | 
| 50K | sub | -0.742 [-1.27, -0.22] | 0.0076 | 0.047 | 0.023 | 
| 50K | ins | -1.401 [-2.34, -0.49] | 0.0038 | 0.011 | 0.012 | 
| 50K | ins_local | -1.281 [-2.19, -0.38] | 0.0066 | 0.012 | 0.02 | 
| 50K | wb | +0.014 [-0.05, +0.09] | 0.7 | 0.79 | 0.89 | 
| 100K | sub | -0.877 [-1.68, -0.06] | 0.04 | 0.076 | 0.079 | 
| 100K | ins | -2.151 [-3.47, -0.85] | 0.0028 | 0.0045 | 0.011 | 
| 100K | ins_local | -2.116 [-3.43, -0.81] | 0.0028 | 0.0043 | 0.011 | 
| 100K | wb | +0.180 [+0.10, +0.26] | 6.7e-05 | 8.8e-05 | 0.00027 | 

## 3. Length trend on the fixed 50-prompt subset (the 100K sample)

Subset size: 50 ids. At every length only stories in this subset are kept, so the sample composition does not change.

### DeepSeek-V4-Flash

| Length | Method | n | Subtype CED | Instance CED | Writing Quality |
|---|---|---:|---|---|---|
| 10K | NstAgent | 50 | 4.642 ± 0.264 (SD 1.867, n=50) | 6.704 ± 0.486 (SD 3.435, n=50) | 8.980 ± 0.064 (SD 0.456, n=50) |
| 10K | RollSum | 50 | 5.923 ± 0.241 (SD 1.702, n=50) | 9.308 ± 0.569 (SD 4.026, n=50) | 8.916 ± 0.079 (SD 0.556, n=50) |
| 20K | NstAgent | 50 | 5.342 ± 0.270 (SD 1.912, n=50) | 8.401 ± 0.538 (SD 3.801, n=50) | 9.108 ± 0.073 (SD 0.519, n=50) |
| 20K | RollSum (last 8) | 50 | 6.666 ± 0.278 (SD 1.966, n=50) | 10.885 ± 0.579 (SD 4.091, n=50) | 9.068 ± 0.061 (SD 0.429, n=50) |
| 50K | NstAgent | 50 | 5.867 ± 0.291 (SD 2.056, n=50) | 8.866 ± 0.538 (SD 3.802, n=50) | 9.140 ± 0.074 (SD 0.524, n=50) |
| 50K | RollSum | 49 | 7.070 ± 0.334 (SD 2.337, n=49) | 11.666 ± 0.656 (SD 4.589, n=49) | 9.016 ± 0.076 (SD 0.532, n=49) |
| 100K | NstAgent | 50 | 4.859 ± 0.264 (SD 1.869, n=50) | 7.239 ± 0.493 (SD 3.488, n=50) | 9.204 ± 0.058 (SD 0.413, n=50) |
| 100K | RollSum | 50 | 7.239 ± 0.288 (SD 2.037, n=50) | 11.820 ± 0.657 (SD 4.647, n=50) | 8.964 ± 0.093 (SD 0.659, n=50) |

| Length | Metric | NstAgent − RollSum [95% CI] | p (t) | Holm |
|---|---|---|---|---|
| 10K | sub | -1.281 [-1.88, -0.70] | 0.00012 | 0.00035 |
| 10K | ins | -2.603 [-3.89, -1.32] | 0.00025 | 0.00075 |
| 10K | wb | +0.064 [-0.04, +0.16] | 0.23 | 0.46 |
| 20K | sub | -1.324 [-1.95, -0.72] | 0.00017 | 0.00035 |
| 20K | ins | -2.483 [-3.74, -1.24] | 0.00039 | 0.00077 |
| 20K | wb | +0.040 [-0.09, +0.16] | 0.55 | 0.55 |
| 50K | sub | -1.163 [-1.80, -0.53] | 0.00087 | 0.00087 |
| 50K | ins | -2.760 [-4.13, -1.37] | 0.00042 | 0.00077 |
| 50K | wb | +0.127 [+0.02, +0.24] | 0.029 | 0.087 |
| 100K | sub | -2.381 [-3.20, -1.59] | 4.4e-07 | 1.8e-06 |
| 100K | ins | -4.581 [-6.37, -2.85] | 4.7e-06 | 1.9e-05 |
| 100K | wb | +0.240 [+0.09, +0.40] | 0.0036 | 0.014 |

### GPT-5.6 Luna

| Length | Method | n | Subtype CED | Instance CED | Writing Quality |
|---|---|---:|---|---|---|
| 10K | NstAgent | 50 | 4.899 ± 0.243 (SD 1.715, n=50) | 6.601 ± 0.435 (SD 3.076, n=50) | 9.276 ± 0.054 (SD 0.383, n=50) |
| 10K | RollSum | 50 | 4.304 ± 0.290 (SD 2.054, n=50) | 6.125 ± 0.457 (SD 3.232, n=50) | 9.256 ± 0.069 (SD 0.487, n=50) |
| 20K | NstAgent | 50 | 5.523 ± 0.294 (SD 2.076, n=50) | 8.257 ± 0.496 (SD 3.507, n=50) | 9.260 ± 0.062 (SD 0.441, n=50) |
| 20K | RollSum | 50 | 6.435 ± 0.361 (SD 2.554, n=50) | 9.551 ± 0.624 (SD 4.415, n=50) | 9.180 ± 0.072 (SD 0.508, n=50) |
| 50K | NstAgent | 50 | 4.894 ± 0.307 (SD 2.171, n=50) | 6.916 ± 0.450 (SD 3.179, n=50) | 9.228 ± 0.049 (SD 0.348, n=50) |
| 50K | RollSum | 50 | 6.124 ± 0.303 (SD 2.144, n=50) | 8.946 ± 0.594 (SD 4.197, n=50) | 9.216 ± 0.053 (SD 0.377, n=50) |
| 100K | NstAgent | 50 | 4.887 ± 0.333 (SD 2.357, n=50) | 7.218 ± 0.522 (SD 3.689, n=50) | 9.244 ± 0.074 (SD 0.526, n=50) |
| 100K | RollSum | 50 | 5.765 ± 0.320 (SD 2.265, n=50) | 9.369 ± 0.710 (SD 5.021, n=50) | 9.064 ± 0.068 (SD 0.479, n=50) |

| Length | Metric | NstAgent − RollSum [95% CI] | p (t) | Holm |
|---|---|---|---|---|
| 10K | sub | +0.595 [-0.13, +1.26] | 0.099 | 0.099 |
| 10K | ins | +0.476 [-0.75, +1.65] | 0.44 | 0.44 |
| 10K | wb | +0.020 [-0.09, +0.13] | 0.73 | 1 |
| 20K | sub | -0.913 [-1.64, -0.17] | 0.02 | 0.061 |
| 20K | ins | -1.294 [-2.48, -0.12] | 0.041 | 0.082 |
| 20K | wb | +0.080 [-0.02, +0.18] | 0.12 | 0.37 |
| 50K | sub | -1.230 [-1.99, -0.49] | 0.0021 | 0.0083 |
| 50K | ins | -2.030 [-3.26, -0.83] | 0.002 | 0.008 |
| 50K | wb | +0.012 [-0.08, +0.10] | 0.8 | 1 |
| 100K | sub | -0.877 [-1.68, -0.06] | 0.04 | 0.079 |
| 100K | ins | -2.151 [-3.47, -0.85] | 0.0028 | 0.0083 |
| 100K | wb | +0.180 [+0.10, +0.26] | 6.7e-05 | 0.00027 |

## 4. 20K ablation (DeepSeek-V4-Flash, terminal window of 7 chapters; RollSum with 8 chapters as reference)

| Variant | Subtype CED | Instance CED | Writing Quality |
|---|---|---|---|
| NstAgent | 5.382 ± 0.197 (SD 1.972, n=100) | 8.350 ± 0.396 (SD 3.962, n=100) | 9.140 ± 0.047 (SD 0.472, n=100) |
| -State | 5.902 ± 0.190 (SD 1.898, n=100) | 9.400 ± 0.460 (SD 4.597, n=100) | 9.046 ± 0.044 (SD 0.441, n=100) |
| -Lookback | 6.577 ± 0.216 (SD 2.157, n=100) | 9.786 ± 0.387 (SD 3.865, n=100) | 8.980 ± 0.049 (SD 0.492, n=100) |
| RollSum (last 8) | 6.761 ± 0.220 (SD 2.199, n=100) | 10.917 ± 0.470 (SD 4.699, n=100) | 8.986 ± 0.061 (SD 0.608, n=100) |

| Pair (variant − full) | Metric | Difference [95% CI] | p (t) | Holm |
|---|---|---|---|---|
| -State − NstAgent | sub | +0.520 [+0.10, +0.94] | 0.017 | 0.017 |
| -State − NstAgent | ins | +1.050 [+0.16, +1.95] | 0.024 | 0.024 |
| -State − NstAgent | wb | -0.094 [-0.17, -0.02] | 0.018 | 0.018 |
| -Lookback − NstAgent | sub | +1.195 [+0.74, +1.65] | 2.3e-06 | 4.5e-06 |
| -Lookback − NstAgent | ins | +1.436 [+0.61, +2.23] | 0.00096 | 0.0019 |
| -Lookback − NstAgent | wb | -0.160 [-0.23, -0.09] | 3.6e-05 | 0.00011 |
| RollSum (last 8) − NstAgent | sub | +1.378 [+0.94, +1.83] | 4.1e-08 | 1.2e-07 |
| RollSum (last 8) − NstAgent | ins | +2.567 [+1.63, +3.52] | 7.9e-07 | 2.4e-06 |
| RollSum (last 8) − NstAgent | wb | -0.154 [-0.26, -0.05] | 0.0053 | 0.011 |

## 5. Instance CED by error category and subtype (Appendix C.1)

Computed per story as errors / checked words × 10⁴, then averaged; the category values sum to the Instance CED of Tables 2 and 3.

### 10K, DeepSeek-V4-Flash

| Error category | Direct | DOME (20) | StoryWriter | RollSum | NstAgent |
|---|---|---|---|---|---|
| characterization | 1.30 | 2.58 | 1.30 | 1.33 | 0.75 |
| factual_detail | 2.76 | 3.71 | 2.93 | 2.85 | 2.16 |
| narrative_style | 0.42 | 2.08 | 1.29 | 0.52 | 0.08 |
| timeline_plot | 3.30 | 6.10 | 4.54 | 2.98 | 2.10 |
| world_building | 1.51 | 2.29 | 1.74 | 1.49 | 1.30 |
| Total | 9.293 | 16.765 | 11.803 | 9.171 | 6.392 |

| Subtype | Direct | DOME (20) | StoryWriter | RollSum | NstAgent |
|---|---|---|---|---|---|
| characterization_memory_contradictions | 1.10 | 2.17 | 1.05 | 1.18 | 0.64 |
| characterization_knowledge_contradictions | 0.09 | 0.10 | 0.13 | 0.09 | 0.06 |
| characterization_skill_power_fluctuations | 0.12 | 0.31 | 0.11 | 0.06 | 0.05 |
| characterization_forgotten_abilities | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 |
| factual_detail_appearance_mismatches | 0.73 | 1.12 | 1.00 | 0.89 | 0.62 |
| factual_detail_nomenclature_confusions | 0.47 | 1.57 | 0.64 | 0.39 | 0.28 |
| factual_detail_quantitative_mismatches | 1.56 | 1.02 | 1.29 | 1.56 | 1.26 |
| narrative_style_perspective_confusions | 0.13 | 0.35 | 0.18 | 0.13 | 0.03 |
| narrative_style_tone_inconsistencies | 0.04 | 0.05 | 0.06 | 0.01 | 0.01 |
| narrative_style_style_shifts | 0.25 | 1.67 | 1.05 | 0.38 | 0.04 |
| timeline_plot_absolute_time_contradictions | 0.31 | 0.25 | 0.24 | 0.25 | 0.21 |
| timeline_plot_duration_timeline_contradictions | 0.99 | 1.02 | 0.77 | 0.91 | 0.71 |
| timeline_plot_simultaneity_contradictions | 0.23 | 0.31 | 0.28 | 0.23 | 0.12 |
| timeline_plot_causeless_effects | 0.26 | 0.46 | 0.59 | 0.20 | 0.22 |
| timeline_plot_causal_logic_violations | 0.88 | 2.08 | 1.26 | 0.71 | 0.53 |
| timeline_plot_abandoned_plot_elements | 0.63 | 1.98 | 1.38 | 0.68 | 0.33 |
| world_building_core_rules_violations | 1.01 | 1.22 | 1.07 | 0.90 | 0.83 |
| world_building_social_norms_violations | 0.19 | 0.35 | 0.23 | 0.18 | 0.18 |
| world_building_geographical_contradictions | 0.31 | 0.71 | 0.44 | 0.41 | 0.29 |

NstAgent is the lowest of the five methods on 15 of the 19 subtypes (unrounded values).

### 10K, GPT-5.6 Luna

| Error category | Direct | DOME (20) | StoryWriter | RollSum | NstAgent |
|---|---|---|---|---|---|
| characterization | 0.92 | 1.80 | 0.79 | 0.64 | 0.76 |
| factual_detail | 2.52 | 3.10 | 2.26 | 2.34 | 2.11 |
| narrative_style | 0.03 | 1.25 | 0.56 | 0.18 | 0.26 |
| timeline_plot | 3.10 | 4.10 | 3.00 | 2.52 | 2.67 |
| world_building | 1.39 | 1.16 | 1.08 | 0.97 | 1.03 |
| Total | 7.961 | 11.419 | 7.679 | 6.653 | 6.826 |

| Subtype | Direct | DOME (20) | StoryWriter | RollSum | NstAgent |
|---|---|---|---|---|---|
| characterization_memory_contradictions | 0.82 | 1.65 | 0.66 | 0.53 | 0.68 |
| characterization_knowledge_contradictions | 0.01 | 0.05 | 0.01 | 0.03 | 0.00 |
| characterization_skill_power_fluctuations | 0.09 | 0.10 | 0.12 | 0.08 | 0.07 |
| characterization_forgotten_abilities | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| factual_detail_appearance_mismatches | 0.57 | 0.95 | 0.97 | 0.93 | 0.69 |
| factual_detail_nomenclature_confusions | 0.40 | 1.20 | 0.59 | 0.42 | 0.59 |
| factual_detail_quantitative_mismatches | 1.55 | 0.95 | 0.70 | 0.99 | 0.82 |
| narrative_style_perspective_confusions | 0.01 | 0.50 | 0.06 | 0.01 | 0.08 |
| narrative_style_tone_inconsistencies | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 |
| narrative_style_style_shifts | 0.02 | 0.75 | 0.49 | 0.17 | 0.17 |
| timeline_plot_absolute_time_contradictions | 0.28 | 0.25 | 0.17 | 0.20 | 0.24 |
| timeline_plot_duration_timeline_contradictions | 0.89 | 0.85 | 0.47 | 0.53 | 0.44 |
| timeline_plot_simultaneity_contradictions | 0.31 | 0.35 | 0.23 | 0.13 | 0.16 |
| timeline_plot_causeless_effects | 0.26 | 0.30 | 0.31 | 0.33 | 0.33 |
| timeline_plot_causal_logic_violations | 0.66 | 1.20 | 0.73 | 0.48 | 0.49 |
| timeline_plot_abandoned_plot_elements | 0.71 | 1.15 | 1.09 | 0.86 | 1.02 |
| world_building_core_rules_violations | 0.90 | 0.45 | 0.62 | 0.63 | 0.62 |
| world_building_social_norms_violations | 0.17 | 0.35 | 0.23 | 0.13 | 0.21 |
| world_building_geographical_contradictions | 0.33 | 0.35 | 0.23 | 0.22 | 0.20 |

NstAgent is the lowest of the five methods on 4 of the 19 subtypes (unrounded values).

### 20K–100K, DeepSeek-V4-Flash

| Error category | 20K NstAgent | 20K RollSum | 50K NstAgent | 50K RollSum | 100K NstAgent | 100K RollSum |
|---|---|---|---|---|---|---|
| characterization | 1.25 | 1.79 | 1.41 | 2.16 | 1.19 | 2.12 |
| factual_detail | 2.51 | 3.23 | 2.74 | 3.36 | 2.22 | 3.80 |
| narrative_style | 0.31 | 0.86 | 0.26 | 0.53 | 0.25 | 0.74 |
| timeline_plot | 2.69 | 3.15 | 2.93 | 3.62 | 2.42 | 3.55 |
| world_building | 1.59 | 1.88 | 1.57 | 1.79 | 1.16 | 1.61 |

| Subtype | 20K NstAgent | 20K RollSum | 50K NstAgent | 50K RollSum | 100K NstAgent | 100K RollSum |
|---|---|---|---|---|---|---|
| characterization_memory_contradictions | 1.07 | 1.62 | 1.29 | 2.01 | 1.05 | 2.09 |
| characterization_knowledge_contradictions | 0.06 | 0.09 | 0.06 | 0.05 | 0.08 | 0.02 |
| characterization_skill_power_fluctuations | 0.11 | 0.08 | 0.06 | 0.09 | 0.06 | 0.00 |
| characterization_forgotten_abilities | 0.02 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 |
| factual_detail_appearance_mismatches | 0.72 | 1.08 | 0.65 | 1.14 | 0.65 | 1.32 |
| factual_detail_nomenclature_confusions | 0.22 | 0.48 | 0.35 | 0.67 | 0.19 | 0.75 |
| factual_detail_quantitative_mismatches | 1.58 | 1.68 | 1.74 | 1.55 | 1.38 | 1.72 |
| narrative_style_perspective_confusions | 0.07 | 0.31 | 0.06 | 0.11 | 0.02 | 0.15 |
| narrative_style_tone_inconsistencies | 0.02 | 0.05 | 0.00 | 0.04 | 0.00 | 0.06 |
| narrative_style_style_shifts | 0.22 | 0.50 | 0.20 | 0.37 | 0.23 | 0.53 |
| timeline_plot_absolute_time_contradictions | 0.26 | 0.26 | 0.31 | 0.33 | 0.21 | 0.34 |
| timeline_plot_duration_timeline_contradictions | 0.94 | 0.99 | 0.90 | 0.86 | 0.93 | 1.08 |
| timeline_plot_simultaneity_contradictions | 0.16 | 0.18 | 0.22 | 0.24 | 0.09 | 0.24 |
| timeline_plot_causeless_effects | 0.14 | 0.22 | 0.16 | 0.28 | 0.02 | 0.06 |
| timeline_plot_causal_logic_violations | 0.65 | 0.79 | 0.57 | 0.69 | 0.32 | 0.65 |
| timeline_plot_abandoned_plot_elements | 0.53 | 0.70 | 0.77 | 1.21 | 0.85 | 1.18 |
| world_building_core_rules_violations | 1.03 | 1.03 | 0.91 | 0.86 | 0.56 | 0.67 |
| world_building_social_norms_violations | 0.15 | 0.24 | 0.17 | 0.20 | 0.02 | 0.36 |
| world_building_geographical_contradictions | 0.41 | 0.61 | 0.50 | 0.74 | 0.58 | 0.58 |

Subtypes where NstAgent is lower (20K / 50K / 100K): 17 / 15 / 15; error categories where it is lower: 15 / 15.

### 20K–100K, GPT-5.6 Luna

| Error category | 20K NstAgent | 20K RollSum | 50K NstAgent | 50K RollSum | 100K NstAgent | 100K RollSum |
|---|---|---|---|---|---|---|
| characterization | 1.07 | 1.14 | 0.88 | 1.10 | 0.94 | 1.48 |
| factual_detail | 2.56 | 2.76 | 2.63 | 2.86 | 2.26 | 2.74 |
| narrative_style | 0.14 | 0.38 | 0.13 | 0.34 | 0.22 | 0.30 |
| timeline_plot | 2.67 | 3.20 | 3.00 | 3.34 | 2.97 | 3.54 |
| world_building | 1.25 | 1.48 | 1.05 | 1.45 | 0.83 | 1.30 |

| Subtype | 20K NstAgent | 20K RollSum | 50K NstAgent | 50K RollSum | 100K NstAgent | 100K RollSum |
|---|---|---|---|---|---|---|
| characterization_memory_contradictions | 0.90 | 1.02 | 0.83 | 0.99 | 0.84 | 1.35 |
| characterization_knowledge_contradictions | 0.05 | 0.02 | 0.01 | 0.03 | 0.02 | 0.04 |
| characterization_skill_power_fluctuations | 0.12 | 0.11 | 0.04 | 0.08 | 0.08 | 0.09 |
| characterization_forgotten_abilities | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| factual_detail_appearance_mismatches | 0.98 | 1.11 | 1.15 | 1.04 | 0.80 | 0.79 |
| factual_detail_nomenclature_confusions | 0.57 | 0.70 | 0.59 | 0.87 | 0.76 | 0.77 |
| factual_detail_quantitative_mismatches | 1.01 | 0.96 | 0.89 | 0.95 | 0.70 | 1.17 |
| narrative_style_perspective_confusions | 0.09 | 0.10 | 0.05 | 0.06 | 0.02 | 0.00 |
| narrative_style_tone_inconsistencies | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| narrative_style_style_shifts | 0.06 | 0.26 | 0.08 | 0.28 | 0.20 | 0.30 |
| timeline_plot_absolute_time_contradictions | 0.23 | 0.21 | 0.10 | 0.25 | 0.17 | 0.27 |
| timeline_plot_duration_timeline_contradictions | 0.53 | 0.57 | 0.42 | 0.51 | 0.31 | 0.54 |
| timeline_plot_simultaneity_contradictions | 0.19 | 0.26 | 0.17 | 0.16 | 0.15 | 0.10 |
| timeline_plot_causeless_effects | 0.25 | 0.47 | 0.33 | 0.24 | 0.13 | 0.18 |
| timeline_plot_causal_logic_violations | 0.42 | 0.51 | 0.43 | 0.51 | 0.30 | 0.52 |
| timeline_plot_abandoned_plot_elements | 1.06 | 1.19 | 1.55 | 1.67 | 1.90 | 1.94 |
| world_building_core_rules_violations | 0.71 | 0.87 | 0.57 | 0.73 | 0.33 | 0.55 |
| world_building_social_norms_violations | 0.24 | 0.22 | 0.14 | 0.32 | 0.15 | 0.26 |
| world_building_geographical_contradictions | 0.31 | 0.40 | 0.34 | 0.40 | 0.35 | 0.48 |

Subtypes where NstAgent is lower (20K / 50K / 100K): 13 / 14 / 14; error categories where it is lower: 15 / 15.

### 20K ablation (DeepSeek-V4-Flash)

| Error category | NstAgent | -State | -Lookback | RollSum (last 8) |
|---|---|---|---|---|
| characterization | 1.25 | 1.61 | 1.52 | 1.79 |
| factual_detail | 2.51 | 2.62 | 2.89 | 3.23 |
| narrative_style | 0.31 | 0.43 | 0.50 | 0.86 |
| timeline_plot | 2.69 | 3.24 | 3.16 | 3.15 |
| world_building | 1.59 | 1.51 | 1.71 | 1.88 |
| Total | 8.350 | 9.400 | 9.786 | 10.917 |

| Subtype | NstAgent | -State | -Lookback | RollSum (last 8) |
|---|---|---|---|---|
| characterization_memory_contradictions | 1.07 | 1.49 | 1.40 | 1.62 |
| characterization_knowledge_contradictions | 0.06 | 0.06 | 0.05 | 0.09 |
| characterization_skill_power_fluctuations | 0.11 | 0.06 | 0.08 | 0.08 |
| characterization_forgotten_abilities | 0.02 | 0.00 | 0.00 | 0.01 |
| factual_detail_appearance_mismatches | 0.72 | 0.64 | 0.72 | 1.08 |
| factual_detail_nomenclature_confusions | 0.22 | 0.43 | 0.48 | 0.48 |
| factual_detail_quantitative_mismatches | 1.58 | 1.56 | 1.69 | 1.68 |
| narrative_style_perspective_confusions | 0.07 | 0.06 | 0.17 | 0.31 |
| narrative_style_tone_inconsistencies | 0.02 | 0.02 | 0.04 | 0.05 |
| narrative_style_style_shifts | 0.22 | 0.35 | 0.29 | 0.50 |
| timeline_plot_absolute_time_contradictions | 0.26 | 0.25 | 0.36 | 0.26 |
| timeline_plot_duration_timeline_contradictions | 0.94 | 0.93 | 1.10 | 0.99 |
| timeline_plot_simultaneity_contradictions | 0.16 | 0.15 | 0.17 | 0.18 |
| timeline_plot_causeless_effects | 0.14 | 0.23 | 0.23 | 0.22 |
| timeline_plot_causal_logic_violations | 0.65 | 0.83 | 0.68 | 0.79 |
| timeline_plot_abandoned_plot_elements | 0.53 | 0.84 | 0.62 | 0.70 |
| world_building_core_rules_violations | 1.03 | 0.93 | 0.92 | 1.03 |
| world_building_social_norms_violations | 0.15 | 0.16 | 0.16 | 0.24 |
| world_building_geographical_contradictions | 0.41 | 0.42 | 0.64 | 0.61 |

NstAgent is the lowest on 9 of the 19 subtypes (unrounded values).
