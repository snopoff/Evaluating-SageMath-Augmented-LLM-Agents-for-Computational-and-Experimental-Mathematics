# Additional paired statistics

All differences are computed on the same 133 items. Intervals are percentile bootstrap intervals with paired resampling.

## Main results

- Mean gain: **8.2 percentage points**.
- Fixed evaluated models, resampling items: 95% CI **5.8 to 10.7 pp**.
- Fixed benchmark items, resampling models: 95% CI **5.7 to 11.1 pp**.
- Resampling both models and items independently: 95% CI **4.5 to 12.3 pp**.
- Positive model-level point estimates: **20/20**.
- Exact sign test across model signs: one-sided p = **9.54e-07**; two-sided p = **1.91e-06**.
- Model-level 95% paired-bootstrap intervals excluding zero: **k = 10**.
- Exact per-model McNemar tests below 0.05: **9 unadjusted**, **4 after Holm correction over 20 models**.
- Models retained by the 20-test Holm correction: **Opus 4.7, Qwen 3.7 Max, MiniMax M3, DeepSeek V3.2**.

The sign test treats the 20 configuration signs as exchangeable units. Because related model configurations share architectures and all models see the same items, it is best presented as a descriptive robustness check, not as the primary inferential result.

## Per-model paired gains

| Model | No tool | Agent | Gain (pp) | 95% paired-bootstrap CI (pp) | Exact McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Sol | 71.4 | 75.9 | 4.5 | [0.0, 9.0] | 0.109 | 1 |
| GPT-5.5 | 67.7 | 75.2 | 7.5 | [2.3, 13.5] | 0.0213 | 0.298 |
| Opus 4.8 | 66.2 | 73.7 | 7.5 | [2.3, 13.5] | 0.0213 | 0.298 |
| Opus 4.7 | 61.7 | 72.9 | 11.3 | [6.0, 17.3] | 0.000275 | 0.00494 |
| Fugu-Ultra | 66.9 | 72.9 | 6.0 | [-0.8, 12.8] | 0.115 | 1 |
| Gemini 3.1 Pro | 66.9 | 69.9 | 3.0 | [-1.5, 7.5] | 0.344 | 1 |
| Qwen 3.7 Max | 42.1 | 69.9 | 27.8 | [18.8, 36.8] | 2.46e-08 | 4.92e-07 |
| GPT-5.6 Terra | 65.4 | 67.7 | 2.3 | [-2.3, 7.5] | 0.549 | 1 |
| Gemini 3.5 Flash | 61.7 | 66.9 | 5.3 | [-0.8, 12.0] | 0.167 | 1 |
| Sonnet 5 | 59.4 | 66.9 | 7.5 | [0.8, 15.0] | 0.0639 | 0.703 |
| Kimi K3 | 60.2 | 64.7 | 4.5 | [-3.0, 12.0] | 0.327 | 1 |
| DeepSeek V4 Pro | 52.6 | 63.2 | 10.5 | [3.0, 18.0] | 0.00936 | 0.15 |
| GLM 5.2 | 53.4 | 62.4 | 9.0 | [1.5, 17.3] | 0.0428 | 0.513 |
| Kimi K2.7 | 57.1 | 58.6 | 1.5 | [-5.3, 8.3] | 0.824 | 1 |
| GPT-5.6 Luna | 51.9 | 56.4 | 4.5 | [-3.0, 12.0] | 0.307 | 1 |
| MiniMax M3 | 38.3 | 55.6 | 17.3 | [9.8, 24.8] | 1.52e-05 | 0.000289 |
| Mistral 3.5 | 48.9 | 51.1 | 2.3 | [-3.0, 8.3] | 0.607 | 1 |
| DeepSeek V3.2 | 33.1 | 48.9 | 15.8 | [7.5, 24.1] | 0.000324 | 0.00551 |
| Grok 4.3 | 39.1 | 42.9 | 3.8 | [-3.8, 11.3] | 0.442 | 1 |
| DeepSeek V4 Flash | 30.8 | 42.1 | 11.3 | [3.0, 19.5] | 0.0167 | 0.25 |

## Tool-enabled comparisons within the top group

A positive difference favors the first model.

| Contrast | Correct | Difference (pp) | 95% paired-bootstrap CI (pp) | Discordant (first/second) | Exact McNemar p | Holm p (3 tests) |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Sol - GPT-5.5 | 101/100 | 0.8 | [-5.3, 6.8] | 9/8 | 1 | 1 |
| GPT-5.6 Sol - Opus 4.8 | 101/98 | 2.3 | [-4.5, 9.0] | 13/10 | 0.678 | 1 |
| GPT-5.5 - Opus 4.8 | 100/98 | 1.5 | [-4.5, 7.5] | 9/7 | 0.804 | 1 |

None of the three top-group contrasts excludes zero. The ranking Sol > GPT-5.5 > Opus 4.8 is therefore descriptive on this benchmark; the paired data do not establish differences among them. Because this group was selected using the same observed scores, these are nominal, post-selection comparisons rather than confirmatory tests.
