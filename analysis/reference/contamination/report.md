# Contamination probes: report

Offline replay: saved probe scores; benchmark correlations use frozen headline decisions.

Corpus built 2026-09-19T10:21:40; groups: benchmark=88, same_vintage=40, fresh=40, famous=40.
Models: anthropic__claude-opus-4-8, google__gemini-3.1-pro-preview, openai__gpt-5.5, openrouter__deepseek_deepseek-v4-pro. Errors: anthropic__claude-opus-4-8=0, google__gemini-3.1-pro-preview=0, openai__gpt-5.5=0, openrouter__deepseek_deepseek-v4-pro=0.

Abstract-continuation coverage: existing sentence-boundary splits are retained; short or single-sentence abstracts use the word boundary nearest 40% of the text. This fallback retains 16 previously excluded papers without changing their source texts or the existing eligible prompts.

## 1. Verbatim-recall probes by group

ROUGE-L between the model continuation and the true text; `8-gram` = share of the model's 8-grams found in the target; `extractable` = share of papers with a verbatim run of >= 50 tokens; CI = 95% paper-level bootstrap of the mean ROUGE-L. AUROC compares the benchmark papers with a control group (0.5 = indistinguishable, 1.0 = every benchmark paper recalled better than every control paper).

### abstract (guided)

| model | group | n | ROUGE-L mean [CI] | median | 8-gram | extractable | refusals |
|---|---|---:|---|---:|---:|---:|---:|
| anthropic__claude-opus-4-8 | famous | 40 | 0.920 [0.844, 0.975] | 1.000 | 0.854 | 0.80 | 0.00 |
| anthropic__claude-opus-4-8 | benchmark | 88 | 0.300 [0.272, 0.332] | 0.268 | 0.055 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | same_vintage | 40 | 0.259 [0.225, 0.293] | 0.233 | 0.016 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | fresh | 40 | 0.220 [0.198, 0.245] | 0.213 | 0.014 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | AUROC benchmark vs same_vintage | | 0.588 | | | | |
| anthropic__claude-opus-4-8 | AUROC benchmark vs fresh | | 0.693 | | | | |
| google__gemini-3.1-pro-preview | famous | 40 | 0.855 [0.770, 0.932] | 0.989 | 0.770 | 0.65 | 0.00 |
| google__gemini-3.1-pro-preview | benchmark | 88 | 0.281 [0.254, 0.310] | 0.259 | 0.047 | 0.01 | 0.02 |
| google__gemini-3.1-pro-preview | same_vintage | 40 | 0.269 [0.230, 0.309] | 0.246 | 0.036 | 0.00 | 0.03 |
| google__gemini-3.1-pro-preview | fresh | 40 | 0.228 [0.196, 0.265] | 0.219 | 0.021 | 0.00 | 0.00 |
| google__gemini-3.1-pro-preview | AUROC benchmark vs same_vintage | | 0.523 | | | | |
| google__gemini-3.1-pro-preview | AUROC benchmark vs fresh | | 0.646 | | | | |
| openai__gpt-5.5 | famous | 40 | 0.791 [0.697, 0.877] | 0.975 | 0.701 | 0.62 | 0.00 |
| openai__gpt-5.5 | benchmark | 88 | 0.259 [0.238, 0.284] | 0.231 | 0.031 | 0.01 | 0.00 |
| openai__gpt-5.5 | same_vintage | 40 | 0.245 [0.222, 0.267] | 0.244 | 0.014 | 0.00 | 0.00 |
| openai__gpt-5.5 | fresh | 40 | 0.206 [0.182, 0.233] | 0.186 | 0.011 | 0.00 | 0.00 |
| openai__gpt-5.5 | AUROC benchmark vs same_vintage | | 0.497 | | | | |
| openai__gpt-5.5 | AUROC benchmark vs fresh | | 0.681 | | | | |
| openrouter__deepseek_deepseek-v4-pro | famous | 40 | 0.821 [0.729, 0.906] | 0.994 | 0.716 | 0.65 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | benchmark | 88 | 0.303 [0.274, 0.335] | 0.266 | 0.058 | 0.01 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | same_vintage | 40 | 0.283 [0.237, 0.340] | 0.242 | 0.088 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | fresh | 40 | 0.221 [0.191, 0.255] | 0.184 | 0.012 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | AUROC benchmark vs same_vintage | | 0.566 | | | | |
| openrouter__deepseek_deepseek-v4-pro | AUROC benchmark vs fresh | | 0.706 | | | | |

### abstract (unguided)

| model | group | n | ROUGE-L mean [CI] | median | 8-gram | extractable | refusals |
|---|---|---:|---|---:|---:|---:|---:|
| anthropic__claude-opus-4-8 | famous | 40 | 0.899 [0.823, 0.963] | 1.000 | 0.838 | 0.82 | 0.00 |
| anthropic__claude-opus-4-8 | benchmark | 88 | 0.250 [0.230, 0.271] | 0.236 | 0.023 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | same_vintage | 40 | 0.261 [0.229, 0.293] | 0.239 | 0.028 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | fresh | 40 | 0.202 [0.183, 0.221] | 0.206 | 0.004 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | AUROC benchmark vs same_vintage | | 0.474 | | | | |
| anthropic__claude-opus-4-8 | AUROC benchmark vs fresh | | 0.653 | | | | |
| google__gemini-3.1-pro-preview | famous | 40 | 0.828 [0.734, 0.915] | 0.989 | 0.765 | 0.60 | 0.00 |
| google__gemini-3.1-pro-preview | benchmark | 88 | 0.235 [0.213, 0.259] | 0.214 | 0.023 | 0.00 | 0.00 |
| google__gemini-3.1-pro-preview | same_vintage | 40 | 0.237 [0.203, 0.274] | 0.222 | 0.017 | 0.00 | 0.00 |
| google__gemini-3.1-pro-preview | fresh | 40 | 0.191 [0.160, 0.228] | 0.172 | 0.012 | 0.00 | 0.00 |
| google__gemini-3.1-pro-preview | AUROC benchmark vs same_vintage | | 0.476 | | | | |
| google__gemini-3.1-pro-preview | AUROC benchmark vs fresh | | 0.657 | | | | |
| openai__gpt-5.5 | famous | 40 | 0.795 [0.701, 0.882] | 0.954 | 0.717 | 0.60 | 0.00 |
| openai__gpt-5.5 | benchmark | 88 | 0.220 [0.203, 0.238] | 0.204 | 0.013 | 0.00 | 0.00 |
| openai__gpt-5.5 | same_vintage | 40 | 0.217 [0.193, 0.243] | 0.197 | 0.010 | 0.00 | 0.00 |
| openai__gpt-5.5 | fresh | 40 | 0.198 [0.177, 0.220] | 0.175 | 0.003 | 0.00 | 0.00 |
| openai__gpt-5.5 | AUROC benchmark vs same_vintage | | 0.493 | | | | |
| openai__gpt-5.5 | AUROC benchmark vs fresh | | 0.584 | | | | |
| openrouter__deepseek_deepseek-v4-pro | famous | 40 | 0.809 [0.717, 0.894] | 0.996 | 0.751 | 0.68 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | benchmark | 88 | 0.274 [0.247, 0.305] | 0.235 | 0.038 | 0.01 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | same_vintage | 40 | 0.270 [0.223, 0.327] | 0.248 | 0.070 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | fresh | 40 | 0.216 [0.191, 0.245] | 0.197 | 0.012 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | AUROC benchmark vs same_vintage | | 0.514 | | | | |
| openrouter__deepseek_deepseek-v4-pro | AUROC benchmark vs fresh | | 0.640 | | | | |

### pretext (guided)

| model | group | n | ROUGE-L mean [CI] | median | 8-gram | extractable | refusals |
|---|---|---:|---|---:|---:|---:|---:|
| anthropic__claude-opus-4-8 | famous | 4 | 0.323 [0.200, 0.405] | 0.365 | 0.104 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | benchmark | 52 | 0.257 [0.224, 0.294] | 0.235 | 0.048 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | same_vintage | 27 | 0.202 [0.158, 0.248] | 0.155 | 0.024 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | fresh | 27 | 0.169 [0.136, 0.206] | 0.161 | 0.016 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | AUROC benchmark vs same_vintage | | 0.625 | | | | |
| anthropic__claude-opus-4-8 | AUROC benchmark vs fresh | | 0.713 | | | | |
| google__gemini-3.1-pro-preview | famous | 4 | 0.269 [0.132, 0.406] | 0.255 | 0.135 | 0.00 | 0.25 |
| google__gemini-3.1-pro-preview | benchmark | 52 | 0.255 [0.219, 0.297] | 0.239 | 0.044 | 0.00 | 0.21 |
| google__gemini-3.1-pro-preview | same_vintage | 27 | 0.203 [0.167, 0.241] | 0.185 | 0.036 | 0.00 | 0.15 |
| google__gemini-3.1-pro-preview | fresh | 27 | 0.213 [0.180, 0.246] | 0.201 | 0.028 | 0.00 | 0.00 |
| google__gemini-3.1-pro-preview | AUROC benchmark vs same_vintage | | 0.599 | | | | |
| google__gemini-3.1-pro-preview | AUROC benchmark vs fresh | | 0.585 | | | | |
| openai__gpt-5.5 | famous | 4 | 0.269 [0.132, 0.396] | 0.281 | 0.026 | 0.00 | 0.00 |
| openai__gpt-5.5 | benchmark | 52 | 0.253 [0.220, 0.288] | 0.236 | 0.055 | 0.02 | 0.00 |
| openai__gpt-5.5 | same_vintage | 27 | 0.206 [0.170, 0.247] | 0.199 | 0.011 | 0.00 | 0.00 |
| openai__gpt-5.5 | fresh | 27 | 0.168 [0.136, 0.201] | 0.153 | 0.012 | 0.00 | 0.00 |
| openai__gpt-5.5 | AUROC benchmark vs same_vintage | | 0.610 | | | | |
| openai__gpt-5.5 | AUROC benchmark vs fresh | | 0.709 | | | | |
| openrouter__deepseek_deepseek-v4-pro | famous | 4 | 0.361 [0.220, 0.482] | 0.390 | 0.040 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | benchmark | 52 | 0.274 [0.239, 0.313] | 0.261 | 0.056 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | same_vintage | 27 | 0.220 [0.183, 0.259] | 0.207 | 0.010 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | fresh | 27 | 0.207 [0.176, 0.240] | 0.204 | 0.011 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | AUROC benchmark vs same_vintage | | 0.633 | | | | |
| openrouter__deepseek_deepseek-v4-pro | AUROC benchmark vs fresh | | 0.651 | | | | |

### pretext (unguided)

| model | group | n | ROUGE-L mean [CI] | median | 8-gram | extractable | refusals |
|---|---|---:|---|---:|---:|---:|---:|
| anthropic__claude-opus-4-8 | famous | 4 | 0.345 [0.262, 0.429] | 0.332 | 0.108 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | benchmark | 52 | 0.267 [0.235, 0.304] | 0.270 | 0.048 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | same_vintage | 27 | 0.199 [0.162, 0.240] | 0.178 | 0.023 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | fresh | 27 | 0.216 [0.178, 0.257] | 0.189 | 0.011 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | AUROC benchmark vs same_vintage | | 0.681 | | | | |
| anthropic__claude-opus-4-8 | AUROC benchmark vs fresh | | 0.631 | | | | |
| google__gemini-3.1-pro-preview | famous | 4 | 0.336 [0.202, 0.460] | 0.378 | 0.222 | 0.00 | 0.00 |
| google__gemini-3.1-pro-preview | benchmark | 52 | 0.296 [0.262, 0.332] | 0.289 | 0.066 | 0.02 | 0.00 |
| google__gemini-3.1-pro-preview | same_vintage | 27 | 0.205 [0.172, 0.240] | 0.199 | 0.017 | 0.00 | 0.00 |
| google__gemini-3.1-pro-preview | fresh | 27 | 0.246 [0.206, 0.288] | 0.250 | 0.043 | 0.00 | 0.00 |
| google__gemini-3.1-pro-preview | AUROC benchmark vs same_vintage | | 0.720 | | | | |
| google__gemini-3.1-pro-preview | AUROC benchmark vs fresh | | 0.618 | | | | |
| openai__gpt-5.5 | famous | 4 | 0.274 [0.181, 0.376] | 0.264 | 0.045 | 0.00 | 0.00 |
| openai__gpt-5.5 | benchmark | 52 | 0.244 [0.213, 0.277] | 0.245 | 0.051 | 0.02 | 0.00 |
| openai__gpt-5.5 | same_vintage | 27 | 0.177 [0.138, 0.222] | 0.125 | 0.012 | 0.00 | 0.00 |
| openai__gpt-5.5 | fresh | 27 | 0.201 [0.164, 0.238] | 0.198 | 0.018 | 0.00 | 0.00 |
| openai__gpt-5.5 | AUROC benchmark vs same_vintage | | 0.676 | | | | |
| openai__gpt-5.5 | AUROC benchmark vs fresh | | 0.611 | | | | |
| openrouter__deepseek_deepseek-v4-pro | famous | 4 | 0.300 [0.240, 0.366] | 0.293 | 0.030 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | benchmark | 52 | 0.264 [0.228, 0.301] | 0.269 | 0.066 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | same_vintage | 27 | 0.209 [0.175, 0.249] | 0.198 | 0.018 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | fresh | 27 | 0.206 [0.168, 0.246] | 0.195 | 0.013 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | AUROC benchmark vs same_vintage | | 0.614 | | | | |
| openrouter__deepseek_deepseek-v4-pro | AUROC benchmark vs fresh | | 0.619 | | | | |

### proof (guided)

| model | group | n | ROUGE-L mean [CI] | median | 8-gram | extractable | refusals |
|---|---|---:|---|---:|---:|---:|---:|
| anthropic__claude-opus-4-8 | famous | 1 | 0.861 | 0.861 | 0.638 | 1.00 | 0.00 |
| anthropic__claude-opus-4-8 | benchmark | 41 | 0.241 [0.210, 0.270] | 0.244 | 0.048 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | same_vintage | 12 | 0.257 [0.201, 0.316] | 0.255 | 0.037 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | fresh | 10 | 0.262 [0.189, 0.332] | 0.284 | 0.045 | 0.00 | 0.00 |
| anthropic__claude-opus-4-8 | AUROC benchmark vs same_vintage | | 0.473 | | | | |
| anthropic__claude-opus-4-8 | AUROC benchmark vs fresh | | 0.446 | | | | |
| google__gemini-3.1-pro-preview | famous | 1 | 0.254 | 0.254 | 0.710 | 0.00 | 0.00 |
| google__gemini-3.1-pro-preview | benchmark | 41 | 0.178 [0.150, 0.207] | 0.169 | 0.028 | 0.00 | 0.46 |
| google__gemini-3.1-pro-preview | same_vintage | 12 | 0.151 [0.101, 0.202] | 0.157 | 0.008 | 0.00 | 0.33 |
| google__gemini-3.1-pro-preview | fresh | 10 | 0.195 [0.142, 0.253] | 0.183 | 0.012 | 0.00 | 0.00 |
| google__gemini-3.1-pro-preview | AUROC benchmark vs same_vintage | | 0.575 | | | | |
| google__gemini-3.1-pro-preview | AUROC benchmark vs fresh | | 0.441 | | | | |
| openai__gpt-5.5 | famous | 1 | 0.794 | 0.794 | 0.633 | 1.00 | 0.00 |
| openai__gpt-5.5 | benchmark | 41 | 0.284 [0.262, 0.306] | 0.274 | 0.028 | 0.00 | 0.00 |
| openai__gpt-5.5 | same_vintage | 12 | 0.344 [0.289, 0.406] | 0.331 | 0.071 | 0.08 | 0.00 |
| openai__gpt-5.5 | fresh | 10 | 0.320 [0.275, 0.377] | 0.295 | 0.039 | 0.00 | 0.00 |
| openai__gpt-5.5 | AUROC benchmark vs same_vintage | | 0.339 | | | | |
| openai__gpt-5.5 | AUROC benchmark vs fresh | | 0.396 | | | | |
| openrouter__deepseek_deepseek-v4-pro | famous | 1 | 0.798 | 0.798 | 0.510 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | benchmark | 41 | 0.232 [0.208, 0.256] | 0.229 | 0.041 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | same_vintage | 12 | 0.288 [0.243, 0.336] | 0.280 | 0.072 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | fresh | 10 | 0.286 [0.242, 0.328] | 0.289 | 0.036 | 0.00 | 0.00 |
| openrouter__deepseek_deepseek-v4-pro | AUROC benchmark vs same_vintage | | 0.307 | | | | |
| openrouter__deepseek_deepseek-v4-pro | AUROC benchmark vs fresh | | 0.298 | | | | |

## 2. Guided minus unguided

Naming the paper should only help a model that has stored this specific document; generic fluency is the same in both modes. Values are mean paired differences in ROUGE-L with a paper-level bootstrap CI.

| model | probe | group | n | delta ROUGE-L [CI] |
|---|---|---|---:|---|
| anthropic__claude-opus-4-8 | abstract | famous | 40 | 0.021 [-0.014, 0.068] |
| anthropic__claude-opus-4-8 | abstract | benchmark | 88 | 0.051 [0.025, 0.078] |
| anthropic__claude-opus-4-8 | abstract | same_vintage | 40 | -0.002 [-0.028, 0.025] |
| anthropic__claude-opus-4-8 | abstract | fresh | 40 | 0.018 [-0.003, 0.040] |
| google__gemini-3.1-pro-preview | abstract | famous | 40 | 0.027 [-0.000, 0.069] |
| google__gemini-3.1-pro-preview | abstract | benchmark | 88 | 0.046 [0.022, 0.071] |
| google__gemini-3.1-pro-preview | abstract | same_vintage | 40 | 0.032 [-0.000, 0.071] |
| google__gemini-3.1-pro-preview | abstract | fresh | 40 | 0.037 [0.010, 0.067] |
| openai__gpt-5.5 | abstract | famous | 40 | -0.004 [-0.038, 0.031] |
| openai__gpt-5.5 | abstract | benchmark | 88 | 0.040 [0.020, 0.064] |
| openai__gpt-5.5 | abstract | same_vintage | 40 | 0.027 [0.004, 0.051] |
| openai__gpt-5.5 | abstract | fresh | 40 | 0.008 [-0.014, 0.030] |
| openrouter__deepseek_deepseek-v4-pro | abstract | famous | 40 | 0.011 [-0.039, 0.063] |
| openrouter__deepseek_deepseek-v4-pro | abstract | benchmark | 88 | 0.030 [-0.001, 0.058] |
| openrouter__deepseek_deepseek-v4-pro | abstract | same_vintage | 40 | 0.013 [-0.012, 0.041] |
| openrouter__deepseek_deepseek-v4-pro | abstract | fresh | 40 | 0.004 [-0.026, 0.037] |
| anthropic__claude-opus-4-8 | pretext | famous | 4 | -0.022 [-0.230, 0.104] |
| anthropic__claude-opus-4-8 | pretext | benchmark | 52 | -0.011 [-0.037, 0.015] |
| anthropic__claude-opus-4-8 | pretext | same_vintage | 27 | 0.003 [-0.032, 0.046] |
| anthropic__claude-opus-4-8 | pretext | fresh | 27 | -0.047 [-0.071, -0.023] |
| google__gemini-3.1-pro-preview | pretext | famous | 4 | -0.067 [-0.132, -0.001] |
| google__gemini-3.1-pro-preview | pretext | benchmark | 52 | -0.041 [-0.074, -0.009] |
| google__gemini-3.1-pro-preview | pretext | same_vintage | 27 | -0.002 [-0.037, 0.031] |
| google__gemini-3.1-pro-preview | pretext | fresh | 27 | -0.033 [-0.058, -0.008] |
| openai__gpt-5.5 | pretext | famous | 4 | -0.004 [-0.046, 0.027] |
| openai__gpt-5.5 | pretext | benchmark | 52 | 0.010 [-0.011, 0.032] |
| openai__gpt-5.5 | pretext | same_vintage | 27 | 0.029 [-0.013, 0.061] |
| openai__gpt-5.5 | pretext | fresh | 27 | -0.033 [-0.055, -0.010] |
| openrouter__deepseek_deepseek-v4-pro | pretext | famous | 4 | 0.061 [-0.067, 0.189] |
| openrouter__deepseek_deepseek-v4-pro | pretext | benchmark | 52 | 0.010 [-0.021, 0.043] |
| openrouter__deepseek_deepseek-v4-pro | pretext | same_vintage | 27 | 0.010 [-0.014, 0.035] |
| openrouter__deepseek_deepseek-v4-pro | pretext | fresh | 27 | 0.001 [-0.036, 0.036] |

## 3. Recognition quizzes

Five options, chance = 20%; one-sided exact binomial p-value against chance. `quiz` = RealMath question wording (benchmark leakage). `quiz_abstract` = the same perturbation scheme on paper abstracts, per group: the fresh group is the control, because the model cannot have seen those abstracts, so its accuracy there is what fluency alone buys. Read `quiz` against it: only the excess over the fresh-abstract accuracy is evidence of exposure.

| model | probe | group | n | accuracy | p vs chance | answered E | letters |
|---|---|---|---:|---:|---:|---:|---|
| anthropic__claude-opus-4-8 | quiz | benchmark | 130 | 0.908 | 6.75e-68 | 0.00 | A:26, B:33, C:39, D:32, E:0 |
| anthropic__claude-opus-4-8 | quiz_abstract | famous | 26 | 0.769 | 6.81e-10 | 0.00 | A:3, B:7, C:9, D:7, E:0 |
| anthropic__claude-opus-4-8 | quiz_abstract | benchmark | 63 | 0.746 | 1.58e-20 | 0.00 | A:9, B:17, C:22, D:15, E:0 |
| anthropic__claude-opus-4-8 | quiz_abstract | same_vintage | 30 | 0.733 | 4.5e-10 | 0.00 | A:3, B:11, C:7, D:9, E:0 |
| anthropic__claude-opus-4-8 | quiz_abstract | fresh | 33 | 0.667 | 7.9e-09 | 0.00 | A:4, B:8, C:10, D:11, E:0 |
| google__gemini-3.1-pro-preview | quiz | benchmark | 130 | 0.985 | 1.83e-86 | 0.00 | A:26, B:36, C:34, D:34, E:0 |
| google__gemini-3.1-pro-preview | quiz_abstract | famous | 26 | 1.000 | 6.71e-19 | 0.00 | A:5, B:6, C:6, D:9, E:0 |
| google__gemini-3.1-pro-preview | quiz_abstract | benchmark | 63 | 0.841 | 1.3e-27 | 0.02 | A:15, B:8, C:22, D:16, E:1 |
| google__gemini-3.1-pro-preview | quiz_abstract | same_vintage | 30 | 0.833 | 1.65e-13 | 0.07 | A:10, B:5, C:7, D:6, E:2 |
| google__gemini-3.1-pro-preview | quiz_abstract | fresh | 33 | 0.455 | 0.000839 | 0.45 | A:6, B:3, C:3, D:6, E:15 |
| openai__gpt-5.5 | quiz | benchmark | 65 | 0.754 | 1.12e-21 | 0.09 | A:8, B:12, C:16, D:23, E:6 |
| openai__gpt-5.5 | quiz_abstract | famous | 26 | 0.769 | 6.81e-10 | 0.00 | A:3, B:6, C:8, D:9, E:0 |
| openai__gpt-5.5 | quiz_abstract | benchmark | 63 | 0.683 | 1.54e-16 | 0.03 | A:5, B:19, C:19, D:18, E:2 |
| openai__gpt-5.5 | quiz_abstract | same_vintage | 30 | 0.700 | 4.48e-09 | 0.00 | A:5, B:8, C:10, D:7, E:0 |
| openai__gpt-5.5 | quiz_abstract | fresh | 33 | 0.485 | 0.000227 | 0.15 | B:8, C:8, D:12, E:5 |
| openrouter__deepseek_deepseek-v4-pro | quiz | benchmark | 130 | 0.562 | 1.25e-19 | 0.00 | A:44, B:28, C:49, D:9, E:0 |
| openrouter__deepseek_deepseek-v4-pro | quiz_abstract | famous | 26 | 0.692 | 7.66e-08 | 0.00 | A:3, B:8, C:9, D:6, E:0 |
| openrouter__deepseek_deepseek-v4-pro | quiz_abstract | benchmark | 63 | 0.556 | 5.17e-10 | 0.00 | A:6, B:16, C:27, D:14, E:0 |
| openrouter__deepseek_deepseek-v4-pro | quiz_abstract | same_vintage | 30 | 0.500 | 0.000231 | 0.00 | A:5, B:7, C:11, D:7, E:0 |
| openrouter__deepseek_deepseek-v4-pro | quiz_abstract | fresh | 33 | 0.485 | 0.000227 | 0.00 | A:3, B:8, C:16, D:6, E:0 |

## 4. Instant answers (no derivation allowed)

The question with an instruction to answer immediately. A research-level result that is produced correctly without any derivation is either trivial or recalled. `direct` is the model's regular no-tool accuracy on the same items; `instant & direct` counts items solved both ways. Correctness source per model is listed below.

| model | n | instant correct | declined (NONE) | direct correct | instant & direct | instant-correct among direct-wrong |
|---|---:|---:|---:|---:|---:|---:|
| anthropic__claude-opus-4-8 | 133 | 46 (34.6%) | 3 | 88 (66.2%) | 41 | 5 |
| google__gemini-3.1-pro-preview | 133 | 54 (40.6%) | 18 | 89 (66.9%) | 52 | 2 |
| openai__gpt-5.5 | 133 | 44 (33.1%) | 9 | 90 (67.7%) | 39 | 5 |
| openrouter__deepseek_deepseek-v4-pro | 133 | 35 (26.3%) | 18 | 70 (52.6%) | 28 | 7 |

- anthropic__claude-opus-4-8: saved scores
- google__gemini-3.1-pro-preview: saved scores
- openai__gpt-5.5: saved scores
- openrouter__deepseek_deepseek-v4-pro: saved scores

## 5. Memorisation vs. direct correctness on the benchmark

If contamination drove the direct baseline, items that the models solve should come from papers they recall better. AUROC = P(recall score of a directly-solved item > score of an unsolved item), per model, on the paper-level ROUGE-L of each probe; 0.5 means recall does not separate solved from unsolved items.

| model | probe | solved n | unsolved n | mean solved | mean unsolved | AUROC |
|---|---|---:|---:|---:|---:|---:|
| anthropic__claude-opus-4-8 | abstract_guided | 88 | 45 | 0.287 | 0.307 | 0.498 |
| anthropic__claude-opus-4-8 | abstract_unguided | 88 | 45 | 0.246 | 0.249 | 0.470 |
| anthropic__claude-opus-4-8 | pretext_guided | 66 | 26 | 0.217 | 0.241 | 0.435 |
| anthropic__claude-opus-4-8 | pretext_unguided | 66 | 26 | 0.235 | 0.228 | 0.475 |
| anthropic__claude-opus-4-8 | proof_guided | 38 | 17 | 0.245 | 0.206 | 0.628 |
| google__gemini-3.1-pro-preview | abstract_guided | 89 | 44 | 0.276 | 0.293 | 0.472 |
| google__gemini-3.1-pro-preview | abstract_unguided | 89 | 44 | 0.211 | 0.274 | 0.308 |
| google__gemini-3.1-pro-preview | pretext_guided | 65 | 27 | 0.282 | 0.200 | 0.703 |
| google__gemini-3.1-pro-preview | pretext_unguided | 65 | 27 | 0.278 | 0.272 | 0.494 |
| google__gemini-3.1-pro-preview | proof_guided | 37 | 18 | 0.177 | 0.151 | 0.547 |
| openai__gpt-5.5 | abstract_guided | 90 | 43 | 0.246 | 0.254 | 0.454 |
| openai__gpt-5.5 | abstract_unguided | 90 | 43 | 0.220 | 0.234 | 0.501 |
| openai__gpt-5.5 | pretext_guided | 69 | 23 | 0.252 | 0.249 | 0.488 |
| openai__gpt-5.5 | pretext_unguided | 69 | 23 | 0.208 | 0.241 | 0.430 |
| openai__gpt-5.5 | proof_guided | 37 | 18 | 0.283 | 0.252 | 0.625 |
| openrouter__deepseek_deepseek-v4-pro | abstract_guided | 70 | 63 | 0.272 | 0.305 | 0.424 |
| openrouter__deepseek_deepseek-v4-pro | abstract_unguided | 70 | 63 | 0.257 | 0.269 | 0.471 |
| openrouter__deepseek_deepseek-v4-pro | pretext_guided | 52 | 40 | 0.256 | 0.287 | 0.411 |
| openrouter__deepseek_deepseek-v4-pro | pretext_unguided | 52 | 40 | 0.221 | 0.257 | 0.403 |
| openrouter__deepseek_deepseek-v4-pro | proof_guided | 26 | 29 | 0.227 | 0.205 | 0.578 |

### Strata

`solved by k` counts the 4 probed models' no-tool runs. Recall scores are averaged over the probed models and over the items in the stratum; `instant` is the share of (model, item) pairs answered correctly with no derivation.

| stratum | items | abstract_guided | abstract_unguided | pretext_guided | pretext_unguided | proof_guided | instant |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 133 | 0.278 | 0.242 | 0.250 | 0.241 | 0.222 | 0.34 |
| answer type: expression | 73 | 0.299 | 0.250 | 0.251 | 0.255 | 0.234 | 0.17 |
| answer type: number | 60 | 0.253 | 0.231 | 0.249 | 0.221 | 0.206 | 0.53 |
| solved by 0/4 models | 32 | 0.295 | 0.262 | 0.244 | 0.261 | 0.198 | 0.05 |
| solved by 1/4 models | 7 | 0.282 | 0.249 | 0.232 | 0.235 | 0.172 | 0.18 |
| solved by 2/4 models | 11 | 0.286 | 0.271 | 0.231 | 0.227 | 0.202 | 0.30 |
| solved by 3/4 models | 24 | 0.277 | 0.216 | 0.266 | 0.278 | 0.241 | 0.31 |
| solved by 4/4 models | 59 | 0.266 | 0.234 | 0.253 | 0.223 | 0.232 | 0.53 |
| cohort 2022 | 52 | 0.313 | 0.258 | 0.264 | 0.270 | 0.225 | 0.22 |
| cohort 2024 | 81 | 0.255 | 0.231 | 0.243 | 0.224 | 0.220 | 0.41 |

## Caveats

- Black-box probes detect verbatim recall; a model can be trained on a paper and recall nothing verbatim, so low scores do not prove absence of exposure. High recall of a paper also does not prove the benchmark answer was memorised; section 5 is the relevant link.
- The fresh group is only a floor if every probed model's training cutoff precedes the fresh window; check the cutoffs before quoting it.
- The famous group is a ceiling reached through millions of copies; ordinary papers are not expected to get there.
