| Run | Seeds | n | Format ↑ | Accuracy ↑ | Macro F1 ↑ | Answer rate ↑ | Unverified quotes ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `few_shot_qwen0.5b` | 1 | 500 | 65.0 [60.8, 69.2] | 37.0 [32.4, 41.2] | 26.3 [23.0, 29.5] | 63.8 [59.6, 68.0] | 3.7 [2.4, 5.1] |
| `sft_qwen0.5b` | 1 | 500 | 100.0 [100.0, 100.0] | 16.8 [13.4, 20.2] | 14.7 [11.6, 17.7] | 72.8 [69.0, 76.6] | 0.8 [0.0, 1.9] |
| `zero_shot_qwen0.5b` | 1 | 500 | 57.0 [52.6, 61.4] | 25.2 [21.0, 29.0] | 19.6 [16.6, 22.2] | 57.0 [52.6, 61.4] | 4.6 [2.4, 7.4] |

Values are percentages. One seed: point estimate [95% bootstrap interval over examples]. Several seeds: mean ± standard deviation across training seeds.
