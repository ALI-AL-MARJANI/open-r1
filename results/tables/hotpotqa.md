| Run | Seeds | n | Format ↑ | Ans EM ↑ | Ans F1 ↑ | Sup EM ↑ | Sup F1 ↑ | Joint F1 ↑ | Chunk F1 ↑ | Unverified quotes ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `few_shot_qwen0.5b` | 1 | 500 | 44.6 [39.8, 49.2] | 2.0 [0.8, 3.2] | 6.8 [5.2, 8.6] | 0.4 [0.0, 1.0] | 4.7 [3.3, 6.1] | 1.0 [0.6, 1.6] | 5.8 [4.1, 7.3] | 19.3 [14.3, 24.8] |
| `sft_qwen0.5b` | 1 | 500 | 99.4 [98.6, 100.0] | 19.0 [15.6, 22.4] | 25.5 [22.1, 29.0] | 0.0 [0.0, 0.0] | 13.1 [11.0, 15.3] | 5.1 [3.7, 6.4] | 21.3 [18.4, 24.0] | 10.9 [8.0, 13.8] |
| `zero_shot_qwen0.5b` | 1 | 500 | 50.6 [46.2, 55.0] | 1.2 [0.4, 2.2] | 6.9 [5.5, 8.3] | 0.2 [0.0, 0.8] | 8.7 [7.0, 10.5] | 1.6 [1.1, 2.3] | 11.7 [9.4, 13.9] | 28.1 [22.9, 33.0] |

Values are percentages. One seed: point estimate [95% bootstrap interval over examples]. Several seeds: mean ± standard deviation across training seeds.
