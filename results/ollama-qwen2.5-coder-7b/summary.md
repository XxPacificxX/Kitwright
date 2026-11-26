## ollama / qwen2.5-coder:7b: 50 tasks, up to 4 attempts

| | One-shot | Self-correcting |
|---|---|---|
| Pass rate | 11/50 (22%, 95% CI 13%-35%) | 12/50 (24%, 95% CI 14%-37%) |
| Model calls per task | 1.00 | 2.92 |
| Input tokens per task | 459 | 3,032 |
| Output tokens per task | 136 | 388 |
| Model time per task | 28.5 s | 101.8 s |
| Cost per task | $0 (local) | $0 (local) |

Self-correction fixed 1 tasks and broke 0 (exact McNemar p = 1.0).

Pass rate when the agent is capped at k attempts: k=1: 22%, k=2: 24%, k=3: 24%, k=4: 24%.

Self-check: the agent declared `VERDICT: DONE` on 30 tasks; 19 of those were actually wrong.

| Category | Tasks | One-shot | Self-correcting |
|---|---|---|---|
| algorithm | 14 | 2 | 2 |
| application | 5 | 1 | 1 |
| state-prep | 15 | 6 | 7 |
| unitary | 16 | 2 | 2 |

Stop reasons: out_of_rounds 20, verified 30. Mean attempts 2.32, mean wall time 105 s per task.
