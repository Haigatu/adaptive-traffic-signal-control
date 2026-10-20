# Policy improvement experiment

1. Preserve original simulation logic, checkpoints and saved test results.
2. Diagnose original policies on matched full-horizon runs.
3. Train a separate candidate from stored training transitions, using a
   queue-aware teacher and the existing feasibility mask. Record this as
   policy distillation, not an additional claim of pure RL training.
4. Select candidates only on validation demand (split `tune`, fresh seeds).
5. Compare the selected policy with original forecast/no-forecast agents on
   fresh test seeds at low, medium and high demand.
6. Integrate the candidate only if its measured aggregate delay improves.
   Report load-specific regressions and the full raw results.
7. Keep the original forecast ablation accessible; label the improved policy
   separately because it adds clocks, a guard and training changes.

Primary metric: delay per requested vehicle. Secondary metrics: arrivals,
queue, no-green time, teleport and collision counts. No seed selection based
on favorable test outcomes.
