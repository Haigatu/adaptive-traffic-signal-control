# Improved demonstration policy

The improved controller uses the existing 56-input DuelingQ network, the
unchanged TGCN predictor, and the existing 120-second no-green feasibility
mask. Its weights were initialized from the guarded continuation and then
trained with supervised policy distillation. It is not a new pure D3QN
reinforcement-learning experiment.

The teacher ranks feasible approaches using current queues, accumulated
waiting, causal forecasts, time since green, and a current-phase bonus.
The deployed controller evaluates the trained neural network, then applies
the feasibility mask. It does not use a fixed SUMO signal program.

## Selection and evaluation

- Training uses only the stored continuation replay: 18,454 current/next
  states, split 90/10 for fitting and label-agreement checks. Adjacent replay
  states may be correlated; label accuracy is not a traffic-performance test.
- Three continuity bonuses (5, 15, 30) were trained for 40 epochs with seed 42.
- Model selection uses PeMS08's `tune` split, demand seed 610, loads
  0.40/0.70/0.95, 3,600 seconds of warm-up and 1,800 measured seconds.
- The selection criterion is equal-load mean delay normalized against
  candidate 5. Candidate 5 was selected before running the final tests.
- First-stage tests use the separate `test` split, seeds 1110 and 1111, all three
  loads, 3,600 seconds of warm-up and 3,600 measured seconds per run.
- All four controllers receive matching route contents and measurement
  horizons. Raw results include route and checkpoint SHA-256 hashes.

The comparison includes the original D3QN without forecasting, original
D3QN with forecasting, previous guarded continuation, and improved policy.
Changes in timers, constraints and training mean that the improvement cannot
be attributed to forecasting alone. There are two test demand seeds and one
training seed; this is a limited demonstration benchmark.

The global selection regressed under congested demand. A second configuration
selects the lowest-delay validation checkpoint separately for each configured
load: candidate 5 at 0.40, candidate 30 at 0.70 and 0.95. Other requested loads
use the nearest configured level. Selection uses the requested demand setting,
not future vehicle observations. This configuration is frozen in
`load_selection.json`, then tested on fresh seeds **1120 and 1121**.
The first-stage results are retained; they are not substituted for the second
test set. `summary_by_load.json` and `test_results_by_load.json` contain the
second-stage benchmark.

## Measured delay

Mean delay proxy in seconds per requested vehicle, two fresh demand seeds
per load, 3,600 measured seconds per run:

| Controller | Load 0.40 | Load 0.70 | Load 0.95 |
| --- | ---: | ---: | ---: |
| Original D3QN, no forecast | 56.39 | 220.01 | 697.71 |
| Original D3QN + forecast | 116.40 | 232.49 | 682.41 |
| Previous guarded continuation | 111.24 | 105.63 | 573.79 |
| Improved distilled policy + forecast | 27.07 | 134.90 | 620.62 |

Compared with no forecast, the load-mean reductions are **52.0%, 38.7%,
and 11.0%**. These are not guarantees: at load 0.95, seed 1121 regressed from
637.99 to 648.14 seconds (+1.6%). The existing guarded continuation is also
competitive: it beats the new distilled policy at loads 0.70 and 0.95.
All four controllers are retained in the raw benchmark
and the website's collapsed benchmark table.

The original 300-second default demo at seed 1000 also regressed in delay
(155.75 to 161.25 seconds), despite a smaller mean queue (126.60 to 116.36).
The demo now observes 1,800 seconds to show longer queue dynamics, and keeps
`--duration 300` available. Changing duration changes the generated demand
profile; the old and new horizons must not be treated as a matched comparison.

The completed default 1,800-second demo (seed 1000, load 0.70, full warm-up)
measured **194.99 -> 105.75 s delay (-45.8%)**, **177.70 -> 84.70 average
queued vehicles (-52.3%)**, and **1,733 -> 1,941 arrivals (+12.0%)**.
This is a separate demonstration example, not the held-out benchmark.
Its sanitized paired summary is saved as `demo_1800_summary.json`.
Both runs produced 1,800 synchronized frames and verified external `online`
signal control throughout. Original artifact hashes remained unchanged.

Original checkpoints and research outputs remain unchanged. The experiment
is stored separately under `artifacts/improvement/`: protocol, validation
results, selection, trained checkpoints, individual tests and summaries.

## Reproduce

```powershell
# Existing included checkpoint; no training needed
.\demo.cmd --policy improved

# Original pair, retained for comparison
.\demo.cmd --policy original

# Re-run selected-policy benchmark (completed cases are cached)
.\.venv\Scripts\python.exe -m atsc.improve test-by-load
```

`python -m atsc.improve train` refuses to overwrite an existing experiment.
To repeat training, use a separate checkout without `artifacts/improvement`.
Run `train`, `validate`, `test`, `select-by-load`, then `test-by-load`.

The bound applies to an approach's time without green, not to each vehicle's
waiting time. Delay is the original speed-loss proxy including pending
insertions. Congestion can remain at high demand even with bounded red time.
