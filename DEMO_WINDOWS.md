# Windows demo

## Start

After running `setup.cmd` and setting `SUMO_HOME`, run:

```powershell
.\demo.cmd
```

A local website opens with two side-by-side live intersections. The left panel
uses original D3QN without forecasting; the right uses the improved distilled
policy with TGCN and a red-time guard. Both use identical
demand and synchronized 0.5-second simulation steps. Vehicles and signals come
from actual SUMO via TraCI, rendered on canvas. After completion the same page
shows a replay, numeric results and percentage differences.

To additionally show the native SUMO windows:

```powershell
.\demo.cmd --sumo-gui
```

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--duration` | 1800 | Measured simulation seconds |
| `--warmup` | 3600 | Simulated seconds of forecast-history collection |
| `--delay` | 0 | Additional milliseconds per simulation step |
| `--seed` | 1000 | Shared demand and SUMO seed |
| `--saturation` | 0.7 | Shared nominal traffic load, greater than 0 and at most 1 |
| `--sumo-gui` | off | Also open the two native SUMO-GUI windows |
| `--no-browser` | off | Generate results without opening the browser |
| `--policy` | improved | Right-hand policy: improved, original, or guarded |

Warm-up uses the original cyclic controller and no presentation delay.
Wait for **AI ACTIVE** before presenting the learned policy. A shortened
warm-up is useful for smoke tests but does not provide the full forecast history.
Changing total scenario duration also changes the generated demand profile;
only compare policies within the same paired run.

Each window has its own log and SUMO connection. A synchronization barrier
keeps both simulation clocks aligned even when one policy switches after
10 seconds and the other uses a 13-second green/yellow decision interval.
The controller verifies every recorded signal against its chosen action
and checks that SUMO is using the external `online` signal program.

## Results

Output: `demo_runs/<timestamp>_paired_<id>/results.html`.
The page shows two intersection replays, key metrics, percentage changes,
and two queue traces. Use Replay and the timeline slider after completion.
Replay supports 5x, 20x and 60x speed; the default 20x shows a 30-minute
recording in approximately 90 seconds. Playback speed does not change metrics.
A collapsed section retains the historical 60-run benchmark. It is separate
from the current live comparison.

```powershell
# Reopen the latest completed report
.\.venv\Scripts\python.exe -m atsc.demo_report

# Open a specific report
.\.venv\Scripts\python.exe -m atsc.demo_report --run demo_runs/<run>

# Optional single-window guarded continuation
.\.venv\Scripts\python.exe -m atsc.demo --agent guarded
```

Use `--policy original` for the original 40-input and 52-input policies.
The improved version adds policy distillation, clocks and a 120-second
no-green constraint. It does not claim a causal improvement from forecasting
alone. The checkpoint is selected by requested load using validation results.
The default observes 30 simulated minutes because a five-minute sample can
miss later queue growth. Use `--duration 300` for a short run.

Vehicle observations are batched using TraCI subscriptions. A parity test
compares states and all metrics with ordinary TraCI on identical traffic.
This changes communication overhead, not signal timing or simulation logic.

## Troubleshooting

- **SUMO not found:** set `SUMO_HOME` to the installation root, e.g. `D:\SUMO`.
  Executables are searched in `SUMO_HOME/bin`, PATH, and the `eclipse-sumo` package.
- **Missing PeMS08:** run `python -m atsc.prepare_demo`. The download must match
  the SHA-256 in the existing manifest. No training is performed.
- **No cars initially:** vehicles take time to reach the central intersection.
  Retain the default full warm-up for presentations.
- **Slow warm-up:** TraCI communicates over local TCP. Two synchronized simulations
  and per-vehicle metrics can take several real minutes on slower computers.
- **Connection denied:** allow local Python-to-SUMO TCP connections.
- **One window paused:** both clocks wait at the synchronization barrier. Resume
  within 120 seconds; longer pauses fail with an explicit timeout.
- **One window closed:** the comparison fails rather than reporting a valid paired
  result. Child logs are retained; rerun the command.
- **Small display:** resize or move the SUMO windows manually. Initial size is
  760 x 800 pixels at positions (0, 0) and (770, 0).

The live page uses a local server bound only to 127.0.0.1 on a free port.
Keep the terminal open; Ctrl+C stops the server. The saved HTML replay needs
no server, CDN, account, or internet connection.
Use Print / PDF to export it. Simulation outputs and weights are separate:
launching the demo never trains or writes checkpoints.
