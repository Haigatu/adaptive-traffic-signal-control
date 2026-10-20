# Contributing

Use Python 3.12 and SUMO 1.27.1. Follow the setup and test commands in README.md.

- Keep demo inference separate from training.
- Do not replace checkpoints or historical experiment results as part of UI changes.
- Compare policies with identical demand, warm-up and measurement horizons.
- Preserve the 0.5-second timestep, 10-second green and 3-second yellow logic.
- Add regression checks for controller, pairing or metric changes.
- Keep the interface and new documentation in English.
- Do not commit `.venv`, downloaded PeMS arrays, local paths, credentials or `demo_runs`.

For bug reports, include the command, Python/SUMO versions, traceback, and
relevant non-sensitive fields from `summary.json`.
