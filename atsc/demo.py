"""Inference-only Windows SUMO-GUI demo using the existing trained policies."""

import argparse
from datetime import datetime
import hashlib
import json
import uuid
import time
from pathlib import Path

import numpy as np
import torch

from .data import ROOT
from .environment import Environment
from .guarded import ClockEnvironment
from .models import DuelingQ
from .network import DIRECTIONS, binary


class DemoPacing:
    """Wall-clock pacing only; simulation timestep and control stay unchanged."""

    step_delay = 0.0
    sync = None
    web_frames = False

    def _tick(self, measure=True):
        if self.sync is not None:
            self.sync.wait(timeout=120)
        super()._tick(measure)
        if self.web_frames:
            t = self.sim.simulation.getTime()
            if not measure and t % 300 == 0:
                self._publish_frame({"stage": "warmup", "time": t, "warmup": self.warmup})
            elif measure and t % 1 == 0:
                center = self.sim.junction.getPosition("J")
                frame = {
                    "stage": "running", "time": t - self.measure_start,
                    "center": center,
                    "signal": self.sim.trafficlight.getRedYellowGreenState("J"),
                    "queue": self.queue_by_approach().tolist(),
                    "vehicles": [
                        [v, *self.sim.vehicle.getPosition(v),
                         self.sim.vehicle.getAngle(v), self.sim.vehicle.getSpeed(v)]
                        for v in self.sim.vehicle.getIDList()
                    ],
                }
                self._publish_frame(frame)
                with (self.run_dir / "frames.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(frame) + "\n")
        if measure and self.step_delay:
            time.sleep(self.step_delay)

    def _publish_frame(self, frame):
        path = self.run_dir / "live.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(frame), encoding="utf-8")
        # A browser poll may briefly hold the file open on Windows.
        for attempt in range(5):
            try:
                temp.replace(path)
                return
            except PermissionError:
                time.sleep(.01)


class GuiEnvironment(DemoPacing, Environment):
    pass


class GuiClockEnvironment(DemoPacing, ClockEnvironment):
    pass


def main(argv=None, sync=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["improved", "guarded", "original", "no_forecast"], default="guarded")
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--web-frames", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--window-pos", help="SUMO window position x,y")
    parser.add_argument("--window-size", default="760,800", help="SUMO window size x,y")
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument("--warmup", type=int, default=3600)
    parser.add_argument("--delay", type=float, default=50,
                        help="GUI delay in milliseconds per 0.5 simulation seconds")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--saturation", type=float, default=0.7)
    parser.add_argument("--headless", action="store_true",
                        help="Test the same TraCI controller without the GUI")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip the comparison page after simulation")
    parser.add_argument("--no-browser", action="store_true",
                        help="Save the comparison page without opening a browser")
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.warmup < 0 or args.delay < 0 or not 0 < args.saturation <= 1:
        parser.error("duration > 0, warmup/delay >= 0, 0 < saturation <= 1 required")

    guarded = args.agent in ("guarded", "improved")
    forecast = args.agent != "no_forecast"
    checkpoint = ROOT / "artifacts" / {
        "improved": "improvement/selected.pt",
        "guarded": "continuation/selected.pt",
        "original": "control/d3qn_forecast_seed0.pt",
        "no_forecast": "control/d3qn_no_forecast_seed0.pt",
    }[args.agent]
    if args.agent == "improved":
        from .improve import improved_checkpoint
        checkpoint = improved_checkpoint(args.saturation)
    predictor = ROOT / "artifacts/forecast/simulation_tgcn.pt" if forecast else None
    for path in (checkpoint, ROOT / "data/PEMS08.npz", *([predictor] if predictor else [])):
        if not path.is_file():
            parser.error(f"Missing required file: {path}. See DEMO_WINDOWS.md.")
    executable = binary("sumo" if args.headless else "sumo-gui")
    torch.set_num_threads(1)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = 56 if guarded else (52 if forecast else 40)
    if ck["inputs"] != expected:
        raise ValueError(f"Checkpoint has {ck['inputs']} inputs; expected {expected}")
    model = DuelingQ(expected)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    run_dir = args.output or ROOT / "demo_runs" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint": str(checkpoint),
        "training_method": ck.get("method", "D3QN"),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "predictor_sha256": hashlib.sha256(predictor.read_bytes()).hexdigest() if predictor else None,
        "sumo": executable, "backend": "traci", "status": "starting",
    }
    env_type = GuiClockEnvironment if guarded else GuiEnvironment
    env = env_type(
        predictor=predictor, with_forecast=forecast, backend="traci", gui=not args.headless,
        run_dir=run_dir, warmup=args.warmup, duration=args.duration,
        gui_name="WITH_FORECAST" if forecast else "NO_FORECAST",
        gui_options=(["--window-size", args.window_size, "--window-pos", args.window_pos]
                     if args.window_pos else []),
        **({"enforce": True} if guarded else {}),
    )
    env.step_delay = args.delay / 1000 if args.web_frames or not args.headless else 0
    from .traci_cache import CachedTraCI
    env.sim = CachedTraCI(env.sim)
    env.sync = sync
    env.web_frames = args.web_frames
    summary = run_dir / "summary.json"
    print(f"Checkpoint: {checkpoint}\nSUMO: {executable}\nLogs: {run_dir}", flush=True)
    print(f"WARM-UP: {args.warmup}s simulated, fixed cyclic control, GUI delay=0.",
          flush=True)
    if args.warmup < 3600:
        print("Short warm-up: forecast history is incomplete; not comparable to experiments.",
              flush=True)
    try:
        state = env.reset(args.seed, args.saturation, "test")
        metadata["sumo_version"] = env.sim.getVersion()
        if not args.headless:
            x, y = env.sim.junction.getPosition("J")
            env.sim.gui.setBoundary("View #0", x - 160, y - 160, x + 160, y + 160)
        print(f"AI ACTIVE: {metadata['training_method']}; signals controlled via TraCI.", flush=True)
        with (run_dir / "decisions.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            done = False
            while not done:
                with torch.inference_mode():
                    values = model(torch.from_numpy(state)[None])[0].numpy()
                if not np.isfinite(values).all():
                    raise RuntimeError("Non-finite Q values; refusing fallback")
                raw_action = int(values.argmax())
                mask = env.action_mask() if guarded else np.ones(4, dtype=bool)
                action = int(np.where(mask, values, -np.inf).argmax())
                before = float(env.sim.simulation.getTime())
                state, _, done, info = env.step(action)
                signal = env.sim.trafficlight.getRedYellowGreenState("J")
                program = env.sim.trafficlight.getProgram("J")
                expected_signal = "".join("G" if i // 3 == action else "r" for i in range(12))
                if program != "online" or signal != expected_signal:
                    raise RuntimeError("SUMO signal does not match the agent action")
                row = {
                    "time": before, "elapsed": env.elapsed, "q_values": values.tolist(),
                    "raw_action": raw_action, "action": action, "direction": DIRECTIONS[action],
                    "feasible": mask.tolist(), "guard_override": action != raw_action,
                    "signal": signal, "program": program,
                    "queue": env.queue_by_approach().tolist(),
                    "vehicles": env.sim.vehicle.getIDCount(), **info,
                }
                if guarded:
                    row["red_age"] = env.red_age.tolist()
                    if env.max_red.max() > 120 + 1e-6:
                        raise RuntimeError("Red-time bound violated")
                log.write(json.dumps(row) + "\n")
                print(f"AI t={env.elapsed:6.1f}s  green={DIRECTIONS[action]}  "
                      f"queue={row['queue']}  vehicles={row['vehicles']}  "
                      f"guard={action != raw_action}  program={program}", flush=True)
        metadata.update(status="completed", metrics=env.metrics())
    except KeyboardInterrupt:
        if sync is not None:
            sync.abort()
        metadata["status"] = "interrupted"
        if env.running:
            metadata["metrics"] = env.metrics()
        print("\nDemo stopped.", flush=True)
    except Exception as error:
        if sync is not None:
            sync.abort()
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        summary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            env.close()
        finally:
            print(f"Summary: {summary}", flush=True)
            if not args.no_report:
                from .demo_report import build_report, open_report
                report = build_report(run_dir)
                print(f"Comparison page: {report}", flush=True)
                if not args.no_browser:
                    open_report(report)


if __name__ == "__main__":
    main()
