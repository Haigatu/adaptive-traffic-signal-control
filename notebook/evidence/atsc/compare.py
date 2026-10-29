"""Run two trained agents in synchronized SUMO windows, then open their results."""

import argparse
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import uuid
import webbrowser

from .demo_report import build_report, open_report

ROOT = Path(__file__).resolve().parents[1]


def worker(argv, barrier):
    try:
        from .demo import main
        main(argv, sync=barrier)
    except BaseException:
        barrier.abort()
        raise


def collect_pair(folder):
    runs = {}
    for name in ("no_forecast", "forecast"):
        child = folder / name
        runs[name] = json.loads((child / "summary.json").read_text(encoding="utf-8"))
        routes = list(child.glob("*.rou.xml"))
        if len(routes) != 1:
            raise ValueError("Expected one demand file per controller")
        runs[name]["route_sha256"] = hashlib.sha256(routes[0].read_bytes()).hexdigest()
    a, b = runs.values()
    for key in ("seed", "saturation", "warmup", "duration", "route_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"Unmatched comparison: {key}")
    complete = all(r["status"] == "completed" for r in runs.values())
    if complete and a["metrics"]["elapsed_s"] != b["metrics"]["elapsed_s"]:
        raise ValueError("Unmatched measurement horizons")
    return {"mode": "paired", "status": "completed" if complete else "incomplete",
            "runs": runs, "seed": a["seed"], "saturation": a["saturation"],
            "duration": a["duration"], "warmup": a["warmup"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--warmup", type=int, default=3600)
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--saturation", type=float, default=0.7)
    parser.add_argument("--policy", choices=["original", "guarded", "improved"],
                        default="improved", help="Right-hand forecasting controller")
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sumo-gui", action="store_true",
                        help="Also open the two native SUMO-GUI windows")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.warmup < 0 or args.delay < 0 or not 0 < args.saturation <= 1:
        parser.error("duration > 0, warmup/delay >= 0, 0 < saturation <= 1 required")
    folder = ROOT / "demo_runs" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_paired_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    from .web_demo import start_server
    initial = {"mode": "paired", "status": "running",
               "seed": args.seed, "saturation": args.saturation,
               "duration": args.duration, "warmup": args.warmup,
               "runs": {"no_forecast": {"agent": "no_forecast"},
                        "forecast": {"agent": args.policy}}}
    server, url = start_server(folder, initial)
    print(f"Live dashboard: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    common = ["--duration", str(args.duration), "--warmup", str(args.warmup),
              "--delay", str(args.delay), "--seed", str(args.seed),
              "--saturation", str(args.saturation), "--no-report", "--web-frames"]
    if not args.sumo_gui or args.headless:
        common.append("--headless")
    processes = []
    print("WEBSITE LEFT: NO FORECAST | RIGHT: WITH FORECAST", flush=True)
    print("Identical demand. Synchronized 0.5-second simulation steps.", flush=True)
    try:
        for name, agent, position in [
            ("no_forecast", "no_forecast", "0,0"),
            ("forecast", args.policy, "770,0"),
        ]:
            child_args = common + ["--agent", agent, "--output", str(folder / name),
                                   "--window-pos", position]
            process = ctx.Process(target=worker, args=(child_args, barrier), name=name)
            process.start()
            processes.append(process)
        while any(p.is_alive() for p in processes):
            for process in processes:
                process.join(timeout=0.2)
                if process.exitcode not in (None, 0):
                    barrier.abort()
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError(f"A simulation failed. See {folder}")
    except BaseException:
        barrier.abort()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        server.run = {**initial, "status": "failed"}
        server.shutdown()
        server.server_close()
        raise
    result = collect_pair(folder)
    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = build_report(folder)
    print(f"Results: {report}", flush=True)
    server.run = result
    # The live page detects completion and loads its portable replay report.
    if args.no_browser:
        server.shutdown()
        server.server_close()
    else:
        print("Dashboard ready. Press Ctrl+C to stop the local server.", flush=True)
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
