"""Offline comparison page: saved matched experiments + a separate live run."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "artifacts/continuation"
CONTROLLERS = [
    "webster", "max_pressure", "d3qn_original",
    "d3qn_guard_only", "d3qn_guard_trained",
]
METRICS = [
    "delay_per_requested_vehicle_s", "throughput", "mean_queue_vehicles",
    "conflicts_per_1000_arrivals", "max_without_green_s",
]


def aggregate(rows):
    """Require the complete paired protocol; never silently average missing runs."""
    expected = set(itertools.product(CONTROLLERS, [1000, 1001, 1002, 1003], [0.4, 0.7, 0.95]))
    keys = [(r["controller"], r["seed"], r["saturation"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected 60 unique matched experiment runs")
    if any(r["elapsed_s"] != 3600 or r["split"] != "test" for r in rows):
        raise ValueError("Comparison requires completed 3600-second test runs")
    result = []
    for saturation in [0.4, 0.7, 0.95]:
        for controller in CONTROLLERS:
            sample = [r for r in rows if r["controller"] == controller
                      and r["saturation"] == saturation]
            group = {"controller": controller, "saturation": saturation, "n": len(sample)}
            for metric in METRICS:
                values = [r[metric] for r in sample]
                group[metric] = (max(values) if metric == "max_without_green_s"
                                 else sum(values) / len(values))
            result.append(group)
    return result


def build_report(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    decisions_path = run_dir / "decisions.jsonl"
    decisions = [json.loads(line) for line in decisions_path.read_text(
        encoding="utf-8").splitlines() if line.strip()] if decisions_path.exists() else []
    source = SAVED / "test_results.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    data = {
        "run": summary, "decisions": decisions, "groups": aggregate(rows),
        "source": "artifacts/continuation/test_results.json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checkpoint_matches": any(
            r.get("controller_sha256") == summary.get("checkpoint_sha256")
            for r in rows if r["controller"].startswith("d3qn")
        ) if summary.get("checkpoint_sha256") else False,
    }
    if summary.get("mode") == "paired":
        data["pair"] = {}
        data["replays"] = {}
        for name in ("no_forecast", "forecast"):
            path = run_dir / name / "decisions.jsonl"
            data["pair"][name] = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            frames = run_dir / name / "frames.jsonl"
            data["replays"][name] = [
                json.loads(line) for line in frames.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ] if frames.exists() else []
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = (Path(__file__).parent / "demo_report.html").read_text(encoding="utf-8")
    output = run_dir / "results.html"
    output.write_text(template.replace("__REPORT_DATA__", payload), encoding="utf-8")
    return output


def open_report(path):
    if not webbrowser.open(Path(path).resolve().as_uri()):
        print(f"Open the report manually: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Existing demo_runs/<run> directory")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    run = args.run
    if run is None:
        summaries = list((ROOT / "demo_runs").glob("*/summary.json"))
        if not summaries:
            parser.error("No saved demo runs; start demo.cmd first")
        run = max(summaries, key=lambda p: p.stat().st_mtime).parent
    path = build_report(run)
    print(path)
    if not args.no_open:
        open_report(path)


if __name__ == "__main__":
    main()
