"""Independent scenario evaluation, paired tests, and a predeclared safety gate."""

import argparse, json, multiprocessing as mp, hashlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from .data import ROOT
from .control import evaluate, checkpoint_path

OUT = ROOT / "artifacts" / "evaluation"


def worker(task):
    name, seed, sat, split, duration = task
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{split}_{name}_{seed}_{sat:.2f}.json"
    if dest.exists():
        cached = json.loads(dest.read_text())
        valid = cached.get("elapsed_s") == duration
        if name.startswith("d3qn"):
            checksum = hashlib.sha256(checkpoint_path(name, 0).read_bytes()).hexdigest()
            valid = valid and cached.get("controller_sha256") == checksum
        if name == "forecast_rule" or name in ("d3qn_forecast", "d3qn_safety"):
            predictor = ROOT / "artifacts" / "forecast" / "simulation_tgcn.pt"
            checksum = hashlib.sha256(predictor.read_bytes()).hexdigest()
            valid = valid and cached.get("predictor_sha256") == checksum
        if valid:
            return cached
    result = evaluate(name, seed, sat, split, duration)
    dest.write_text(json.dumps(result, indent=2))
    return result


def sweep(names, seeds, loads, split="test", workers=3, duration=3600):
    tasks = [(n, s, l, split, duration) for n in names for s in seeds for l in loads]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = []
        for row in pool.imap_unordered(worker, tasks):
            rows.append(row)
            print(
                "evaluated",
                row["controller"],
                row["seed"],
                row["saturation"],
                "delay",
                round(row["delay_per_requested_vehicle_s"], 2),
                flush=True,
            )
    return rows


def summarize():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for p in OUT.glob("test_*.json")]
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["controller", "saturation", "seed"])
    df.to_csv(OUT / "raw_results.csv", index=False)
    metrics = [
        "delay_per_requested_vehicle_s",
        "mean_queue_vehicles",
        "throughput",
        "conflicts_per_1000_arrivals",
        "tet_vehicle_s",
        "co2_kg",
        "pending_insertion",
        "remaining_in_network",
    ]
    summary = df.groupby(["controller", "saturation"])[metrics].agg(
        ["mean", "std", "count"]
    )
    summary.to_csv(OUT / "summary.csv")
    tests = []
    if "d3qn_forecast" in set(df.controller):
        for other in ["max_pressure", "d3qn_no_forecast"]:
            for load in [0.4, 0.7, 0.95]:
                a = df[
                    (df.controller == "d3qn_forecast") & (df.saturation == load)
                ].set_index("seed")
                b = df[(df.controller == other) & (df.saturation == load)].set_index(
                    "seed"
                )
                idx = a.index.intersection(b.index)
                if len(idx) < 2:
                    continue
                d = (
                    a.loc[idx, "delay_per_requested_vehicle_s"]
                    - b.loc[idx, "delay_per_requested_vehicle_s"]
                )
                p = (
                    1.0
                    if np.allclose(d, 0)
                    else float(wilcoxon(d, alternative="two-sided").pvalue)
                )
                tests.append(
                    {
                        "comparison": f"d3qn_forecast vs {other}",
                        "saturation": load,
                        "paired_scenarios": len(idx),
                        "mean_difference_s": float(d.mean()),
                        "p_two_sided": p,
                        "scope": "scenario variability for one training seed; no claim about training-seed robustness",
                    }
                )
    (OUT / "paired_tests.json").write_text(json.dumps(tests, indent=2))
    tune = [json.loads(p.read_text()) for p in OUT.glob("tune_*.json")]
    if tune:
        t = pd.DataFrame(tune)
        g = t.groupby("controller")[
            ["delay_per_requested_vehicle_s", "conflicts_per_1000_arrivals"]
        ].mean()
        if {"d3qn_forecast", "d3qn_safety"} <= set(g.index):
            ref = g.loc["d3qn_forecast"]
            safe = g.loc["d3qn_safety"]
            eligible = bool(
                safe.iloc[0] <= 1.05 * ref.iloc[0] and safe.iloc[1] < ref.iloc[1]
            )
            result = {
                "selected_weight": 0.5 if eligible else 1.0,
                "candidate_weights_evaluated": [0.5, 1.0],
                "passes_delay_and_conflict_gate": eligible,
                "reference": ref.to_dict(),
                "safety_candidate": safe.to_dict(),
                "selection_split": "tune only",
                "note": "Pilot compares one safety weight. Full 0/.25/.5/.75 sweep is not completed.",
            }
            (OUT / "safety_selection.json").write_text(json.dumps(result, indent=2))
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--names",
        default="webster,max_pressure,forecast_rule,d3qn_no_forecast,d3qn_forecast,d3qn_safety",
    )
    p.add_argument("--start-seed", type=int, default=200)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--loads", default=".4,.7,.95")
    p.add_argument("--split", choices=["train", "tune", "test"], default="test")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--summarize-only", action="store_true")
    ar = p.parse_args()
    if not ar.summarize_only:
        sweep(
            ar.names.split(","),
            range(ar.start_seed, ar.start_seed + ar.n_seeds),
            [float(v) for v in ar.loads.split(",")],
            ar.split,
            ar.workers,
            ar.duration,
        )
    summarize()
