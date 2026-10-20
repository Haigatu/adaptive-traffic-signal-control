"""Reproduce the delivered pilot; test evaluation is skipped unless asked."""

import argparse, subprocess, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args):
    subprocess.run([sys.executable, "-u", "-m", *args], cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-epochs", type=int, default=6)
    p.add_argument("--agent-episodes", type=int, default=30)
    p.add_argument("--evaluation-seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--evaluate", action="store_true")
    ar = p.parse_args()
    run("atsc.data")
    run("atsc.network")
    for ds in ["PEMS04", "PEMS08"]:
        run("atsc.forecast", "--dataset", ds, "--epochs", str(ar.forecast_epochs))
    run("atsc.finetune")
    run("atsc.control", "calibrate")
    for name in ["d3qn_no_forecast", "d3qn_forecast", "d3qn_safety"]:
        run(
            "atsc.control",
            "train",
            "--name",
            name,
            "--episodes",
            str(ar.agent_episodes),
        )
    run(
        "atsc.experiment",
        "--names",
        "d3qn_forecast,d3qn_safety",
        "--split",
        "tune",
        "--start-seed",
        "100",
        "--n-seeds",
        "10",
        "--loads",
        ".7",
        "--workers",
        str(ar.workers),
    )
    if ar.evaluate:
        selection = json.loads(
            (ROOT / "artifacts/evaluation/safety_selection.json").read_text()
        )
        names = "webster,max_pressure,forecast_rule,d3qn_no_forecast,d3qn_forecast"
        if selection["passes_delay_and_conflict_gate"]:
            names += ",d3qn_safety"
        run(
            "atsc.experiment",
            "--names",
            names,
            "--n-seeds",
            str(ar.evaluation_seeds),
            "--workers",
            str(ar.workers),
        )


if __name__ == "__main__":
    main()
