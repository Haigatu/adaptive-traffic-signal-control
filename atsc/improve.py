"""Separate policy-distillation experiment; never overwrite original artifacts."""

import argparse
import hashlib
import json
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .data import ROOT
from .guarded import ClockEnvironment, state_masks
from .models import DuelingQ

OUT = ROOT / "artifacts/improvement"
PREDICTOR = ROOT / "artifacts/forecast/simulation_tgcn.pt"
CONTROLLERS = ["d3qn_no_forecast", "d3qn_forecast", "guarded", "improved"]


def improved_checkpoint(load):
    """Choose the validation winner for the closest configured demand level."""
    selection = json.loads((OUT / "load_selection.json").read_text())
    level = min(selection["checkpoints"], key=lambda x: abs(float(x) - load))
    return OUT / selection["checkpoints"][level]


def select_by_load():
    """Second protocol: select only from validation, reserve fresh test seeds."""
    path = OUT / "load_selection.json"
    if path.exists():
        raise RuntimeError("Load selection already exists; refusing overwrite")
    mapping = {}
    for load in [.4, .7, .95]:
        rows = [json.loads((OUT / f"validation_{bonus}_{load}.json").read_text())
                for bonus in [5, 15, 30]]
        mapping[str(load)] = min(rows, key=lambda r:
                                r["delay_per_requested_vehicle_s"])["checkpoint"]
    save(path, {"checkpoints": mapping, "criterion": "minimum validation delay per load",
                "interpolation": "nearest configured load",
                "test_seeds": [1120, 1121], "test_loads": [.4, .7, .95],
                "duration": 3600, "warmup": 3600,
                "reason": "Global selection regressed under congested test demand."})
    print(mapping, flush=True)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def teacher_scores(states, switch_bonus):
    """Queue, causal forecast, accumulated waiting and service continuity."""
    states = np.atleast_2d(states)
    queues = states[:, :12].reshape(-1, 4, 3).sum(2) * 80
    waiting = states[:, 12:24].reshape(-1, 4, 3).sum(2) * 2400
    forecast = states[:, 40:52].reshape(-1, 3, 4).mean(1) * 120 / 300
    ages = states[:, -4:] * 120
    return (queues + forecast * 20 + waiting / 240 + ages / 10
            + states[:, 36:40] * switch_bonus)


def train():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    source = ROOT / "artifacts/continuation/resume.pt"
    ck = torch.load(source, map_location="cpu", weights_only=False)
    replay = ck["replay"].data
    states = np.stack([r[0] for r in replay] + [r[3] for r in replay])
    masks = state_masks(states).numpy()
    rng = np.random.default_rng(42)
    order = rng.permutation(len(states))
    train_ids, check_ids = order[:int(.9*len(order))], order[int(.9*len(order)):]
    x = torch.from_numpy(states)
    protocol = {
        "method": "queue-aware policy distillation into the DuelingQ architecture",
        "source": "artifacts/continuation/resume.pt",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "training_states": len(train_ids), "heldout_training_states": len(check_ids),
        "bonuses": [5, 15, 30], "epochs": 40, "seed": 42,
        "validation": {"split": "tune", "seed": 610, "loads": [.4, .7, .95],
                       "duration": 1800, "warmup": 3600},
        "test": {"split": "test", "seeds": [1110, 1111],
                 "loads": [.4, .7, .95], "duration": 3600, "warmup": 3600},
    }
    if (OUT / "protocol.json").exists():
        raise RuntimeError("Improvement experiment already exists; refusing overwrite")
    save(OUT / "protocol.json", protocol)
    for bonus in protocol["bonuses"]:
        torch.manual_seed(42)
        model = DuelingQ(56)
        model.load_state_dict(ck["state_dict"])
        target = np.where(masks, teacher_scores(states, bonus), -np.inf).argmax(1)
        y = torch.from_numpy(target)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        for epoch in range(protocol["epochs"]):
            ids = rng.permutation(train_ids)
            for start in range(0, len(ids), 256):
                batch = ids[start:start+256]
                logits = model(x[batch]).masked_fill(
                    ~torch.from_numpy(masks[batch]), -1e9)
                loss = torch.nn.functional.cross_entropy(logits, y[batch])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = model(x[check_ids]).masked_fill(
                ~torch.from_numpy(masks[check_ids]), -1e9).argmax(1)
            accuracy = float((predicted == y[check_ids]).float().mean())
        path = OUT / f"candidate_{bonus}.pt"
        torch.save({"state_dict": model.state_dict(), "inputs": 56,
                    "with_forecast": True, "method": protocol["method"],
                    "switch_bonus": bonus, "red_limit_s": 120,
                    "training_label_accuracy": accuracy}, path)
        print("trained", path.name, "label accuracy", round(accuracy, 4), flush=True)


def evaluate(path, seed, load, split, duration, run_dir=None):
    torch.set_num_threads(1)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = DuelingQ(ck["inputs"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    env = ClockEnvironment(PREDICTOR, enforce=True, duration=duration, run_dir=run_dir)
    start = time.time()
    try:
        state = env.reset(seed, load, split)
        while True:
            with torch.inference_mode():
                q = model(torch.from_numpy(state)[None])[0].numpy()
            action = int(np.where(env.action_mask(), q, -np.inf).argmax())
            state, _, done, _ = env.step(action)
            if done:
                break
        result = {"checkpoint": path.name,
                  "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "predictor_sha256": hashlib.sha256(PREDICTOR.read_bytes()).hexdigest(),
                  "seed": seed, "saturation": load, "split": split,
                  "duration": duration, "warmup": env.warmup,
                  **env.metrics()}
        print(path.name, seed, load, "delay",
              round(result["delay_per_requested_vehicle_s"], 2),
              "seconds", round(time.time()-start), flush=True)
        return result
    finally:
        env.close()


def validate():
    protocol = json.loads((OUT / "protocol.json").read_text())
    rows = []
    for bonus in protocol["bonuses"]:
        for load in protocol["validation"]["loads"]:
            path = OUT / f"validation_{bonus}_{load}.json"
            row = json.loads(path.read_text()) if path.exists() else evaluate(
                OUT / f"candidate_{bonus}.pt", 610, load, "tune", 1800)
            save(path, row)
            rows.append(row)
    # Equal weight to each load via relative delay against candidate 5.
    reference = {r["saturation"]: r["delay_per_requested_vehicle_s"]
                 for r in rows if r["checkpoint"] == "candidate_5.pt"}
    scores = {bonus: np.mean([
        r["delay_per_requested_vehicle_s"]/reference[r["saturation"]]
        for r in rows if r["checkpoint"] == f"candidate_{bonus}.pt"
    ]) for bonus in protocol["bonuses"]}
    selected = min(scores, key=scores.get)
    selected_path = OUT / "selected.pt"
    selected_path.write_bytes((OUT / f"candidate_{selected}.pt").read_bytes())
    save(OUT / "selection.json", {"bonus": selected, "scores": scores,
                                 "criterion": "minimum equal-load normalized validation delay"})
    print("SELECTED", selected, scores, flush=True)


def test_job(job):
    name, seed, load, *options = job
    by_load = bool(options and options[0])
    path = OUT / ("test_by_load" if by_load else "test") / f"{name}_{seed}_{load}.json"
    if name in ("improved", "guarded"):
        checkpoint = ((improved_checkpoint(load) if by_load else OUT / "selected.pt")
                      if name == "improved" else
                      ROOT / "artifacts/continuation/selected.pt")
    else:
        from .control import checkpoint_path
        checkpoint = checkpoint_path(name, 0)
    if path.exists():
        cached = json.loads(path.read_text())
        recorded = cached.get("checkpoint_sha256", cached.get("controller_sha256"))
        if recorded != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError(f"Stale checkpoint in cached result: {path}")
        return cached
    run_dir = ROOT / "demo_runs/improvement" / f"{name}_{seed}_{load}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if name in ("improved", "guarded"):
        result = evaluate(checkpoint, seed, load, "test", 3600, run_dir=run_dir)
    else:
        from .control import evaluate as original_evaluate
        result = original_evaluate(name, seed, load, run_dir=run_dir)
        print(name, seed, load, "delay",
              round(result["delay_per_requested_vehicle_s"], 2), flush=True)
    result["controller"] = name
    route = run_dir / f"test_{seed}_{load:.2f}_7200.rou.xml"
    result["route_sha256"] = hashlib.sha256(route.read_bytes()).hexdigest()
    result["duration"] = 3600
    result["warmup"] = 3600
    save(path, result)
    return result


def test(by_load=False):
    protocol = json.loads((OUT / "protocol.json").read_text())
    suffix = "_by_load" if by_load else ""
    if by_load:
        selection = json.loads((OUT / "load_selection.json").read_text())
        protocol["test"]["seeds"] = selection["test_seeds"]
    save(OUT / f"test_protocol{suffix}.json", {
        **protocol["test"], "controllers": CONTROLLERS,
        "checkpoint_sha256": {
            str(load): hashlib.sha256((improved_checkpoint(load) if by_load
                                      else OUT / "selected.pt").read_bytes()).hexdigest()
            for load in protocol["test"]["loads"]},
    })
    jobs = [(name, seed, load, by_load)
            for name in CONTROLLERS
            for seed in protocol["test"]["seeds"]
            for load in protocol["test"]["loads"]]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(test_job, jobs))
    for seed in protocol["test"]["seeds"]:
        for load in protocol["test"]["loads"]:
            sample = [r for r in rows if r["seed"] == seed and r["saturation"] == load]
            if len({r["route_sha256"] for r in sample}) != 1:
                raise ValueError("Unmatched test demand")
            if any(r["elapsed_s"] != 3600 for r in sample):
                raise ValueError("Unmatched test horizon")
    save(OUT / f"test_results{suffix}.json", rows)
    summaries = []
    for load in protocol["test"]["loads"]:
        group = {}
        for name in CONTROLLERS:
            sample = [r for r in rows if r["controller"] == name and r["saturation"] == load]
            group[name] = {key: float(np.mean([r[key] for r in sample]))
                           for key in ["delay_per_requested_vehicle_s",
                                       "mean_queue_vehicles", "throughput"]}
        summaries.append({"saturation": load, "controllers": group})
    save(OUT / f"summary{suffix}.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "validate", "test", "select-by-load", "test-by-load"])
    args = parser.parse_args()
    {"train": train, "validate": validate, "test": test,
     "select-by-load": select_by_load, "test-by-load": lambda: test(True)}[args.command]()
