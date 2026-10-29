"""Reproducible guarded D3QN continuation, validation, fresh paired evaluation.

python -m atsc.continuation train --episodes 30 [--resume]
python -m atsc.continuation evaluate --workers 3
The protocol is fixed before inspecting the new test results.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .control import Baseline, Replay, checkpoint_path
from .data import ROOT
from .guarded import ClockEnvironment, MAX_RED, masked_values
from .models import DuelingQ

OUT = ROOT / "artifacts" / "continuation"
PREDICTOR = ROOT / "artifacts" / "forecast" / "simulation_tgcn.pt"
SOURCE = checkpoint_path("d3qn_forecast", 0)
PROTOCOL = {
    "version": "guarded-v1",
    "source_episodes": 30,
    "additional_episodes": 30,
    "training_seed": 0,
    "training_scenario_seeds": list(range(10000, 10030)),
    "training_saturation": 0.7,
    "validation_seeds": [500, 501, 502],
    "validation_saturation": 0.7,
    "selection": "minimum mean validation delay among episodes 0, 10, 20, 30",
    "test_seeds": [1000, 1001, 1002, 1003],
    "test_saturations": [0.4, 0.7, 0.95],
    "controllers": [
        "webster",
        "max_pressure",
        "d3qn_original",
        "d3qn_guard_only",
        "d3qn_guard_trained",
    ],
    "red_limit_s": MAX_RED,
    "duration_s": 3600,
    "warmup_s": 3600,
    "weight": 1.0,
    "scope": "one synthetic intersection, one training seed; exploratory paired comparison",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def save_torch(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def expanded_model():
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model = DuelingQ(56)
    state = dict(source["state_dict"])
    weights = state["body.0.weight"]
    state["body.0.weight"] = torch.cat([weights, torch.zeros(len(weights), 4)], dim=1)
    model.load_state_dict(state)
    return model


def evaluate_one(name, seed, saturation, split="test", model_path=None, record=False):
    torch.set_num_threads(1)
    is_agent = name.startswith("d3qn")
    guarded = name in ("d3qn_guard_only", "d3qn_guard_trained")
    path = model_path or (
        OUT / "selected.pt" if name == "d3qn_guard_trained" else SOURCE
    )
    model = None
    ck = None
    if is_agent:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = DuelingQ(ck["inputs"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
    env = ClockEnvironment(
        PREDICTOR if is_agent else None, enforce=guarded, record=record
    )
    state = env.reset(seed, saturation, split)
    baseline = None if is_agent else Baseline(name, saturation)
    interventions = 0
    try:
        while True:
            if model is None:
                action = baseline.action(env)
            else:
                with torch.no_grad():
                    values = model(torch.from_numpy(state[: ck["inputs"]])[None])[
                        0
                    ].numpy()
                action = int(values.argmax())
                if guarded:
                    mask = env.action_mask()
                    interventions += int(not mask[action])
                    action = int(np.where(mask, values, -np.inf).argmax())
            state, _, done, _ = env.step(action)
            if done:
                break
        row = {
            "controller": name,
            "seed": seed,
            "saturation": saturation,
            "split": split,
            "controller_sha256": digest(path) if is_agent else None,
            "predictor_sha256": digest(PREDICTOR) if is_agent else None,
            "protocol_sha256": digest(OUT / "protocol.json"),
            "additional_training_episodes": ck.get("additional_episodes", 0)
            if ck
            else 0,
            "guard_overrides": interventions,
            **env.metrics(),
        }
        if record:
            save_json(
                OUT / "replay.json",
                {
                    "controller": name,
                    "seed": seed,
                    "saturation": saturation,
                    "frames": env.trace,
                    "metrics": row,
                },
            )
        return row
    finally:
        env.close()


def validate(model_path, episode):
    rows = [
        evaluate_one("d3qn_guard_trained", seed, 0.7, "tune", model_path)
        for seed in PROTOCOL["validation_seeds"]
    ]
    result = {
        "episode": episode,
        "mean_delay": float(
            np.mean([r["delay_per_requested_vehicle_s"] for r in rows])
        ),
        "runs": rows,
    }
    save_json(OUT / f"validation_{episode:03d}.json", result)
    print(
        "validation", episode, "mean delay", round(result["mean_delay"], 3), flush=True
    )
    return result


def train(episodes=30, resume=False):
    if episodes != PROTOCOL["additional_episodes"]:
        raise ValueError("This saved protocol uses exactly 30 continuation episodes")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    OUT.mkdir(exist_ok=True)
    protocol = {
        **PROTOCOL,
        "source_sha256": digest(SOURCE),
        "predictor_sha256": digest(PREDICTOR),
    }
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Existing protocol differs; preserve prior experiment")
    save_json(protocol_path, protocol)
    if (OUT / "resume.pt").exists() and not resume:
        raise RuntimeError("Existing run found; use --resume rather than overwrite")
    model = expanded_model()
    target = DuelingQ(56)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = Replay(capacity=12000, seed=0)
    rows = []
    steps = 0
    best = float("inf")
    norm = json.loads(
        (ROOT / "artifacts" / "control" / "normalization.json").read_text()
    )

    def inference_checkpoint(ep):
        return {
            "state_dict": model.state_dict(),
            "inputs": 56,
            "additional_episodes": ep,
            "source_episodes": 30,
            "source_sha256": digest(SOURCE),
            "red_limit_s": MAX_RED,
            "protocol_sha256": digest(protocol_path),
            "with_forecast": True,
            "weight": 1.0,
            "training_seed": 0,
        }

    def select(ep):
        nonlocal best
        path = OUT / f"candidate_{ep:03d}.pt"
        save_torch(path, inference_checkpoint(ep))
        result = validate(path, ep)
        if result["mean_delay"] < best:
            best = result["mean_delay"]
            save_torch(OUT / "selected.pt", inference_checkpoint(ep))
            save_json(
                OUT / "selection.json",
                {
                    "episode": ep,
                    "validation_mean_delay": best,
                    "criterion": PROTOCOL["selection"],
                    "checkpoint_sha256": digest(OUT / "selected.pt"),
                },
            )

    if resume:
        ck = torch.load(OUT / "resume.pt", map_location="cpu", weights_only=False)
        if ck["protocol_sha256"] != digest(protocol_path):
            raise RuntimeError("Resume protocol mismatch")
        model.load_state_dict(ck["state_dict"])
        target.load_state_dict(ck["target_state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        buffer = ck["replay"]
        rng.bit_generator.state = ck["numpy_rng"]
        torch.set_rng_state(ck["torch_rng"])
        rows, steps, best = ck["history"], ck["steps"], ck["best_validation_delay"]
    else:
        select(0)

    env = ClockEnvironment(
        PREDICTOR, enforce=True, dscale=norm["dscale"], cscale=norm["cscale"]
    )
    start = time.time()
    for ep in range(len(rows), episodes):
        state = env.reset(10000 + ep, 0.7, "train")
        epsilon = max(0.05, 0.3 - 0.25 * ep / 23)
        total = 0
        losses = []
        try:
            while True:
                if rng.random() < epsilon:
                    action = int(rng.choice(np.flatnonzero(env.action_mask())))
                else:
                    with torch.no_grad():
                        action = int(
                            masked_values(model, torch.from_numpy(state)[None]).argmax()
                        )
                ns, reward, done, info = env.step(action)
                buffer.add(state, action, reward, ns, float(done), info["duration"])
                state = ns
                total += reward
                steps += 1
                if len(buffer.data) >= 256:
                    idx, (bs, ba, br, bn, bd, bt), iw = buffer.sample(
                        32, 0.4 + 0.6 * ep / 29
                    )
                    with torch.no_grad():
                        best_actions = masked_values(model, bn).argmax(1, keepdim=True)
                        y = br + (1 - bd) * torch.pow(0.99, bt / 10) * target(
                            bn
                        ).gather(1, best_actions).squeeze(1)
                    pred = model(bs).gather(1, ba.long()[:, None]).squeeze(1)
                    td = y - pred
                    loss = (
                        iw
                        * torch.nn.functional.smooth_l1_loss(pred, y, reduction="none")
                    ).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step()
                    buffer.priority[idx] = np.abs(td.detach().numpy()) + 1e-5
                    losses.append(float(loss))
                if steps % 500 == 0:
                    target.load_state_dict(model.state_dict())
                if done:
                    break
            row = {
                "episode": ep + 1,
                "epsilon": epsilon,
                "reward": total,
                "loss": float(np.mean(losses)) if losses else None,
                **env.metrics(),
            }
        finally:
            env.close()
        rows.append(row)
        save_json(OUT / "training_history.json", rows)
        print(
            "guarded episode",
            ep + 1,
            "delay",
            round(row["delay_per_requested_vehicle_s"], 2),
            "max red",
            row["max_without_green_s"],
            "seconds",
            round(time.time() - start),
            flush=True,
        )
        if (ep + 1) % 10 == 0:
            select(ep + 1)
        save_torch(
            OUT / "resume.pt",
            {
                **inference_checkpoint(ep + 1),
                "target_state_dict": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "replay": buffer,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "history": rows,
                "steps": steps,
                "best_validation_delay": best,
            },
        )


def cached_eval(job):
    name, seed, saturation = job
    path = OUT / "evaluation" / f"{name}_{seed}_{saturation:.2f}.json"
    ckpath = OUT / "selected.pt" if name == "d3qn_guard_trained" else SOURCE
    expected = digest(ckpath) if name.startswith("d3qn") else None
    if path.exists():
        row = json.loads(path.read_text())
        if (
            row.get("controller_sha256") == expected
            and row.get("protocol_sha256") == digest(OUT / "protocol.json")
            and row.get("predictor_sha256")
            == (digest(PREDICTOR) if name.startswith("d3qn") else None)
        ):
            return row
    row = evaluate_one(name, seed, saturation)
    save_json(path, row)
    print(
        name,
        seed,
        saturation,
        "delay",
        round(row["delay_per_requested_vehicle_s"], 2),
        "red",
        row["max_without_green_s"],
        flush=True,
    )
    return row


def evaluate(workers=3, baselines_only=False):
    names = PROTOCOL["controllers"][:-1] if baselines_only else PROTOCOL["controllers"]
    jobs = [
        (name, seed, sat)
        for sat in PROTOCOL["test_saturations"]
        for seed in PROTOCOL["test_seeds"]
        for name in names
    ]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as pool:
        rows = list(pool.map(cached_eval, jobs))
    save_json(
        OUT / ("baseline_results.json" if baselines_only else "test_results.json"), rows
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate", "replay"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        train(args.episodes, args.resume)
    elif args.command == "evaluate":
        evaluate(args.workers, args.baselines_only)
    else:
        print(
            json.dumps(
                evaluate_one("d3qn_guard_trained", 1000, 0.7, record=True), indent=2
            )
        )
