"""Double-DQN, prioritized replay, fixed/pressure/forecast controllers."""

import argparse, json, random, time, hashlib
import numpy as np
import torch
from .data import ROOT
from .models import DuelingQ
from .environment import Environment


class Replay:
    def __init__(self, capacity=50000, seed=0):
        self.capacity = capacity
        self.data = []
        self.priority = np.zeros(capacity, np.float32)
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def add(self, *transition):
        p = max(float(self.priority[: len(self.data)].max()), 1.0) if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
        self.priority[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, n, beta):
        p = self.priority[: len(self.data)] ** 0.6
        p /= p.sum()
        idx = self.rng.choice(len(p), n, p=p, replace=False)
        weights = (len(p) * p[idx]) ** (-beta)
        weights /= weights.max()
        fields = [
            torch.as_tensor(np.stack(v), dtype=torch.float32)
            for v in zip(*(self.data[i] for i in idx))
        ]
        return idx, fields, torch.from_numpy(weights)


class Baseline:
    def __init__(self, name, saturation):
        self.name = name
        self.age = 0.0
        self.last = 0
        cycle = np.clip((1.5 * 12 + 5) / max(1 - saturation, 0.05), 52, 180)
        self.green = float(np.ceil(((cycle - 12) / 4) / 10) * 10)

    def action(self, env):
        if self.name == "webster":
            if self.age >= self.green:
                self.last = (self.last + 1) % 4
                self.age = 0
            self.age += 10
            return self.last
        if self.name == "forecast_rule":
            demand = np.maximum(env.forecast.mean(0), 1.0)
            allocation = (
                10.0 + 68.0 * demand / demand.sum()
            )  # 120s target cycle incl. 12s yellow
            if self.age >= allocation[self.last]:
                self.last = (self.last + 1) % 4
                self.age = 0
            self.age += 10
            return self.last
        if self.name == "max_pressure":
            values = env.pressures()
        else:
            raise ValueError(self.name)
        # Keep current phase on a tie to avoid pointless clearance loss.
        return env.phase if values[env.phase] >= max(values) else int(np.argmax(values))


def checkpoint_path(name, seed):
    return ROOT / "artifacts" / "control" / f"{name}_seed{seed}.pt"


def evaluate(
    name, seed, saturation, split="test", duration=3600, record=False, training_seed=0
):
    torch.set_num_threads(1)
    predictor = ROOT / "artifacts" / "forecast" / "simulation_tgcn.pt"
    ck = None
    if name.startswith("d3qn"):
        ck = torch.load(
            checkpoint_path(name, training_seed), map_location="cpu", weights_only=False
        )
        q = DuelingQ(ck["inputs"])
        q.load_state_dict(ck["state_dict"])
        q.eval()
    needs_predictor = ck is not None or name == "forecast_rule"
    env = Environment(
        predictor if needs_predictor else None,
        with_forecast=(ck["with_forecast"] if ck else True),
        duration=duration,
        record=record,
    )
    s = env.reset(seed, saturation, split)
    baseline = Baseline(name, saturation) if ck is None else None
    dvals = []
    cvals = []
    latencies = []
    try:
        while True:
            decision_start = time.perf_counter()
            if ck:
                with torch.no_grad():
                    action = int(q(torch.from_numpy(s)[None]).argmax())
            else:
                action = baseline.action(env)
            latencies.append((time.perf_counter() - decision_start) * 1000)
            s, r, done, info = env.step(action)
            dvals.append(info["interval_delay"])
            cvals.append(info["interval_conflicts"])
            if done:
                break
        result = {
            "controller": name,
            "seed": seed,
            "training_seed": training_seed,
            "saturation": saturation,
            "split": split,
            **env.metrics(),
            "max_interval_delay": max(dvals),
            "max_interval_conflicts": max(cvals),
            "decision_latency_ms_mean": float(np.mean(latencies)),
            "decision_latency_ms_p95": float(np.percentile(latencies, 95)),
            "training_episodes": ck["episodes_completed"] if ck else None,
            "controller_sha256": hashlib.sha256(
                checkpoint_path(name, training_seed).read_bytes()
            ).hexdigest()
            if ck
            else None,
            "predictor_sha256": hashlib.sha256(predictor.read_bytes()).hexdigest()
            if needs_predictor
            else None,
        }
        if record:
            path = ROOT / "artifacts" / "replay.json"
            path.write_text(
                json.dumps(
                    {
                        "controller": name,
                        "seed": seed,
                        "saturation": saturation,
                        "frames": env.trace,
                        "metrics": result,
                    }
                )
            )
        return result
    finally:
        env.close()


def calibrate():
    rows = [
        evaluate("webster", seed, sat, "train")
        for seed, sat in [(11, 0.4), (12, 0.7), (13, 0.95)]
    ]
    result = {
        "dscale": max(r["max_interval_delay"] for r in rows),
        "cscale": max(1, max(r["max_interval_conflicts"] for r in rows)),
        "runs": rows,
    }
    out = ROOT / "artifacts" / "control"
    out.mkdir(exist_ok=True)
    (out / "normalization.json").write_text(json.dumps(result, indent=2))
    return result


def train(name, episodes=30, seed=0, duration=3600, weight=None):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    forecast = name != "d3qn_no_forecast"
    w = 0.5 if name == "d3qn_safety" else 1.0
    if weight is not None:
        if not 0 <= weight <= 1:
            raise ValueError("Efficiency weight must be between zero and one.")
        w = float(weight)
    predictor = ROOT / "artifacts" / "forecast" / "simulation_tgcn.pt"
    norm = json.loads(
        (ROOT / "artifacts" / "control" / "normalization.json").read_text()
    )
    env = Environment(
        predictor if forecast else None,
        forecast,
        w,
        duration=duration,
        dscale=norm["dscale"],
        cscale=norm["cscale"],
    )
    inputs = 52 if forecast else 40
    q = DuelingQ(inputs)
    target = DuelingQ(inputs)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    buffer = Replay(seed=seed)
    global_step = 0
    rows = []
    start = time.time()
    for ep in range(episodes):
        sat = 0.7
        scenario_seed = seed * 10000 + ep
        s = env.reset(scenario_seed, sat, "train")
        done = False
        total = 0
        losses = []
        epsilon = max(0.05, 1 - 0.95 * ep / max(1, int(episodes * 0.8)))
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(4))
            else:
                with torch.no_grad():
                    action = int(q(torch.from_numpy(s)[None]).argmax())
            ns, r, done, info = env.step(action)
            buffer.add(s, action, r, ns, float(done), info["duration"])
            s = ns
            total += r
            global_step += 1
            if len(buffer.data) >= 256:
                idx, (bs, ba, br, bn, bd, bt), iw = buffer.sample(
                    32, min(1, 0.4 + 0.6 * ep / episodes)
                )
                with torch.no_grad():
                    best = q(bn).argmax(1, keepdim=True)
                    # Semi-Markov adjustment for 10s holds / 13s changes.
                    y = br + (1 - bd) * torch.pow(0.99, bt / 10) * target(bn).gather(
                        1, best
                    ).squeeze(1)
                pred = q(bs).gather(1, ba.long()[:, None]).squeeze(1)
                td = y - pred
                loss = (
                    iw * torch.nn.functional.smooth_l1_loss(pred, y, reduction="none")
                ).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10)
                opt.step()
                buffer.priority[idx] = np.abs(td.detach().numpy()) + 1e-5
                losses.append(loss.item())
            if global_step % 500 == 0:
                target.load_state_dict(q.state_dict())
        metrics = env.metrics()
        env.close()
        row = {
            "episode": ep + 1,
            "epsilon": epsilon,
            "reward": total,
            "loss": float(np.mean(losses)) if losses else None,
            "saturation": sat,
            "seconds": time.time() - start,
            **metrics,
        }
        rows.append(row)
        ck = {
            "state_dict": q.state_dict(),
            "target_state_dict": target.state_dict(),
            "optimizer": opt.state_dict(),
            "name": name,
            "inputs": inputs,
            "with_forecast": forecast,
            "weight": w,
            "training_seed": seed,
            "episodes_completed": ep + 1,
            "steps": global_step,
            "normalization": norm,
            "duration": duration,
        }
        path = checkpoint_path(name, seed)
        path.parent.mkdir(exist_ok=True)
        torch.save(ck, path)
        path.with_suffix(".json").write_text(json.dumps(rows, indent=2))
        print(
            name,
            "episode",
            ep + 1,
            "reward",
            round(total, 2),
            "delay",
            round(metrics["delay_per_requested_vehicle_s"], 2),
            "conflicts",
            metrics["conflicts"],
            "seconds",
            round(time.time() - start),
            flush=True,
        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "calibrate", "evaluate"])
    p.add_argument("--name", default="d3qn_forecast")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--training-seed", type=int, default=0)
    p.add_argument("--saturation", type=float, default=0.7)
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--record", action="store_true")
    p.add_argument(
        "--weight", type=float, help="Efficiency weight for a new training run"
    )
    ar = p.parse_args()
    if ar.command == "calibrate":
        print(json.dumps(calibrate(), indent=2))
    elif ar.command == "train":
        train(ar.name, ar.episodes, ar.seed, ar.duration, ar.weight)
    else:
        print(
            json.dumps(
                evaluate(
                    ar.name,
                    ar.seed,
                    ar.saturation,
                    duration=ar.duration,
                    record=ar.record,
                    training_seed=ar.training_seed,
                ),
                indent=2,
            )
        )
