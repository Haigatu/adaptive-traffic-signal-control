"""Collect real loop counts under fixed timing and adapt PeMS T-GCN to SUMO."""

import json
import numpy as np
import torch
import multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from .data import ROOT
from .models import Forecaster
from .environment import Environment
from .forecast import scores


def collect_one(task):
    split, k = task
    torch.set_num_threads(1)
    cache = ROOT / "data" / "loop_cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"{split}_{k}.npy"
    if path.exists():
        return split, k, np.load(path)
    # Adapt on low/medium demand; high demand remains a transfer stress test.
    e = Environment(duration=7200, warmup=3600)
    e.reset(7000 + k, [0.4, 0.7][k % 2], split)
    import libsumo as sim

    while sim.simulation.getTime() < 10800:
        t = sim.simulation.getTime()
        e._signal(int(t // 30) % 4, t % 30 >= 27)
        e._tick(measure=False)
    data = np.stack(e.all_counts)
    e.close()
    np.save(path, data)
    return split, k, data


def collect():
    all_parts = {}
    out = ROOT / "data" / "simulation_counts.npz"
    tasks = [
        (s, k) for s, n in [("train", 12), ("tune", 4), ("test", 4)] for k in range(n)
    ]
    with mp.get_context("spawn").Pool(3) as pool:
        for split, k, data in pool.imap_unordered(collect_one, tasks):
            all_parts.setdefault(split, []).append((k, data))
            print("counts", split, k + 1, flush=True)
    all_parts = {s: np.stack([d for k, d in sorted(v)]) for s, v in all_parts.items()}
    np.savez_compressed(out, **all_parts)
    return all_parts


def run(epochs=60):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    file = ROOT / "data" / "simulation_counts.npz"
    parts = dict(np.load(file)) if file.exists() else collect()
    ck = torch.load(
        ROOT / "artifacts" / "forecast" / "PEMS08" / "tgcn.pt", weights_only=False
    )
    m = Forecaster("tgcn", ck["hidden"])
    m.load_state_dict(ck["state_dict"])
    # Keep pretraining normalization to preserve the checkpoint's input semantics.
    mean, std = ck["mean"], ck["std"]
    a = torch.eye(4).to_sparse()
    sets = {}
    for part, raw in parts.items():
        xs = []
        ys = []
        for series in raw:
            for t in range(12, len(series) - 2):
                xs.append(series[t - 12 : t])
                ys.append(series[t : t + 3])
        sets[part] = ((np.stack(xs) - mean) / std, (np.stack(ys) - mean) / std)
    train = DataLoader(
        TensorDataset(*[torch.from_numpy(v) for v in sets["train"]]),
        batch_size=32,
        shuffle=True,
    )
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)
    best = float("inf")
    history = []
    path = ROOT / "artifacts" / "forecast" / "simulation_tgcn.pt"
    for ep in range(epochs):
        m.train()
        for x, y in train:
            loss = torch.nn.functional.l1_loss(m(x, a), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
            opt.step()
        m.eval()
        with torch.no_grad():
            val = (
                float(
                    torch.mean(
                        torch.abs(
                            m(torch.from_numpy(sets["tune"][0]), a)
                            - torch.from_numpy(sets["tune"][1])
                        )
                    )
                )
                * std
            )
        history.append({"epoch": ep + 1, "validation_mae": val})
        if val < best:
            best = val
            torch.save(
                {
                    "state_dict": m.state_dict(),
                    "kind": "tgcn",
                    "hidden": ck["hidden"],
                    "mean": mean,
                    "std": std,
                    "pretrained_on": "PEMS08",
                    "finetuned_on": "SUMO induction loops",
                    "best_epoch": ep + 1,
                },
                path,
            )
    m.load_state_dict(torch.load(path, weights_only=False)["state_dict"])
    with torch.no_grad():
        p = m(torch.from_numpy(sets["test"][0]), a).numpy() * std + mean
    y = sets["test"][1] * std + mean
    persist = np.repeat((sets["test"][0][:, -1:] * std + mean), 3, axis=1)
    result = {
        "tgcn": scores(y, np.maximum(p, 0)),
        "persistence": scores(y, persist),
        "training_windows": len(sets["train"][0]),
        "validation_windows": len(sets["tune"][0]),
        "test_windows": len(sets["test"][0]),
        "history": history,
    }
    (path.with_suffix(".json")).write_text(json.dumps(result, indent=2))
    print(result["tgcn"], flush=True)


if __name__ == "__main__":
    run()
