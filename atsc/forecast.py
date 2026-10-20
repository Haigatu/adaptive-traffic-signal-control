"""Train/evaluate every forecaster on chronological, full-node PeMS splits."""

import argparse, json, time, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import ROOT, load, Windows
from .models import Forecaster


def scores(y, p):
    e = p - y
    mask = y > 0
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e * e).mean())),
        "mape_nonzero_percent": float((np.abs(e[mask]) / y[mask]).mean() * 100),
        "mape_coverage": float(mask.mean()),
    }


@torch.no_grad()
def predict(model, loader, a, mean, std):
    model.eval()
    ys = []
    ps = []
    for x, y in loader:
        ps.append(model(x, a).numpy() * std + mean)
        ys.append(y.numpy() * std + mean)
    return np.concatenate(ys), np.maximum(np.concatenate(ps), 0)


def run(ds, kind, epochs=8, stride=3, seed=42, hidden=24):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    out = ROOT / "artifacts" / "forecast" / ds
    out.mkdir(parents=True, exist_ok=True)
    raw, adj = load(ds)
    n = len(raw)
    c1 = int(0.6 * n)
    c2 = int(0.8 * n)
    mean = float(raw[:c1].mean())
    std = float(raw[:c1].std())
    sets = [
        Windows(raw, 0, c1, mean, std, stride),
        Windows(raw, c1, c2, mean, std, 3),
        Windows(raw, c2, n, mean, std),
    ]
    loaders = [DataLoader(s, batch_size=32, shuffle=i == 0) for i, s in enumerate(sets)]
    a = torch.from_numpy(adj).to_sparse()
    start = time.time()
    if kind in ("ha", "persistence"):
        idx = sets[2].indices
        y = np.stack([raw[t : t + 3] for t in idx])
        if kind == "ha":
            slots = np.arange(c1) % 288
            ha = np.stack([raw[:c1][slots == s].mean(0) for s in range(288)])
            p = np.stack([ha[(t + np.arange(3)) % 288] for t in idx])
        else:
            p = np.stack([np.repeat(raw[t - 1 : t], 3, axis=0) for t in idx])
        hist = []
    else:
        model = Forecaster(kind, hidden)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best = float("inf")
        hist = []
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0
            count = 0
            for x, yb in loaders[0]:
                pred = model(x, a)
                loss = torch.nn.functional.l1_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                loss_sum += loss.item() * len(x)
                count += len(x)
            yv, pv = predict(model, loaders[1], a, mean, std)
            val = scores(yv, pv)["mae"]
            row = {
                "epoch": epoch,
                "train_mae": loss_sum / count * std,
                "validation_mae": val,
                "seconds": time.time() - start,
            }
            hist.append(row)
            if val < best:
                best = val
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "kind": kind,
                        "hidden": hidden,
                        "mean": mean,
                        "std": std,
                        "dataset": ds,
                        "seed": seed,
                        "epoch": epoch,
                    },
                    out / f"{kind}.pt",
                )
            print(ds, kind, row, flush=True)
        ck = torch.load(out / f"{kind}.pt", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        y, p = predict(model, loaders[2], a, mean, std)
    result = {
        "dataset": ds,
        "model": kind,
        "seed": seed,
        "epochs_requested": epochs if hist else 0,
        "train_stride": stride,
        "train_samples": len(sets[0]),
        "validation_stride": 3,
        "test_samples": len(sets[2]),
        "nodes": raw.shape[1],
        "train_end": c1,
        "validation_end": c2,
        "time_points": n,
        "training_seconds": time.time() - start,
        **scores(y, p),
        "horizon_mae": [float(np.abs(y[:, i] - p[:, i]).mean()) for i in range(3)],
        "history": hist,
    }
    (out / f"{kind}.json").write_text(json.dumps(result, indent=2))
    # Small genuine held-out trace for plotting.
    np.savez_compressed(
        out / f"{kind}_trace.npz", actual=y[:288, :, 0], predicted=p[:288, :, 0]
    )
    print(
        "RESULT",
        json.dumps({k: v for k, v in result.items() if k != "history"}),
        flush=True,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="PEMS08")
    p.add_argument("--models", default="tgcn,gru,lstm,mlp,ha,persistence")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=24)
    ar = p.parse_args()
    for k in ar.models.split(","):
        run(ar.dataset, k, ar.epochs, ar.stride, ar.seed, ar.hidden)
