"""Public PeMS data, causal preparation, chronological windows."""

from pathlib import Path
import hashlib, json, urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def download():
    root = ROOT / "data"
    root.mkdir(exist_ok=True)
    old_manifest = (
        json.loads((root / "manifest.json").read_text())
        if (root / "manifest.json").exists()
        else {}
    )
    manifest = {}
    for ds in ["PEMS04", "PEMS08"]:
        for ext in ["npz", "csv"]:
            name = f"{ds}.{ext}"
            dest = root / name
            url = f"https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data/{ds}/{name}"
            if not dest.exists():
                print("Downloading", name, flush=True)
                with (
                    urllib.request.urlopen(url, timeout=120) as r,
                    open(str(dest) + ".part", "wb") as f,
                ):
                    while chunk := r.read(1024 * 1024):
                        f.write(chunk)
                Path(str(dest) + ".part").replace(dest)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            if name in old_manifest and old_manifest[name]["sha256"] != digest:
                raise RuntimeError(
                    f"{name}: SHA-256 differs from the saved experiment. Keep the original manifest and investigate the source before retraining."
                )
            manifest[name] = {
                "url": url,
                "sha256": digest,
                "bytes": dest.stat().st_size,
            }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load(ds):
    raw = np.load(ROOT / "data" / f"{ds}.npz")["data"][:, :, 0].astype("float32")
    # A zero can be real: preserve zeros. Forward fill invalids without looking ahead.
    for i in range(len(raw)):
        bad = ~np.isfinite(raw[i]) | (raw[i] < 0)
        raw[i, bad] = raw[i - 1, bad] if i else 0
    n = raw.shape[1]
    a = np.zeros((n, n), np.float32)
    edges = np.genfromtxt(ROOT / "data" / f"{ds}.csv", delimiter=",", skip_header=1)
    sigma = max(float(np.std(edges[:, 2])), 1e-6)
    for u, v, d in edges:
        w = np.exp(-((d / sigma) ** 2))
        if w >= 0.1:
            a[int(u), int(v)] = a[int(v), int(u)] = w
    a += np.eye(n, dtype=np.float32)
    scale = np.maximum(a.sum(1), 1e-6) ** -0.5
    a = scale[:, None] * a * scale[None, :]
    return raw, a


class Windows(Dataset):
    def __init__(self, raw, start, end, mean, std, stride=1):
        self.raw = raw
        self.mean = mean
        self.std = std
        # Targets entirely inside split; observed context may precede its boundary.
        self.indices = np.arange(max(start, 12), end - 2, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = self.indices[i]
        return (
            torch.from_numpy((self.raw[t - 12 : t] - self.mean) / self.std),
            torch.from_numpy((self.raw[t : t + 3] - self.mean) / self.std),
        )


if __name__ == "__main__":
    download()
