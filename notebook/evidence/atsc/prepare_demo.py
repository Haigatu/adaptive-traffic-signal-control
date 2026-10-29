"""Download the demo's public demand dataset and verify the recorded SHA-256."""

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "data/PEMS08.npz"
    entry = json.loads((ROOT / "data/manifest.json").read_text())["PEMS08.npz"]
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("Existing PeMS08 does not match the experiment manifest")
        print("PeMS08 verified.")
        return
    temporary = target.with_suffix(".npz.part")
    print("Downloading PeMS08 (18.5 MB)...", flush=True)
    with urllib.request.urlopen(entry["url"], timeout=120) as response:
        with temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != entry["sha256"]:
        raise RuntimeError("Downloaded PeMS08 checksum mismatch")
    temporary.replace(target)
    print("PeMS08 verified and ready.")


if __name__ == "__main__":
    main()
