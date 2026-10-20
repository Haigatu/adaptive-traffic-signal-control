"""Bundle code, small data, weights, metrics and offline viewers, not raw PeMS."""

from pathlib import Path
import zipfile, json, hashlib, argparse

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent / "deliverable" / "ATSC_Simulation_Trained.zip"


def include(p):
    r = p.relative_to(ROOT)
    if any(part in r.parts for part in ("__pycache__", ".git", ".ruff_cache", ".pytest_cache")):
        return False
    if r.parts[:2] == ("data", "runs") or r.parts[:2] == ("data", "loop_cache"):
        return False
    if r.parent == Path("data") and r.name in ["PEMS04.npz", "PEMS08.npz"]:
        return False
    if r.suffix in [".part", ".tmp"]:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEST)
    DEST = parser.parse_args().output.resolve()
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and include(p))
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.name != "MANIFEST_SHA256.json"
    }
    (ROOT / "MANIFEST_SHA256.json").write_text(json.dumps(manifest, indent=2))
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and include(p))
    DEST.parent.mkdir(exist_ok=True)
    staging = DEST.with_suffix(".zip.partial")
    with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, Path("ATSC_Project") / p.relative_to(ROOT))
    staging.replace(DEST)
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        print(
            json.dumps(
                {
                    "path": str(DEST),
                    "bytes": DEST.stat().st_size,
                    "files": len(z.namelist()),
                }
            )
        )
