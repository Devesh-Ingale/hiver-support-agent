"""Download twcs.csv from Kaggle and record a manifest (size, sha256, rows) for reproducibility."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "thoughtvector/customer-support-on-twitter"


def _kaggle_executable() -> str:
    exe = shutil.which("kaggle")
    if exe:
        return exe
    candidate = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("kaggle CLI not found; `pip install kaggle` and put kaggle.json in ~/.kaggle")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(raw_dir: Path, force: bool = False) -> Path:
    """Fetch the dataset into raw_dir; returns the path of twcs.csv. Idempotent unless force=True."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv = raw_dir / "twcs.csv"
    if csv.exists() and not force:
        return csv
    subprocess.run(
        [_kaggle_executable(), "datasets", "download", "-d", DATASET, "-p", str(raw_dir), "--unzip", "--force"],
        check=True,
    )
    # the archive unpacks to raw/twcs/twcs.csv (plus sample.csv); flatten so paths are stable
    nested = raw_dir / "twcs" / "twcs.csv"
    if nested.exists():
        shutil.move(str(nested), str(csv))
        shutil.rmtree(raw_dir / "twcs", ignore_errors=True)
    if not csv.exists():
        found = list(raw_dir.rglob("twcs.csv"))
        if not found:
            raise FileNotFoundError(f"twcs.csv not found under {raw_dir} after download")
        shutil.move(str(found[0]), str(csv))
    write_manifest(csv)
    return csv


def write_manifest(csv: Path) -> Path:
    rows = -1  # header excluded
    with csv.open("rb") as f:
        for rows, _ in enumerate(f):
            pass
    manifest = {"file": csv.name, "bytes": csv.stat().st_size, "sha256": sha256_of(csv), "lines_excl_header": rows}
    out = csv.with_name("MANIFEST.json")
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
