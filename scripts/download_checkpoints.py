"""Download the released checkpoints (target years 2019-2025) and check their SHA-256 sums.

python scripts/download_checkpoints.py --output checkpoints
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://github.com/boukuhaku/HWSF/releases/download/v1.0.0/"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default="checkpoints")
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2019, 2026)))
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE + "SHA256SUMS", timeout=60) as r:
        sums = dict(line.split()[::-1] for line in r.read().decode().splitlines() if line.strip())
    for year in args.years:
        name = f"hwsf_{year}.pt"
        path = out / name
        if not path.exists():
            print("downloading", name)
            urllib.request.urlretrieve(BASE + name, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sums[name]:
            raise SystemExit(f"checksum mismatch for {name}")
        print("ok", path)


if __name__ == "__main__":
    main()
