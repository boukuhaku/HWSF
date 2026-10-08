"""Annual refitting of HWSF (Algorithm 1): fit the origins 2012..T in order and save the checkpoints of the
evaluated target years.

python scripts/train.py --dataset data/dataset.npz --target-years 2019 2020 2021 2022 2023 2024 2025 --output runs/hwsf

Each origin t uses only the records of the years before t. Origins 2012..2018 are fitted without the correction
networks; they provide the earlier-origin predictions that later origins need (docs/training.md). Records and
checkpoints already present in --output are reused, so an interrupted run can be resumed.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwsf.training import fit_origins, load_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="training dataset (.npz), see docs/training.md")
    ap.add_argument("--target-years", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args()
    data = load_dataset(args.dataset)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)  # noqa: E731
    fit_origins(data, args.target_years, args.output, evaluate_from=min(args.target_years), device=args.device, log=log)


if __name__ == "__main__":
    main()
