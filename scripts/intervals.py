"""Prediction intervals calibrated with the residuals of earlier evaluation years (eq. 12) and their coverage.

python scripts/intervals.py --estimates results/estimates_2019_2025.csv --level 0.95
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwsf.evaluation import interval_summary  # noqa: E402
from hwsf.uncertainty import forward_intervals  # noqa: E402

METHODS = {
    "hwsf": ("gaussian", True),            # equation (12) with kappa = max(u_in, 1)
    "constant-scale": ("gaussian", False),  # kappa = 1
    "split-conformal": ("conformal", False),
    "input-scaled-conformal": ("conformal", True),
    "year-block-conformal": ("year_block", False),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--estimates", default="results/estimates_2019_2025.csv")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.68, 0.90, 0.95])
    ap.add_argument("--output", help="optional CSV with the intervals of every record (method hwsf)")
    args = ap.parse_args()
    frame = pd.read_csv(args.estimates, dtype={"area_code": str}).rename(
        columns={"observed_yield": "y_true", "input_standard_uncertainty": "u_input"})
    rows, keep = [], []
    for name, (method, scaled) in METHODS.items():
        for level in args.levels:
            iv = forward_intervals(frame, level, method, scaled)
            rows.append(dict(method=name, level=level, **interval_summary(iv).to_dict("records")[0]))
            if name == "hwsf":
                keep.append(iv.assign(level=level))
    pd.set_option("display.float_format", "{:.3f}".format)
    print(pd.DataFrame(rows).to_string(index=False))
    if args.output:
        pd.concat(keep).to_csv(args.output, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
