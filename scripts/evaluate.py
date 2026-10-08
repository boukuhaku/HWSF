"""Errors of HWSF over the evaluation records: overall, by year, by prefecture, and the MSE decomposition (eq. 3).

python scripts/evaluate.py --estimates results/estimates_2019_2025.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwsf.evaluation import mse_decomposition, summary_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--estimates", default="results/estimates_2019_2025.csv")
    ap.add_argument("--column", default="estimate", help="prediction column to score")
    args = ap.parse_args()
    frame = pd.read_csv(args.estimates, dtype={"area_code": str})
    pd.set_option("display.float_format", "{:.3f}".format)
    print("Overall\n", summary_table(frame, args.column, "observed_yield").to_string(index=False), "\n")
    print("By year\n", summary_table(frame, args.column, "observed_yield", by="year").to_string(index=False), "\n")
    print("By prefecture\n", summary_table(frame, args.column, "observed_yield", by="prefecture").to_string(index=False), "\n")
    d = mse_decomposition(frame, args.column, "observed_yield")
    print(f"MSE decomposition: prefecture-year {d['prefecture_year']:.3f}, municipal {d['municipal']:.3f} (kg/10a)^2")


if __name__ == "__main__":
    main()
