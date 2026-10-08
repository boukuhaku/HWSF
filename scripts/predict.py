"""Estimate the municipal yields of one target year with a fitted HWSF checkpoint.

python scripts/predict.py --checkpoint checkpoints/hwsf_2025.pt --dataset data/dataset.npz --year 2025 --output estimates_2025.csv
python scripts/predict.py --checkpoint checkpoints/hwsf_2025.pt --inputs data/inputs_2025.npz --output estimates_2025.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwsf import HWSF, load_inputs  # noqa: E402
from hwsf.constants import PREFECTURES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--inputs", help="TargetYearInputs (.npz), see docs/data.md")
    ap.add_argument("--dataset", help="training dataset (.npz); used with --year instead of --inputs")
    ap.add_argument("--year", type=int)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    model = HWSF.from_checkpoint(args.checkpoint, device=args.device)
    if args.inputs:
        inputs = load_inputs(args.inputs)
    else:
        from hwsf.training.dataset import load_dataset, target_year_inputs
        inputs = target_year_inputs(load_dataset(args.dataset), args.year or int(model.year))
    if [tuple(m) for m in model.municipalities] != list(zip(inputs.municipality_prefecture, inputs.area_code)):
        raise SystemExit("The municipalities of the inputs do not match the checkpoint.")
    out = model.predict(inputs)
    frame = pd.DataFrame(dict(prefecture=inputs.municipality_prefecture, area_code=inputs.area_code, year=inputs.year,
                              estimate=out["estimate"], municipal_deviation=out["municipal_deviation"]))
    frame["prefecture_level"] = frame.prefecture.map(dict(zip(PREFECTURES, out["prefecture_level"])))
    frame.to_csv(args.output, index=False, float_format="%.4f")
    print(f"wrote {len(frame)} estimates to {args.output}")


if __name__ == "__main__":
    main()
