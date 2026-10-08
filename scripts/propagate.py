"""Monte Carlo propagation of input uncertainty for one target year (Section IV-F).

python scripts/propagate.py --checkpoint checkpoints/hwsf_2025.pt --dataset data/dataset.npz --year 2025 \
    --sources data/uncertainty_2025.npz --draws 1024 --output u_in_2025.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwsf import HWSF, load_inputs  # noqa: E402
from hwsf import uncertainty as U  # noqa: E402

SCENARIOS = {
    "joint": dict(),                                   # four independent sources (primary case)
    "weather": dict(sources=(U.WEATHER,)),
    "sst-maps": dict(sources=(U.SST_MAPS,)),
    "sst-indices": dict(sources=(U.SST_INDICES,)),
    "yields": dict(sources=(U.YIELDS,)),
    "coherent": dict(coherent=True),                   # full correlation within each source
    "block14": dict(block=14),                         # 14-day weather blocks
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--inputs", help="TargetYearInputs (.npz)")
    ap.add_argument("--dataset", help="training dataset (.npz); used with --year instead of --inputs")
    ap.add_argument("--year", type=int)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--draws", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--scenario", choices=list(SCENARIOS) + ["temperature"], default="joint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    model = HWSF.from_checkpoint(args.checkpoint, device=args.device)
    if args.inputs:
        inputs = load_inputs(args.inputs)
    else:
        from hwsf.training.dataset import load_dataset, target_year_inputs
        inputs = target_year_inputs(load_dataset(args.dataset), args.year or int(model.year))
    sources = U.load_sources(args.sources)
    options = SCENARIOS.get(args.scenario) or dict(sources=(U.TEMPERATURE,), prefecture_ids=inputs.prefecture_ids())
    res = U.propagate(model, inputs, sources, draws=args.draws, batch_size=args.batch_size, **options)
    stable, rates = U.stability(res["draws"])
    pd.DataFrame(dict(prefecture=inputs.municipality_prefecture, area_code=inputs.area_code, year=inputs.year,
                      estimate=res["estimate"], input_standard_uncertainty=res["input_standard_uncertainty"],
                      mean_shift=res["mean_shift"])).to_csv(args.output, index=False, float_format="%.6f")
    print(json.dumps(dict(draws=args.draws, scenario=args.scenario, stable=stable, **rates), indent=1))


if __name__ == "__main__":
    main()
