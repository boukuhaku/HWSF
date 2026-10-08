# Reproducing the paper

## 1. Tables from the published estimates (no data needed)

`results/estimates_2019_2025.csv` contains, for the 560 municipality-year records of 2019–2025, the observed MAFF
yield, the HWSF estimate, and the input standard uncertainty (1024 joint Monte Carlo draws).

```bash
python scripts/evaluate.py       # RMSE 11.495, MAE 8.769, bias -0.290, R^2 0.877; by year and prefecture; MSE decomposition
python scripts/intervals.py      # 95%: 453/480 = 94.4% coverage, mean width 39.7, interval score 58.6 (and 90%, 68%)
```

`scripts/intervals.py` also reports the comparison intervals of the paper (constant scale, split conformal,
input-scaled conformal, and year-block conformal).

## 2. Estimates from the released checkpoints

With the training dataset of [training.md](training.md) (built from the source data with `scripts/prepare_dataset.py`):

```bash
python scripts/download_checkpoints.py --output checkpoints
for y in 2019 2020 2021 2022 2023 2024 2025; do
  python scripts/predict.py --checkpoint checkpoints/hwsf_$y.pt --dataset data/dataset.npz --year $y --output estimates_$y.csv
done
```

The checkpoints reproduce the published estimates of all 560 records to within 1e-5 kg/10 a.

## 3. Input standard uncertainty

```bash
python scripts/propagate.py --checkpoint checkpoints/hwsf_2025.pt --dataset data/dataset.npz --year 2025 \
    --sources data/uncertainty_2025.npz --draws 1024 --device cuda --output u_in_2025.csv
```

The random streams are fixed per target year, draw, and source, so the draws are reproducible. With the released
checkpoints and the same sources, the input standard uncertainties agree with the published values to within 1e-6
kg/10 a. Other scenarios of Table IV: `--scenario coherent` (full correlation within each source), `block14`,
`weather`, `sst-maps`, `sst-indices`, `yields`, and `temperature`.

## 4. Retraining

```bash
python scripts/train.py --dataset data/dataset.npz --target-years 2019 2020 2021 2022 2023 2024 2025 --output runs/hwsf
```

Retraining follows the recipe of the paper with the same seeds and update schedules. GPU kernels and the order of
floating-point reductions can differ between machines and implementations, so the retrained weights differ slightly
from the released ones; see the retraining check below.

### Retraining check

* **One origin, earlier-origin records from the released runs.** Refitting the target year 2025 with this code
  (378 s on an RTX 4050 Laptop GPU) gives estimates that differ from the released ones by at most 0.026 kg/10 a
  (mean absolute difference 0.008 kg/10 a); the RMSE of the 80 records is 13.235 kg/10 a (released: 13.241).
* **All origins from scratch.** Running `scripts/train.py` for the origins 2012–2025 (42 minutes on the same GPU)
  selects the same numbers of regional-network updates as the released runs in every origin and both member groups.
  Over the 560 records of 2019–2025, the retrained estimates differ from the released ones by at most 0.16 kg/10 a
  (mean absolute difference 0.04 kg/10 a); their RMSE is 11.500 kg/10 a (released: 11.495).
