# Training

HWSF is refitted every year. For a target year (origin) *t*, every fitted quantity uses only records of the years
before *t*: standardization statistics, regressions, networks, and the numbers of updates. `scripts/train.py` runs the
origins in increasing order and saves one checkpoint per evaluated target year.

```bash
python scripts/train.py --dataset data/dataset.npz --target-years 2019 2020 2021 2022 2023 2024 2025 --output runs/hwsf
```

## Training dataset

`hwsf.training.Dataset` (`load_dataset` / `save_dataset`, one `.npz` file) holds the source series in model order:

| Field | Shape | Content |
|---|---|---|
| `prefecture`, `area_code` | (n,) | municipalities in model order |
| `latitude`, `longitude` | (n,) | representative grid cells |
| `years` | (Y,) | consecutive years from 2008 |
| `weather` | (Y, n, 121, 8) | daily channels (`hwsf.prepare.naro_channels`) |
| `yields` | (Y, n) | municipal yields (kg/10 a) |
| `prefecture_yields` | (Y, 4) | official prefecture yields (kg/10 a) |
| `power` | (Y, 4, 20) | NASA POWER April–July monthly means of T2M, T2M_MAX, T2M_MIN, log(1 + PRECTOTCORR), ALLSKY_SFC_SW_DWN (index 5·month + variable) |
| `forecasts` | (Y, 4, 4) | prefecture CFSv2 fields (NaN before 2012) |
| `sst_indices` | (Y, 30) | SST-index values |
| `sst_years` | (Z,) | consecutive years from 1982 |
| `sst_fields` | (Z, 6, 50, 180) | January–June OISST on the 2° sample grid, NaN on land |

`scripts/prepare_dataset.py` builds this file from the downloaded source data (see [data.md](data.md)).

## Stages of one origin (Algorithm 1)

1. **Base networks and historical features.** For each of the six weather members:
   - *Regional network* (float64, full batch, AdamW, learning rate 0.003). The number of updates (400 or 1200) is
     selected for each group of three members with an inner fit on the years up to *t* − 3, scored on *t* − 2 and
     *t* − 1 with the mean of the three members.
   - *Annual-weather* and *local-weather networks*: 600 updates with batches of 64, loss
     mean[(g + h − y)²] + 0.1 mean(h²) in standardized yield units, with the regional network frozen.
   - *FiLM regional conditioning* of the local-weather network: 200 further updates (learning rates 1e-4 and 1e-3)
     with a penalty 0.1 mean[(h − h_initial)²].
   - *Statistical calibration*: fixed-filter prediction (six banks, weighted ridge, penalty 0.01),
     weather-and-forecast correction (targets: observed prefecture means minus the annual-weather predictions made at
     earlier origins, penalty 0.1), and the SST-index correction (official prefecture yields, penalty 0.01);
     base prefecture yield by equation (6).
   - *History weighting network* on the responses from 2011 (three reference years each) and *history correction
     network* on the estimates of the origins 2012…*t* − 1 (200 updates each); base municipal deviation d⁰.
2. **Correction networks.** Five initializations each of the prefecture correction network (eq. 9; prefecture-years
   from 2012) and the municipal correction network (eq. 10; municipal-years from 2014), 200 updates each. Their targets
   use the base predictions made at the earlier origins.
3. **SST-map pathway.** Ten SST encoders pretrained by masked reconstruction on the SST maps of 1982…*t* − 1
   (300 updates; each cell hidden with probability 0.5, every mask paired with its complement), then frozen. For each
   member, ten SST fusion networks (eq. 11, 200 updates) while both correction outputs stay fixed.

All networks except the regional network use AdamW with weight decay 0.01, learning rate 0.001 (unless stated), and
gradient-norm clipping at 5; the output layers of the correction networks start at zero. Seeds: weather members
7301–7303 and 102301–102303 (regional network: member seed − 300), correction and fusion initialization *k* of a member:
member seed + 1000003·*k*, SST encoder *k*: 1930001 + 1009·*k*.

## Origin records

The history correction network and the targets of the correction networks use predictions that earlier origins made
for their own year. After fitting origin *t*, `fit_origins` saves `origin_t.npz` (`OriginRecord`) with

| Field | Used by later origins for |
|---|---|
| prefecture means of the annual-weather predictions, fixed-filter predictions, weather-and-forecast and SST-index corrections, base prefecture yields (per member) | weather-and-forecast targets, prediction-state features, prefecture targets |
| centered local-weather representations of the year and of its three reference years | history correction inputs |
| history-weighted municipal deviations | history correction targets |
| base municipal deviations d⁰ | municipal targets |

Origins 2012…2018 need only stage 1. The evaluation in the paper fits the origins 2012…2025 and evaluates 2019–2025.

## Compute

One evaluated origin trains 166 networks (45,000 updates, plus six inner fits of the regional network for selecting
its number of updates) and fits 39 ridge regressions. On a laptop GPU (RTX 4050) an evaluated origin takes three to six
minutes and an origin without corrections about two minutes; all origins 2012–2025 take about 40 minutes. A CPU also
works but is slower.

## Reproducibility

The released checkpoints reproduce the published estimates. Retraining with this code follows the same recipe,
seeds, and update schedules; GPU kernels and the order of floating-point reductions can still change the fitted
weights slightly, so retrained estimates differ from the released ones by small amounts (see
[reproduce.md](reproduce.md)).
