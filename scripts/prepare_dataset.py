"""Build the training dataset (data/dataset.npz) from the downloaded source data (docs/data.md, docs/training.md).

python scripts/prepare_dataset.py --naro naro_daily.csv --yields municipal_yields.csv \
    --prefecture-yields prefecture_yields.csv --power power_daily.csv --forecasts forecasts.csv \
    --cpc sstoi.indices --jma nino_west_sst.txt --last-year 2025 --output data/dataset.npz

Inputs
  --naro               NARO daily values at the representative cells, long format with columns prefecture,
                       area_code, date, variable (APCPRA, GSR, RH, TMP_max, TMP_mea, TMP_min), value, latitude,
                       longitude
  --yields             municipal yields: prefecture, area_code, year, yield (kg/10a)
  --prefecture-yields  official prefecture yields: prefecture, year, yield
  --power              NASA POWER daily values: prefecture, date, T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, ALLSKY_SFC_SW_DWN
  --forecasts          CFSv2 prefecture fields: prefecture, year, temperature_lead1, temperature_lead2,
                       log_precipitation_rate_lead1, log_precipitation_rate_lead2 (see hwsf.prepare.forecast_fields)
  --cpc, --jma         NOAA CPC sstoi.indices and the JMA NINO.WEST SST file
  --oisst              optional .npz with {year: (6, 50, 180)} fields; downloaded from NOAA PSL when omitted
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hwsf import prepare as P  # noqa: E402
from hwsf.constants import FORECAST_FIELDS, PREFECTURES  # noqa: E402
from hwsf.training.dataset import FIRST_FORECAST_YEAR, FIRST_SST_YEAR, FIRST_YEAR, Dataset, save_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("naro", "yields", "prefecture-yields", "power", "forecasts", "cpc", "jma"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--oisst")
    ap.add_argument("--last-year", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    years = np.arange(FIRST_YEAR, args.last_year + 1)

    naro = pd.read_csv(args.naro, dtype={"area_code": str})
    naro["area_code"] = naro.area_code.str.zfill(5)
    raw = P.read_naro_daily(naro)
    site = naro.groupby("area_code")[["prefecture", "latitude", "longitude"]].first().reset_index()
    site["order"] = [PREFECTURES.index(p) for p in site.prefecture]
    site = site.sort_values(["order", "area_code"]).reset_index(drop=True)
    codes = site.area_code.tolist()
    weather = np.stack([np.stack([P.naro_channels(raw[(a, int(y))]) for a in codes]) for y in years])

    y = pd.read_csv(args.yields, dtype={"area_code": str})
    y["area_code"] = y.area_code.str.zfill(5)
    ym = y.set_index(["area_code", "year"])["yield"]
    yields = np.array([[ym.get((a, int(t)), np.nan) for a in codes] for t in years], float)
    py = pd.read_csv(args.prefecture_yields).set_index(["prefecture", "year"])["yield"]
    prefecture_yields = np.array([[py.get((p, int(t)), np.nan) for p in PREFECTURES] for t in years], float)
    pw = P.power_monthly(pd.read_csv(args.power))
    power = np.array([[pw.get((p, int(t)), np.full(20, np.nan)) for p in PREFECTURES] for t in years])
    fc = pd.read_csv(args.forecasts).set_index(["prefecture", "year"])
    forecasts = np.full((len(years), 4, 4), np.nan)
    for k, t in enumerate(years):
        if t >= FIRST_FORECAST_YEAR:
            forecasts[k] = fc.loc[[(p, int(t)) for p in PREFECTURES], list(FORECAST_FIELDS)].to_numpy(float)
    cpc, jma = Path(args.cpc).read_text(), Path(args.jma).read_text()
    indices = np.array([P.sst_indices(cpc, jma, int(t)) for t in years])
    if args.oisst:
        with np.load(args.oisst) as z:
            fields = {int(k): z[k] for k in z.files}
    else:
        fields = P.fetch_oisst_series(FIRST_SST_YEAR, args.last_year)
    sst_years = np.arange(FIRST_SST_YEAR, args.last_year + 1)
    data = Dataset(prefecture=site.prefecture.to_numpy(), area_code=np.array(codes), latitude=site.latitude.to_numpy(float),
                   longitude=site.longitude.to_numpy(float), years=years, weather=weather, yields=yields,
                   prefecture_yields=prefecture_yields, power=power, forecasts=forecasts, sst_indices=indices,
                   sst_years=sst_years, sst_fields=np.stack([fields[int(t)] for t in sst_years]).astype(np.float32))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_dataset(args.output, data)
    print(f"wrote {args.output}: {len(codes)} municipalities, {years[0]}-{years[-1]}")


if __name__ == "__main__":
    main()
