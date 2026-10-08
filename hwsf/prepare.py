"""Preparation of the model inputs from the source data (Section III; docs/data.md).

The functions here turn downloaded source files into the arrays of ``TargetYearInputs``:

* NARO daily weather at the representative grid point of each municipality -> eight daily channels;
* regional-network inputs: April-July monthly climate means of the six daily variables over the training years, plus
  latitude and longitude of the representative point;
* SST indices: absolute monthly SST of NOAA CPC Nino 1+2, 3, 4, 3.4 and JMA NINO.WEST, January-June;
* OISST v2.1 monthly SST fields sampled every 2 degrees (50 x 180 points) for January-June;
* CFSv2 forecast fields interpolated bilinearly to the representative points and averaged per prefecture.
"""
import io
import re
import struct
import urllib.request

import numpy as np
import pandas as pd

from .constants import MONTHS, N_DAYS, PREFECTURES
from .data import TargetYearInputs, season_dates

NARO_VARIABLES = ("APCPRA", "GSR", "RH", "TMP_max", "TMP_mea", "TMP_min")


# ---------------------------------------------------------------------------------------------- daily weather
def read_naro_daily(frame):
    """Daily NARO values in long format (columns area_code, date, variable, value) -> dict
    {(area_code, year): (121, 6) array in ``NARO_VARIABLES`` order}, April 1 - July 30.

    Days outside April 1 - July 30 are dropped before the values are used. Where the daily mean temperature exceeds
    the daily maximum (one record in the study data), the maximum is raised to the mean."""
    f = frame.copy()
    f["area_code"] = f["area_code"].astype(str).str.zfill(5)
    f["date"] = f["date"].astype(str).str[:10]
    md = f["date"].str[5:]
    f = f[(md >= "04-01") & (md <= "07-30")]
    wide = f.pivot_table(index=["area_code", "date"], columns="variable", values="value", aggfunc="first")
    wide = wide[list(NARO_VARIABLES)].reset_index()
    wide["TMP_max"] = np.maximum(wide["TMP_max"], wide["TMP_mea"])
    wide["year"] = wide["date"].str[:4].astype(int)
    out = {}
    for (area, year), g in wide.groupby(["area_code", "year"], sort=True):
        g = g.sort_values("date")
        if len(g) != N_DAYS:
            raise ValueError(f"{area} {year}: {len(g)} days instead of {N_DAYS}")
        out[(area, int(year))] = g[list(NARO_VARIABLES)].to_numpy(float)
    return out


def naro_channels(raw):
    """Eight model channels (..., 121, 8) from daily NARO values (..., 121, 6) in ``NARO_VARIABLES`` order."""
    p, gsr, rh, tmax, tmean, tmin = (raw[..., k] for k in range(6))
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))  # noqa: E731
    vpd = (es(tmax) + es(tmin)) / 2 * (1 - rh / 100)
    return np.stack([np.log1p(p), gsr, rh, tmax, tmean, tmin, vpd, gsr * vpd], -1)


def monthly_means(channels):
    """April-July means of the first six channels: (..., 24), index 6 * month + channel."""
    return np.concatenate([channels[..., a:b, :6].mean(-2) for a, b in MONTHS], -1)


def regional_inputs(channels_by_year, latitude, longitude, years):
    """Regional-network inputs (n, 26): climate means over ``years`` (the training years), latitude, longitude.

    channels_by_year: {year: (n, 121, 8)} daily channels of the municipalities in model order."""
    climate = np.mean([monthly_means(channels_by_year[y]) for y in years], 0)
    return np.column_stack([climate, latitude, longitude])


# ---------------------------------------------------------------------------------------------- SST indices
def read_cpc_sstoi(text, year):
    """Absolute January-June SST (deg C) of Nino 1+2, 3, 4, 3.4 from the CPC file ``sstoi.indices``: 24 values in
    the order month-major (nino12, nino3, nino4, nino34)."""
    lines = text.strip().splitlines()
    if lines[0].split() != ["YR", "MON", "NINO1+2", "ANOM", "NINO3", "ANOM", "NINO4", "ANOM", "NINO3.4", "ANOM"]:
        raise ValueError("unexpected CPC header")
    values = {}
    for line in lines[1:]:
        tok = line.split()
        if int(tok[0]) == year and 1 <= int(tok[1]) <= 6:
            values[int(tok[1])] = [float(tok[k]) for k in (2, 4, 6, 8)]
    return np.array([v for m in range(1, 7) for v in values[m]])


def read_jma_ninowest(text, year):
    """Absolute January-June SST (deg C, 0.1 deg C resolution) of NINO.WEST from the JMA file (rows: year v1..v12)."""
    for line in text.strip().splitlines():
        tok = line.split()
        if tok and tok[0] == str(year):
            values = np.array([float(v) for v in tok[1:7]])
            if np.any(values > 99):
                raise ValueError("missing NINO.WEST value")
            return values
    raise ValueError(f"year {year} not found")


def sst_indices(cpc_text, jma_text, year):
    """The 30 SST-index values of ``constants.SST_INDEX_NAMES``."""
    return np.r_[read_cpc_sstoi(cpc_text, year), read_jma_ninowest(jma_text, year)]


# ---------------------------------------------------------------------------------------------- OISST maps
OISST_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
LATITUDE_SAMPLES = -89.875 + 0.25 * np.arange(204, 597, 8)    # 50 points, -38.875 ... 59.125
LONGITUDE_SAMPLES = 0.125 + 0.25 * np.arange(4, 1437, 8)      # 180 points, 1.125 ... 359.125


def decode_dods(payload):
    """Decode an OPeNDAP .dods response of the OISST variable ``sst[time][lat][lon]`` (XDR, big-endian)."""
    head, data = payload.split(b"\nData:\n", 1)
    t, ny, nx = map(int, re.search(rb"Float32 sst\[time = (\d+)\]\[lat = (\d+)\]\[lon = (\d+)\]", head).groups())
    out, offset = [], 0
    for count, fmt in ((t * ny * nx, ">f4"), (t, ">f8"), (ny, ">f4"), (nx, ">f4")):
        n, n2 = struct.unpack(">II", data[offset:offset + 8])
        assert n == n2 == count
        offset += 8
        size = np.dtype(fmt).itemsize * n
        out.append(np.frombuffer(data[offset:offset + size], fmt).astype(float))
        offset += size
    sst = out[0].reshape(t, ny, nx)
    return np.where(sst > -100, sst, np.nan), out[1], out[2], out[3]


def fetch_oisst_fields(year, url=OISST_URL):
    """January-June monthly OISST fields of ``year`` on the 2-degree sample grid: (6, 50, 180), NaN on land."""
    fields = []
    for month in range(1, 7):
        index = (year - 1981) * 12 + month - 9          # time index 0 is September 1981
        query = f"{url}.dods?sst[{index}:1:{index}][204:8:596][4:8:1436]"
        with urllib.request.urlopen(query, timeout=120) as r:
            sst, _, lat, lon = decode_dods(r.read())
        np.testing.assert_allclose(lat, LATITUDE_SAMPLES)
        np.testing.assert_allclose(lon, LONGITUDE_SAMPLES)
        fields.append(sst[0])
    return np.stack(fields)


def fetch_oisst_series(first_year, last_year, url=OISST_URL):
    """January-June OISST fields of the years first_year..last_year: {year: (6, 50, 180)}, six requests in total."""
    out = {}
    for month in range(1, 7):
        lo = (first_year - 1981) * 12 + month - 9
        hi = lo + (last_year - first_year) * 12
        with urllib.request.urlopen(f"{url}.dods?sst[{lo}:12:{hi}][204:8:596][4:8:1436]", timeout=300) as r:
            sst, _, lat, lon = decode_dods(r.read())
        np.testing.assert_allclose(lat, LATITUDE_SAMPLES)
        np.testing.assert_allclose(lon, LONGITUDE_SAMPLES)
        for k, year in enumerate(range(first_year, last_year + 1)):
            out.setdefault(year, [None] * 6)[month - 1] = sst[k]
    return {y: np.stack(v) for y, v in out.items()}


# ---------------------------------------------------------------------------------------------- NASA POWER
POWER_VARIABLES = ("T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN")


def power_monthly(frame):
    """April-July monthly means (July 1-30) of daily NASA POWER values for each prefecture and year.

    frame: columns prefecture, date, and ``POWER_VARIABLES``; precipitation enters as log(1 + mm/day) per day.
    Returns {(prefecture, year): (20,) array, index 5 * month + variable}."""
    f = frame.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f[(f.date.dt.month >= 4) & (f.date.dt.month <= 7) & ~((f.date.dt.month == 7) & (f.date.dt.day == 31))]
    f["PRECTOTCORR"] = np.log1p(f["PRECTOTCORR"])
    f["year"], f["month"] = f.date.dt.year, f.date.dt.month
    means = f.groupby(["prefecture", "year", "month"])[list(POWER_VARIABLES)].mean()
    out = {}
    for (p, y), g in means.groupby(level=[0, 1]):
        out[(p, int(y))] = g.loc[(p, y)].loc[[4, 5, 6, 7]].to_numpy(float).ravel()
    return out


def sst_climatology(fields_by_year, years=range(1982, 2012), floor=0.2):
    """Monthly climatology and SD (floor 0.2 deg C) of the 10 x 10 degree area means, and the ocean fraction of each
    cell, from {year: (6, 50, 180)} fields. Returns (climatology, scale, ocean_fraction)."""
    lat_w = np.cos(np.deg2rad(LATITUDE_SAMPLES))[:, None]
    coarse = []
    for y in years:
        f = fields_by_year[y]
        valid = np.isfinite(f)
        w = valid * lat_w
        block = lambda a: a.reshape(a.shape[:-2] + (10, 5, 36, 5)).sum((-3, -1))  # noqa: E731
        coarse.append(block(np.where(valid, f, 0) * w) / np.maximum(block(w), 1e-12))
    coarse = np.array(coarse)
    mask = np.isfinite(fields_by_year[years[0]][0])
    ocean = mask.reshape(10, 5, 36, 5).sum((1, 3)) / 25
    return coarse.mean(0), np.maximum(coarse.std(0), floor), ocean


# ---------------------------------------------------------------------------------------------- forecasts
def bilinear(grid, lat0, lon0, latitude, longitude):
    """Bilinear interpolation of values on a 1-degree grid (grid[i, j] at lat0 + i, lon0 + j) to points."""
    s, w = np.floor(latitude).astype(int), np.floor(longitude).astype(int)
    dy, dx = latitude - s, longitude - w
    i, j = s - lat0, w - lon0
    return ((1 - dy) * (1 - dx) * grid[i, j] + (1 - dy) * dx * grid[i, j + 1] + dy * (1 - dx) * grid[i + 1, j]
            + dy * dx * grid[i + 1, j + 1])


def forecast_fields(temperature_k, precipitation_rate, lat0, lon0, latitude, longitude, prefecture):
    """Prefecture forecast fields (P, 4) from CFSv2 ensemble-mean grids of the July run.

    temperature_k, precipitation_rate: (2, ny, nx) grids at nominal leads +1 and +2 (K; kg m-2 s-1).
    The values are interpolated to the representative points, converted (deg C; log(1 + mm/day)), and averaged over
    the municipalities of each prefecture."""
    t = [bilinear(temperature_k[k], lat0, lon0, latitude, longitude) - 273.15 for k in range(2)]
    p = [np.log1p(bilinear(precipitation_rate[k], lat0, lon0, latitude, longitude) * 86400) for k in range(2)]
    values = np.column_stack(t + p)
    prefecture = np.asarray(prefecture)
    return np.stack([values[prefecture == name].mean(0) for name in PREFECTURES])


# ---------------------------------------------------------------------------------------------- assembly
def target_year_inputs(year, municipalities, channels_by_year, yields, regional, forecasts, sst_fields, sst_index_values):
    """Assemble ``TargetYearInputs``.

    municipalities: DataFrame with prefecture and area_code in model order; channels_by_year: {year: (n, 121, 8)};
    yields: DataFrame with area_code, year, yield (kg/10a); regional: (n, 26) from ``regional_inputs``."""
    codes = municipalities.area_code.astype(str).tolist()
    table = yields.assign(area_code=yields.area_code.astype(str)).set_index(["area_code", "year"])["yield"]
    ref_years = (year - 3, year - 2, year - 1)
    return TargetYearInputs(
        year=year, municipality_prefecture=municipalities.prefecture.tolist(), regional_inputs=regional,
        weather=channels_by_year[year], reference_weather=np.stack([channels_by_year[y] for y in ref_years], 1),
        reference_yields=np.array([[table[(a, y)] for y in ref_years] for a in codes], float),
        forecasts=np.asarray(forecasts, float), sst_fields=np.asarray(sst_fields, float),
        sst_indices=np.asarray(sst_index_values, float))


def season_index():
    """Month-day labels of the 121 input days (April 1 - July 30)."""
    return pd.Index(season_dates(), name="month_day")


__all__ = [name for name in dir() if not name.startswith("_") and name not in ("io", "re", "struct", "urllib", "np", "pd")]
