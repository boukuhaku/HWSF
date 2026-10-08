"""Training dataset: every source series that the annual refitting of HWSF reads (docs/training.md).

A ``Dataset`` covers the study municipalities for the years 2008 to the last target year. Values that are not yet
known at an origin (for example the yields of the target year) may be present in the file: the training code reads
only years before the origin, plus the inputs of the target year that are available before the cutoff.
"""
from dataclasses import dataclass

import numpy as np

from ..constants import PREFECTURES

FIRST_YEAR = 2008            # first year of weather and yield records used for training
FIRST_FORECAST_YEAR = 2012   # first year with archived CFSv2 July forecasts
FIRST_SST_YEAR = 1982        # first year of the SST maps


@dataclass
class Dataset:
    """Source series in model order (prefectures alphabetically, then area codes).

    prefecture, area_code: (n,) municipality identifiers; latitude, longitude: (n,) representative grid cells.
    years:            (Y,) consecutive years from 2008.
    weather:          (Y, n, 121, 8) daily channels (``hwsf.prepare.naro_channels``).
    yields:           (Y, n) municipal yields (kg/10a); NaN where not published.
    prefecture_yields:(Y, 4) official prefecture yields (kg/10a), for the SST-index correction.
    power:            (Y, 4, 20) NASA POWER April-July monthly means of T2M, T2M_MAX, T2M_MIN, log(1 + PRECTOTCORR),
                      ALLSKY_SFC_SW_DWN for each prefecture (index 5 * month + variable).
    forecasts:        (Y, 4, 4) prefecture CFSv2 forecast fields; NaN before 2012.
    sst_indices:      (Y, 30) SST-index values (``constants.SST_INDEX_NAMES``).
    sst_years:        (Z,) consecutive years from 1982.
    sst_fields:       (Z, 6, 50, 180) January-June OISST fields on the 2-degree grid, NaN on land.
    """
    prefecture: np.ndarray
    area_code: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    years: np.ndarray
    weather: np.ndarray
    yields: np.ndarray
    prefecture_yields: np.ndarray
    power: np.ndarray
    forecasts: np.ndarray
    sst_indices: np.ndarray
    sst_years: np.ndarray
    sst_fields: np.ndarray

    def __post_init__(self):
        self.prefecture = np.asarray(self.prefecture).astype(str)
        self.area_code = np.asarray(self.area_code).astype(str)
        order = sorted(range(len(self.area_code)), key=lambda i: (PREFECTURES.index(self.prefecture[i]), self.area_code[i]))
        if order != list(range(len(order))):
            raise ValueError("municipalities must be in model order (prefecture, then area code)")
        if self.years[0] != FIRST_YEAR or np.any(np.diff(self.years) != 1):
            raise ValueError("years must be consecutive from 2008")

    def index(self, year):
        return int(year - self.years[0])

    @property
    def prefecture_ids(self):
        return np.array([PREFECTURES.index(p) for p in self.prefecture])

    @property
    def counts(self):
        """Number of study municipalities of each prefecture."""
        return np.bincount(self.prefecture_ids, minlength=len(PREFECTURES)).astype(float)


def target_year_inputs(dataset, year):
    """``TargetYearInputs`` of a target year from a dataset (regional inputs use the climate of 2008..year-1)."""
    from ..data import TargetYearInputs
    from ..prepare import monthly_means
    d, k = dataset, dataset.index(year)
    climate = np.mean([monthly_means(d.weather[d.index(t)]) for t in range(int(d.years[0]), year)], 0)
    refs = [d.index(year - j) for j in (3, 2, 1)]
    inputs = TargetYearInputs(
        year=year, municipality_prefecture=list(d.prefecture), regional_inputs=np.column_stack([climate, d.latitude, d.longitude]),
        weather=d.weather[k], reference_weather=np.stack([d.weather[r] for r in refs], 1),
        reference_yields=np.stack([d.yields[r] for r in refs], 1), forecasts=d.forecasts[k],
        sst_fields=d.sst_fields[int(year - d.sst_years[0])], sst_indices=d.sst_indices[k])
    inputs.area_code = list(d.area_code)
    return inputs


FIELDS = ("prefecture", "area_code", "latitude", "longitude", "years", "weather", "yields", "prefecture_yields", "power",
          "forecasts", "sst_indices", "sst_years", "sst_fields")


def save_dataset(path, dataset):
    np.savez_compressed(path, **{k: np.asarray(getattr(dataset, k)) for k in FIELDS})


def load_dataset(path):
    with np.load(path, allow_pickle=False) as z:
        return Dataset(**{k: z[k] for k in FIELDS})
