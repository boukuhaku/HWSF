"""Input preparation: daily weather channels, calendar encodings, weather summaries, and SST maps (Section III)."""
from dataclasses import dataclass
import datetime as dt

import numpy as np

from .constants import N_DAYS, PREFECTURES, WEATHER_SUMMARIES


def vapor_pressure_deficit(max_temperature, min_temperature, relative_humidity):
    """VPD (kPa) from daily maximum and minimum temperature (deg C) and relative humidity (%)."""
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))  # noqa: E731
    return (es(max_temperature) + es(min_temperature)) / 2 * (1 - relative_humidity / 100)


def weather_channels(precipitation, solar_radiation, relative_humidity, max_temperature, mean_temperature,
                     min_temperature):
    """Eight daily channels (..., 121, 8) from the six daily NARO variables (..., 121) of one or more sites."""
    vpd = vapor_pressure_deficit(max_temperature, min_temperature, relative_humidity)
    return np.stack([np.log1p(precipitation), solar_radiation, relative_humidity, max_temperature, mean_temperature,
                     min_temperature, vpd, solar_radiation * vpd], -1)


def calendar_encoding(n_days=N_DAYS):
    """Sine and cosine of the day position within the April 1 - July 30 window: (121, 2)."""
    phase = 2 * np.pi * np.arange(n_days) / n_days
    return np.c_[np.sin(phase), np.cos(phase)]


def season_dates(n_days=N_DAYS):
    """Month-day strings of the 121 days from April 1 (a non-leap reference year)."""
    start = dt.date(2001, 4, 1)
    return np.array([(start + dt.timedelta(i)).strftime("%m-%d") for i in range(n_days)])


def summary_windows():
    """Boolean day masks and channel indices of the six weather summaries."""
    md = season_dates()
    return [((md >= a) & (md <= b), channel) for a, b, channel in WEATHER_SUMMARIES]


@dataclass
class TargetYearInputs:
    """Everything HWSF reads for one target year t, available before July 31, 00:00 JST.

    Arrays may carry a leading sample axis (used by the Monte Carlo propagation); without it, one sample is assumed.

    municipality_prefecture: prefecture name of each of the n municipalities (order of the fitted model).
    regional_inputs:   (n, 26)  24 April-July monthly climate means of the six daily variables over the training
                       years (index 6 * month + variable), latitude, and longitude.
    weather:           ([S,] n, 121, 8) daily channels of year t (see ``weather_channels``).
    reference_weather: (n, 3, 121, 8) daily channels of the reference years t-3, t-2, t-1.
    reference_yields:  ([S,] n, 3) observed municipal yields (kg/10a) of t-3, t-2, t-1.
    forecasts:         ([S,] P, 4) CFSv2 forecast fields of each prefecture (``constants.FORECAST_FIELDS``).
    sst_fields:        ([S,] 6, 50, 180) January-June monthly OISST (deg C) on the 2-degree sample grid; NaN on land.
    sst_indices:       ([S,] 30) SST-index values in the order of ``constants.SST_INDEX_NAMES``.
    """
    year: int
    municipality_prefecture: list
    regional_inputs: np.ndarray
    weather: np.ndarray
    reference_weather: np.ndarray
    reference_yields: np.ndarray
    forecasts: np.ndarray
    sst_fields: np.ndarray
    sst_indices: np.ndarray

    def prefecture_ids(self):
        return np.array([PREFECTURES.index(p) for p in self.municipality_prefecture])


INPUT_FIELDS = ("regional_inputs", "weather", "reference_weather", "reference_yields", "forecasts", "sst_fields",
                "sst_indices")


def load_inputs(path):
    """Read ``TargetYearInputs`` from an .npz file with the fields of ``save_inputs`` (see docs/data.md)."""
    with np.load(path, allow_pickle=False) as z:
        inputs = TargetYearInputs(year=int(z["year"]), municipality_prefecture=[str(p) for p in z["municipality_prefecture"]],
                                  **{k: z[k] for k in INPUT_FIELDS})
        inputs.area_code = [str(a) for a in z["area_code"]] if "area_code" in z.files else None
    return inputs


def save_inputs(path, inputs, area_code):
    """Write ``TargetYearInputs`` and the municipality codes to an .npz file."""
    np.savez_compressed(path, year=inputs.year, municipality_prefecture=np.array(inputs.municipality_prefecture),
                        area_code=np.array(area_code), **{k: np.asarray(getattr(inputs, k)) for k in INPUT_FIELDS})
