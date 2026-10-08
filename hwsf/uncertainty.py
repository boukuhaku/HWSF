"""Uncertainty evaluation (Section IV-F): Monte Carlo propagation of input uncertainty and prediction intervals.

Step 1. For each target year, perturb the target-year inputs and pass every draw through the complete inference chain
with all weights and calibrations fixed. The SD of the draws is the input standard uncertainty u_in of each record.
Four sources are perturbed independently:
  * spatial representativeness of the weather inputs: daily differences between the paddy-area-weighted mean of all
    NARO grid cells of a municipality and the value at its representative grid point, resampled in circular 7-day
    blocks within each month (same blocks for all municipalities and variables) and added to the daily inputs;
  * SST maps: the monthly mean of the daily OISST error SDs, with one standard normal number per 10 x 10 degree cell
    and month (fully correlated within a month and a cell);
  * SST indices: rectangular errors of half the reporting resolution (+-0.005 deg C for the four CPC indices,
    +-0.05 deg C for NINO.WEST);
  * yields of the three reference years: rectangular errors of +-0.5 kg/10a.
CFSv2 forecasts are held fixed; their errors enter through the residuals of step 2.

Step 2. Prediction interval of each record: y_hat +- z_{1-alpha/2} sigma_t kappa, with kappa = max(u_in, 1 kg/10a)
and sigma_t the root mean square of the scaled errors e / kappa of all records of the earlier evaluation years (eq. 12).
"""
from dataclasses import dataclass, field, replace
import math

import numpy as np
from scipy.stats import norm

from .constants import MONTHS

ROOT_SEED = 2201024
WEATHER, SST_MAPS, SST_INDICES, YIELDS, TEMPERATURE = 1, 2, 3, 4, 5
# Six raw daily variables used for perturbation: mean, max, min temperature, precipitation, radiation, humidity.
RAW_VARIABLES = ("mean_temperature", "max_temperature", "min_temperature", "precipitation", "solar_radiation",
                 "relative_humidity")
PLAUSIBLE_RANGES = ((-80.0, 60.0), (-80.0, 60.0), (-80.0, 60.0), (0.0, 2000.0), (0.0, 60.0), (0.0, 100.0))
INDEX_HALF_WIDTH = np.r_[np.full(24, 0.005), np.full(6, 0.05)]
YIELD_HALF_WIDTH = 0.5
# Validation RMSEs of the NARO grid for mean, maximum, and minimum temperature (Ohno et al., 2016).
TEMPERATURE_RMSE = (0.66, 0.98, 1.10)


def generator(year, trial, source, root_seed=ROOT_SEED):
    """Independent random stream for one target year, draw, and input source."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([root_seed, int(year), int(trial), int(source)])))


def raw_from_channels(channels):
    """Six raw daily variables (..., 6) from the eight model channels."""
    c = channels
    return np.stack([c[..., 4], c[..., 3], c[..., 5], np.expm1(c[..., 0]), c[..., 1], c[..., 2]], -1)


def channels_from_raw(raw):
    """Eight model channels from the six raw daily variables (inverse of ``raw_from_channels``)."""
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))  # noqa: E731
    vpd = (es(raw[..., 1]) + es(raw[..., 2])) / 2 * (1 - raw[..., 5] / 100)
    return np.stack([np.log1p(raw[..., 3]), raw[..., 4], raw[..., 5], raw[..., 1], raw[..., 0], raw[..., 2], vpd,
                     raw[..., 4] * vpd], -1)


@dataclass
class InputUncertainty:
    """Input-uncertainty sources of one target year.

    representativeness: (n, 121, 6) area-mean minus representative-point daily weather (``RAW_VARIABLES`` order).
    sst_error_sd:       (6, 50, 180) monthly mean of the daily OISST error SDs (deg C) on the 2-degree grid; 0 on land.
    index_half_width:   (30,) half reporting resolution of the SST indices.
    yield_half_width:   half reporting resolution of the reference-year yields (kg/10a).
    """
    representativeness: np.ndarray
    sst_error_sd: np.ndarray
    index_half_width: np.ndarray = field(default_factory=lambda: INDEX_HALF_WIDTH.copy())
    yield_half_width: float = YIELD_HALF_WIDTH

    def draw(self, inputs, trial, sources=(WEATHER, SST_MAPS, SST_INDICES, YIELDS), coherent=False, block=7,
             prefecture_ids=None):
        """One perturbed copy of ``inputs`` (a ``TargetYearInputs`` without a sample axis).

        ``coherent`` uses one random number per year for each source (full correlation within the source); ``block``
        is the weather block length in days. Source TEMPERATURE perturbs mean, maximum, and minimum temperature with
        the validation RMSEs as SDs, one normal number per prefecture and 7-day block (needs ``prefecture_ids``)."""
        year = inputs.year
        raw = raw_from_channels(np.asarray(inputs.weather, float))
        sst = np.zeros_like(self.sst_error_sd)
        index = np.zeros(30)
        history = np.zeros_like(np.asarray(inputs.reference_yields, float))
        if WEATHER in sources:
            rng = generator(year, trial, WEATHER)
            days = np.arange(raw.shape[1])
            for a, b in MONTHS:
                chosen = []
                while len(chosen) < b - a:
                    start = int(rng.integers(b - a))
                    chosen.extend(((start + np.arange(block)) % (b - a) + a).tolist())
                days[a:b] = chosen[:b - a]
            raw = raw + self.representativeness[:, days]
        if SST_MAPS in sources:
            rng = generator(year, trial, SST_MAPS)
            if coherent:
                noise = np.full(self.sst_error_sd.shape, rng.normal())
            else:
                noise = np.repeat(np.repeat(rng.normal(size=(6, 10, 36)), 5, axis=1), 5, axis=2)
            sst = self.sst_error_sd * noise
        if SST_INDICES in sources:
            rng = generator(year, trial, SST_INDICES)
            u = np.repeat(rng.uniform(-1, 1), 30) if coherent else rng.uniform(-1, 1, 30)
            index = self.index_half_width * u
        if YIELDS in sources:
            rng = generator(year, trial, YIELDS)
            shape = history.shape
            u = np.tile(rng.uniform(-1, 1, shape[-1]), (shape[0], 1)) if coherent else rng.uniform(-1, 1, shape)
            history = self.yield_half_width * u
        if TEMPERATURE in sources:
            rng = generator(year, trial, TEMPERATURE)
            n_blocks = math.ceil(raw.shape[1] / 7)
            noise = np.repeat(rng.normal(size=(int(np.max(prefecture_ids)) + 1, n_blocks)), 7, axis=1)[:, :raw.shape[1]]
            raw[:, :, :3] = raw[:, :, :3] + noise[prefecture_ids, :, None] * np.array(TEMPERATURE_RMSE)
        # Restore physical ranges and minimum <= mean <= maximum temperature, then recompute the channels.
        for v, (low, high) in enumerate(PLAUSIBLE_RANGES):
            raw[:, :, v] = np.clip(raw[:, :, v], low, high)
        raw[:, :, 1] = np.maximum(raw[:, :, 1], raw[:, :, 0])
        raw[:, :, 2] = np.minimum(raw[:, :, 2], raw[:, :, 0])
        fields = np.asarray(inputs.sst_fields, float) + sst
        return replace(inputs, weather=channels_from_raw(raw), sst_fields=fields,
                       sst_indices=np.asarray(inputs.sst_indices, float) + index,
                       reference_yields=np.asarray(inputs.reference_yields, float) + history)


def load_sources(path):
    """Read the input-uncertainty sources of one target year (fields of ``InputUncertainty``) from an .npz file."""
    with np.load(path, allow_pickle=False) as z:
        extra = {k: z[k] for k in ("index_half_width",) if k in z.files}
        if "yield_half_width" in z.files:
            extra["yield_half_width"] = float(z["yield_half_width"])
        return InputUncertainty(representativeness=z["representativeness"], sst_error_sd=z["sst_error_sd"], **extra)


def stack(draws):
    """Stack perturbed ``TargetYearInputs`` along a new sample axis (reference weather is shared)."""
    first = draws[0]
    return replace(first, weather=np.stack([d.weather for d in draws]), sst_fields=np.stack([d.sst_fields for d in draws]),
                   sst_indices=np.stack([d.sst_indices for d in draws]),
                   reference_yields=np.stack([d.reference_yields for d in draws]),
                   forecasts=np.stack([np.asarray(d.forecasts, float) for d in draws]))


def propagate(model, inputs, uncertainty, draws=1024, batch_size=64, **options):
    """Monte Carlo propagation (JCGM 101). Returns the draws (draws x n), the unperturbed estimate, and u_in."""
    base = model.predict(inputs)["estimate"]
    values = []
    for start in range(0, draws, batch_size):
        batch = stack([uncertainty.draw(inputs, trial, **options) for trial in range(start, min(draws, start + batch_size))])
        values.append(model.predict(batch)["estimate"])
    values = np.concatenate(values)
    return dict(draws=values, estimate=base, input_standard_uncertainty=values.std(0, ddof=1),
                mean_shift=values.mean(0) - base)


def stability(draws, fraction=0.95):
    """Whether the first half of the draws already gives stable means, SDs, and 2.5th/97.5th percentiles: for at
    least ``fraction`` of the records, |mean change| <= max(0.2, 0.1 SD), |SD change| <= 10 %, and each percentile
    change <= max(0.5, 0.2 SD), comparing the first half with all draws."""
    half, full = draws[: len(draws) // 2], draws
    sd = full.std(0, ddof=1)
    mean_ok = np.abs(half.mean(0) - full.mean(0)) <= np.maximum(0.2, 0.1 * sd)
    sd_ok = np.abs(half.std(0, ddof=1) - sd) <= 0.1 * sd
    q_half, q_full = np.quantile(half, [0.025, 0.975], axis=0), np.quantile(full, [0.025, 0.975], axis=0)
    q_ok = (np.abs(q_half - q_full) <= np.maximum(0.5, 0.2 * sd)).all(0)
    rates = dict(mean=mean_ok.mean(), sd=sd_ok.mean(), endpoints=q_ok.mean())
    return all(r >= fraction for r in rates.values()), rates


# ---------------------------------------------------------------------------------------------- prediction intervals
def interval_scale(u_input, floor=1.0):
    """kappa = max(u_in, 1 kg/10a)."""
    return np.maximum(np.asarray(u_input, float), floor)


def residual_scale(errors, kappa):
    """sigma_t: root mean square of the scaled errors e / kappa of the calibration records."""
    return float(np.sqrt(np.mean((np.asarray(errors) / np.asarray(kappa)) ** 2)))


def gaussian_interval(estimate, kappa, sigma, level):
    """Equation (12): estimate +- z_{1-alpha/2} sigma kappa."""
    half = norm.ppf((1 + level) / 2) * sigma * np.asarray(kappa)
    return estimate - half, estimate + half


def conformal_quantile(scores, level):
    """Split conformal half-width: the ceil((n + 1) level)-th smallest score (infinite if it does not exist)."""
    scores = np.sort(np.asarray(scores, float))
    k = math.ceil((len(scores) + 1) * level)
    return scores[k - 1] if k <= len(scores) else math.inf


def interval_score(lower, upper, y, level):
    """Interval score of Gneiting and Raftery (2007); lower is better."""
    a = 1 - level
    return (upper - lower) + 2 / a * np.maximum(lower - y, 0) + 2 / a * np.maximum(y - upper, 0)


def forward_intervals(table, level, method="gaussian", scaled=True):
    """Prediction intervals of each evaluation year calibrated with all earlier evaluation years.

    ``table`` is a DataFrame with columns year, y_true, estimate, and (if ``scaled``) u_input. ``method`` is
    "gaussian" (eq. 12), "conformal" (split conformal on |e| / kappa), or "year_block" (split conformal on the largest
    |e| / kappa of each calibration year). Returns a copy with lower, upper, hit, width, and interval_score; the first
    year has no earlier residuals and is omitted."""
    out = []
    years = sorted(table.year.unique())
    for year in years[1:]:
        cal, query = table[table.year < year], table[table.year == year].copy()
        kc = interval_scale(cal.u_input) if scaled else np.ones(len(cal))
        kq = interval_scale(query.u_input) if scaled else np.ones(len(query))
        e = (cal.y_true - cal.estimate).to_numpy() / kc
        if method == "gaussian":
            lower, upper = gaussian_interval(query.estimate.to_numpy(), kq, residual_scale(e, 1.0), level)
        else:
            scores = np.abs(e) if method == "conformal" else \
                cal.assign(s=np.abs(e)).groupby("year").s.max().to_numpy()
            half = conformal_quantile(scores, level) * kq
            lower, upper = query.estimate.to_numpy() - half, query.estimate.to_numpy() + half
        y = query.y_true.to_numpy()
        query["lower"], query["upper"] = lower, upper
        query["hit"] = ((y >= lower) & (y <= upper)).astype(int)
        query["width"] = upper - lower
        query["interval_score"] = interval_score(lower, upper, y, level)
        out.append(query)
    import pandas as pd
    return pd.concat(out, ignore_index=True)
