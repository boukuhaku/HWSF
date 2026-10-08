"""Statistical calibration of the base prefecture yield (Section IV-B, Supplementary Section S3).

* fixed-filter prediction p^fil: ridge regression on the outputs of fixed random temporal filters applied to
  prefecture-mean weather (six filter banks, seeds 8701-8706);
* weather-and-forecast correction c^wf: ridge regression on six weather summaries and four forecast fields;
* SST-index correction c^SST: ridge regression on the 30 SST-index values (fitted to official prefecture yields with
  20 NASA POWER weather means that are used only during fitting).
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .constants import CLIP, MONTHS

FILTER_BANK_SEEDS = (8701, 8702, 8703, 8704, 8705, 8706)
FILTER_LENGTHS = (7, 9, 11)
FILTER_DILATIONS = (1, 2, 4, 8)
FILTERS_PER_GROUP = 64
FILTER_INPUT_CHANNELS = 16   # eight weather channels and eight zero placeholders


def make_filter_bank(seed):
    """768 fixed random temporal filters: 64 for each length (7, 9, 11) and dilation (1, 2, 4, 8). Half of them
    select one channel and the others 2, 4, 8, or 16 channels; Gaussian weights are demeaned and unit-normalized,
    and biases are uniform on [-1, 1]."""
    rng = np.random.default_rng(seed)
    bank = []
    for length in FILTER_LENGTHS:
        for dilation in FILTER_DILATIONS:
            weight = np.zeros((FILTERS_PER_GROUP, FILTER_INPUT_CHANNELS, length))
            for j in range(FILTERS_PER_GROUP):
                count = 1 if j < FILTERS_PER_GROUP // 2 else int(rng.choice([2, 4, 8, 16]))
                channels = rng.choice(FILTER_INPUT_CHANNELS, count, replace=False)
                values = rng.normal(size=(count, length))
                values -= values.mean()
                values /= np.linalg.norm(values)
                weight[j, channels] = values
            bias = rng.uniform(-1, 1, FILTERS_PER_GROUP)
            bank.append((length, dilation, weight, bias))
    return bank


def filter_features(x, bank):
    """Monthly maxima and positive proportions of the filter outputs plus monthly channel means (6208 values).

    x: (N, 121, 16) tensor of prefecture-mean standardized channels. Returns float64 values rounded to float32, as in
    the fitted regressions."""
    a = x.transpose(1, 2).double()
    blocks = []
    for length, dilation, weight, bias in bank:
        w = torch.as_tensor(weight, dtype=torch.float64, device=a.device)
        b = torch.as_tensor(bias, dtype=torch.float64, device=a.device)
        z = F.conv1d(a, w, b, padding=((length - 1) * dilation) // 2, dilation=dilation)
        for lo, hi in MONTHS:
            v = z[:, :, lo:hi]
            blocks.extend([v.max(2).values, (v > 0).double().mean(2)])
    blocks += [a[:, :, lo:hi].mean(2) for lo, hi in MONTHS]
    return torch.cat(blocks, 1).float().double()


class FixedFilterRegression(nn.Module):
    """Fixed-filter prediction p^fil for each prefecture-year.

    Weighted standardization (scale floor 0.05, clipping at +-6) of the 6208 filter features and the 16 prefecture-mean
    regional features, multiplied by sqrt(0.75/6208) and sqrt(0.25/16); a ridge regression (penalty 0.01) predicts the
    standardized mean-yield residual relative to the regional baseline. The six bank predictions are averaged."""

    def __init__(self, n_banks=6, dynamic=6208, static=16):
        super().__init__()
        self.register_buffer("coefficient", torch.zeros(n_banks, dynamic + static, dtype=torch.float64))
        self.register_buffer("intercept", torch.zeros(n_banks, dtype=torch.float64))
        self.register_buffer("design_mean", torch.zeros(n_banks, dynamic + static, dtype=torch.float64))
        self.register_buffer("dynamic_mean", torch.zeros(n_banks, dynamic, dtype=torch.float64))
        self.register_buffer("dynamic_scale", torch.ones(n_banks, dynamic, dtype=torch.float64))
        self.register_buffer("static_mean", torch.zeros(n_banks, static, dtype=torch.float64))
        self.register_buffer("static_scale", torch.ones(n_banks, static, dtype=torch.float64))
        self.register_buffer("yield_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("yield_scale", torch.ones((), dtype=torch.float64))
        self.dynamic, self.static = dynamic, static

    def forward(self, features, regional, baseline):
        """features: list of (B, P, 6208) bank features; regional: (P, 16) prefecture-mean regional features;
        baseline: (P,) prefecture-mean regional baseline. Returns (B, P) predictions in kg/10a."""
        predictions = []
        for k, f in enumerate(features):
            dynamic = ((f - self.dynamic_mean[k]) / self.dynamic_scale[k]).clamp(-CLIP, CLIP) * (0.75 / self.dynamic) ** 0.5
            static = ((regional - self.static_mean[k]) / self.static_scale[k]).clamp(-CLIP, CLIP) * (0.25 / self.static) ** 0.5
            design = torch.cat([dynamic, static.expand(len(f), -1, -1)], 2)
            h = self.intercept[k] + (design - self.design_mean[k]) @ self.coefficient[k]
            predictions.append((baseline + h) * self.yield_scale + self.yield_mean)
        return torch.stack(predictions).mean(0)


class WeatherForecastCorrection(nn.Module):
    """Weather-and-forecast correction c^wf = 0.5 sigma_r x beta: prefecture-centered predictors, scaled with floor
    0.05, clipped at +-6, and multiplied by sqrt(0.5/6) (weather summaries) or sqrt(0.5/4) (forecasts)."""

    def __init__(self, n_prefectures=4, n_predictors=10):
        super().__init__()
        self.register_buffer("center", torch.zeros(n_prefectures, n_predictors, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(n_predictors, dtype=torch.float64))
        self.register_buffer("mass", torch.ones(n_predictors, dtype=torch.float64))
        self.register_buffer("coefficient", torch.zeros(n_predictors, dtype=torch.float64))
        self.register_buffer("target_scale", torch.ones((), dtype=torch.float64))

    def forward(self, predictors):
        """predictors: (B, P, 10) six weather summaries and four forecast fields. Returns (B, P) in kg/10a."""
        z = ((predictors - self.center) / self.scale).clamp(-CLIP, CLIP) * self.mass
        return 0.5 * self.target_scale * (z @ self.coefficient)


class SSTIndexCorrection(nn.Module):
    """SST-index correction c^SST, shared by all prefectures in a year: only the 30 SST coefficients of the ridge fit
    are used at query time, times the yield scale. Standardization uses past-only means and scales (floor 0.05,
    clipping at +-6) and the multipliers sqrt(0.25/24) (Nino indices) and sqrt(0.125/6) (NINO.WEST)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(30, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(30, dtype=torch.float64))
        self.register_buffer("coefficient", torch.zeros(30, dtype=torch.float64))
        self.register_buffer("yield_scale", torch.ones((), dtype=torch.float64))

    def forward(self, indices):
        """indices: (B, 30). Returns (B,) in kg/10a."""
        z = ((indices - self.mean) / self.scale).clamp(-CLIP, CLIP)
        mass = torch.cat([torch.full((24,), (0.25 / 24) ** 0.5), torch.full((6,), (0.125 / 6) ** 0.5)]).to(z)
        return self.yield_scale * ((z * mass) @ self.coefficient)
