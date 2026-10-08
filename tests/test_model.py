"""Unit tests on synthetic inputs (no data or checkpoints needed)."""
import numpy as np
import pytest
import torch

from hwsf import HWSF, TargetYearInputs
from hwsf.calibration import make_filter_bank
from hwsf.data import weather_channels
from hwsf.evaluation import metrics, mse_decomposition
from hwsf.uncertainty import conformal_quantile, gaussian_interval, interval_scale, residual_scale

PREFS = ["fukui"] * 3 + ["ishikawa"] * 2 + ["niigata"] * 4 + ["toyama"] * 2


def synthetic_inputs(n=len(PREFS), seed=0, samples=None):
    rng = np.random.default_rng(seed)

    def weather(shape):
        p, r, h = rng.gamma(0.5, 6, shape), rng.uniform(5, 25, shape), rng.uniform(50, 95, shape)
        tmean = rng.normal(20, 3, shape)
        return weather_channels(p, r, h, tmean + rng.uniform(2, 6, shape), tmean, tmean - rng.uniform(2, 6, shape))

    lead = () if samples is None else (samples,)
    lat = np.linspace(-80, 18, 50)
    fields = rng.normal(20, 5, lead + (6, 50, 180))
    fields[..., :5, :20] = np.nan                                 # some land cells
    return TargetYearInputs(
        year=2025, municipality_prefecture=PREFS, regional_inputs=rng.normal(size=(n, 26)),
        weather=weather(lead + (n, 121)), reference_weather=np.moveaxis(weather((3, n, 121)), 0, 1),
        reference_yields=rng.normal(530, 30, lead + (n, 3)), forecasts=rng.normal(size=lead + (4, 4)),
        sst_fields=fields, sst_indices=rng.normal(27, 1, lead + (30,))), lat


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = HWSF(PREFS)
    _, lat = synthetic_inputs()
    m.sst_latitude.copy_(torch.as_tensor(lat))
    m.weather_scale.fill_(5.0)
    m.yield_mean.fill_(530.0)
    m.yield_scale.fill_(30.0)
    for member in m.members:   # non-zero correction outputs, so that the centering is actually tested
        for net in list(member.municipal_corrections) + list(member.prefecture_corrections) + list(member.sst_fusion):
            torch.nn.init.normal_(net.output.weight, std=0.1)
    return m.double().eval()


def test_parameter_count(model):
    assert sum(p.numel() for p in model.parameters()) == 1_777_262


def test_hierarchy_and_centering(model):
    inputs, _ = synthetic_inputs()
    out = model.predict(inputs, return_components=True)
    ids = inputs.prefecture_ids()
    assert np.all(np.isfinite(out["estimate"]))
    for j in range(4):
        assert abs(out["municipal_deviation"][ids == j].mean()) < 1e-9          # centered deviations (eq. 5)
    np.testing.assert_allclose(out["estimate"], out["prefecture_level"][ids] + out["municipal_deviation"], atol=1e-9)
    mu0 = out["base_prefecture_yield"]
    expected = (0.875 * out["annual_weather_prediction"] + 0.125 * out["fixed_filter_prediction"]
                + 0.25 * out["weather_forecast_correction"] + 0.75 * out["sst_index_correction"])     # eq. (6)
    np.testing.assert_allclose(mu0, expected, atol=1e-9)


def test_sample_axis_matches_single_prediction(model):
    batch, _ = synthetic_inputs(samples=3, seed=1)
    out = model.predict(batch)["estimate"]
    for k in range(3):
        single, _ = synthetic_inputs(samples=3, seed=1)
        for name in ("weather", "reference_yields", "forecasts", "sst_fields", "sst_indices"):
            setattr(single, name, getattr(batch, name)[k])
        np.testing.assert_allclose(model.predict(single)["estimate"], out[k], atol=1e-9)


def test_filter_bank_is_deterministic():
    a, b = make_filter_bank(8701), make_filter_bank(8701)
    assert len(a) == 12 and all(np.array_equal(x[2], y[2]) for x, y in zip(a, b))
    weights = np.concatenate([w[2].ravel() for w in a])
    assert abs(weights.sum()) < 1e-8                            # every filter is demeaned


def test_intervals():
    kappa = interval_scale([0.3, 2.0])
    np.testing.assert_array_equal(kappa, [1.0, 2.0])
    assert residual_scale([3.0, -4.0], 1.0) == pytest.approx(np.sqrt(12.5))
    lo, hi = gaussian_interval(np.array([100.0]), np.array([1.0]), 10.0, 0.95)
    assert hi[0] - lo[0] == pytest.approx(2 * 1.959964 * 10.0, rel=1e-6)
    assert conformal_quantile(np.arange(1, 20), 0.95) == 19
    assert conformal_quantile(np.arange(1, 6), 0.95) == np.inf


def test_metrics_and_decomposition():
    import pandas as pd
    frame = pd.DataFrame(dict(prefecture=["a", "a", "b", "b"], year=[1, 1, 1, 1], y_true=[10.0, 12.0, 8.0, 9.0],
                              estimate=[11.0, 12.0, 10.0, 7.0]))
    m = metrics(frame.y_true, frame.estimate)
    d = mse_decomposition(frame)
    assert d["prefecture_year"] + d["municipal"] == pytest.approx(m["rmse"] ** 2)
