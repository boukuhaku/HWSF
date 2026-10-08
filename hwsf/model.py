"""HWSF model for one target year: base predictions, hierarchical correction, and the SST-map pathway (Section IV).

``HWSF`` holds the six weather members, the ten SST encoders, and the fixed preprocessing statistics of one target
year. ``HWSF.predict`` maps ``TargetYearInputs`` to municipal estimates and their components:

    y_hat[i] = mu_hat[c] + d_hat[i],   mean_i d_hat[i] = 0 within each prefecture-year            (eq. 5)
    mu0      = 0.875 p_ann + 0.125 p_fil + 0.25 c_wf + 0.75 c_SST                                (eq. 6)
    mu_hat   = mu0 + u + v,   d_hat = d0 + q,   q = a - prefecture mean of a                     (eqs. 7-8)
"""
import numpy as np
import torch
from torch import nn

from .calibration import (FILTER_BANK_SEEDS, FixedFilterRegression, SSTIndexCorrection, WeatherForecastCorrection,
                          filter_features, make_filter_bank)
from .constants import (CLIP, LOCAL_WEATHER_CHANNELS, N_CORRECTION_INITIALIZATIONS, N_DAYS, N_SST_REPLICAS,
                        N_WEATHER_MEMBERS, PREFECTURES, STATE_SCALE, WEIGHT_ANNUAL, WEIGHT_FIXED_FILTER,
                        WEIGHT_SST_INDEX, WEIGHT_WEATHER_FORECAST)
from .data import calendar_encoding, summary_windows
from .nn.networks import (HistoryNetwork, RegionalNetwork, ResidualMLP, SSTEncoder, WeatherNetwork)


def _clip(x):
    return x.clamp(-CLIP, CLIP)


class Member(nn.Module):
    """One weather member: base networks, statistical calibration, history networks, and its correction networks
    (five prefecture and five municipal initializations, ten SST fusion networks)."""

    def __init__(self, n_regional_inputs=26, n_prefectures=len(PREFECTURES)):
        super().__init__()
        self.regional = RegionalNetwork(n_regional_inputs).double()   # fitted in float64
        self.register_buffer("regional_mean", torch.zeros(n_regional_inputs, dtype=torch.float64))
        self.register_buffer("regional_scale", torch.ones(n_regional_inputs, dtype=torch.float64))
        self.annual = WeatherNetwork(in_channels=10, depthwise=True)
        self.local = WeatherNetwork(in_channels=8, depthwise=False, film=True)
        self.fixed_filter = FixedFilterRegression()
        self.weather_forecast = WeatherForecastCorrection(n_prefectures)
        self.history_weighting = HistoryNetwork()
        self.history_correction = HistoryNetwork()
        for name in ("history_weighting", "history_correction"):
            self.register_buffer(name + "_mean", torch.zeros(34, dtype=torch.float64))
            self.register_buffer(name + "_scale", torch.ones(34, dtype=torch.float64))
        # Yield standard deviation of the local-weather model, used to scale the reference-year residuals.
        self.register_buffer("residual_scale", torch.ones((), dtype=torch.float64))
        # Historical mean of the base prefecture yield of each prefecture (prediction-state feature 1).
        self.register_buffer("level_center", torch.zeros(n_prefectures, dtype=torch.float64))
        # Standardization of the 80 current features (Table II rows 1-3).
        self.register_buffer("feature_mean", torch.zeros(80, dtype=torch.float64))
        self.register_buffer("feature_scale", torch.ones(80, dtype=torch.float64))
        self.prefecture_corrections = nn.ModuleList(ResidualMLP(294) for _ in range(N_CORRECTION_INITIALIZATIONS))
        self.municipal_corrections = nn.ModuleList(ResidualMLP(147) for _ in range(N_CORRECTION_INITIALIZATIONS))
        self.register_buffer("hidden_mean", torch.zeros(N_CORRECTION_INITIALIZATIONS, 64, dtype=torch.float64))
        self.register_buffer("hidden_scale", torch.ones(N_CORRECTION_INITIALIZATIONS, 64, dtype=torch.float64))
        self.sst_fusion = nn.ModuleList(ResidualMLP(64 + 16) for _ in range(N_SST_REPLICAS))


class HWSF(nn.Module):
    """HWSF fitted for one target year and one set of municipalities."""

    def __init__(self, municipality_prefecture, n_regional_inputs=26):
        super().__init__()
        ids = torch.tensor([PREFECTURES.index(p) for p in municipality_prefecture])
        n, n_pref = len(ids), len(PREFECTURES)
        self.register_buffer("year", torch.zeros((), dtype=torch.long))
        self.register_buffer("prefecture_ids", ids)
        # Daily standardization: municipality-by-calendar-day centers and pooled channel scales from earlier years.
        self.register_buffer("weather_center", torch.zeros(n, N_DAYS, 8, dtype=torch.float64))
        self.register_buffer("weather_scale", torch.ones(8, dtype=torch.float64))
        # Yield standardization of the training records (kg/10a).
        self.register_buffer("yield_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("yield_scale", torch.ones((), dtype=torch.float64))
        # Standardization of the weather summaries and forecasts, and of the SST indices, for the municipal features.
        self.register_buffer("environment_center", torch.zeros(n_pref, 10, dtype=torch.float64))
        self.register_buffer("environment_scale", torch.ones(10, dtype=torch.float64))
        self.register_buffer("index_mean", torch.zeros(30, dtype=torch.float64))
        self.register_buffer("index_scale", torch.ones(30, dtype=torch.float64))
        # SST maps: sample grid, 1982-2011 monthly climatology and SD (floor 0.2 deg C), ocean fraction of each cell.
        self.register_buffer("sst_latitude", torch.zeros(50, dtype=torch.float64))
        self.register_buffer("sst_longitude", torch.zeros(180, dtype=torch.float64))
        self.register_buffer("sst_climatology", torch.zeros(6, 10, 36, dtype=torch.float64))
        self.register_buffer("sst_scale", torch.ones(6, 10, 36, dtype=torch.float64))
        self.register_buffer("ocean_fraction", torch.zeros(10, 36, dtype=torch.float64))
        self.register_buffer("sst_feature_mean", torch.zeros(N_SST_REPLICAS, 16, dtype=torch.float64))
        self.register_buffer("sst_feature_scale", torch.ones(N_SST_REPLICAS, 16, dtype=torch.float64))
        self.sst_index = SSTIndexCorrection()
        self.sst_encoders = nn.ModuleList(SSTEncoder() for _ in range(N_SST_REPLICAS))
        self.members = nn.ModuleList(Member(n_regional_inputs, n_pref) for _ in range(N_WEATHER_MEMBERS))
        self.municipality_prefecture = list(municipality_prefecture)
        self._banks = None

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path, municipalities):
        """Save the fitted model; ``municipalities`` lists (prefecture, area code) in model order."""
        torch.save(dict(format="hwsf-checkpoint-1", year=int(self.year), municipalities=[list(m) for m in municipalities],
                        state_dict=self.state_dict()), path)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Load a checkpoint written by ``save_checkpoint``. Inference runs in float64 (weights are stored in float32
        except the float64 regional network), which reproduces the published estimates."""
        ck = torch.load(path, map_location=device, weights_only=True)
        model = cls([p for p, _ in ck["municipalities"]])
        model.load_state_dict(ck["state_dict"])
        model.municipalities = [tuple(m) for m in ck["municipalities"]]
        return model.double().to(device).eval()

    # ------------------------------------------------------------------ helpers
    def _group_matrix(self):
        ids = self.prefecture_ids
        m = torch.zeros(len(PREFECTURES), len(ids), dtype=torch.float64, device=ids.device)
        for j in range(len(PREFECTURES)):
            m[j, ids == j] = 1.0 / float((ids == j).sum())
        return m

    def _mean(self, x, axis):
        """Equal-municipality prefecture mean along ``axis`` (the municipality axis)."""
        return torch.movedim(torch.tensordot(self._group_matrix(), torch.movedim(x, axis, 0), dims=1), 0, axis)

    def _center(self, x, axis):
        """Subtract the prefecture mean along the municipality axis (within each prefecture-year)."""
        m = self._mean(x, axis)
        return x - torch.index_select(m, axis, self.prefecture_ids)

    def banks(self):
        if self._banks is None:
            self._banks = [make_filter_bank(seed) for seed in FILTER_BANK_SEEDS]
        return self._banks

    def sst_maps(self, fields):
        """Encoder input (S, 8, 10, 36) from January-June OISST fields (S, 6, 50, 180) on the 2-degree grid."""
        valid = torch.isfinite(fields)
        weights = valid * torch.cos(torch.deg2rad(self.sst_latitude))[:, None]
        values = torch.where(valid, fields, torch.zeros_like(fields))
        s = len(fields)
        den = weights.reshape(s, 6, 10, 5, 36, 5).sum((3, 5))
        coarse = (values * weights).reshape(s, 6, 10, 5, 36, 5).sum((3, 5)) / den.clamp_min(1e-12)
        anomaly = _clip((coarse - self.sst_climatology) / self.sst_scale) * (den > 0)
        extra = torch.stack([self.ocean_fraction.expand(s, -1, -1), torch.zeros_like(self.ocean_fraction).expand(s, -1, -1)], 1)
        return torch.cat([anomaly, extra], 1).float().double(), coarse

    def sst_features(self, image):
        """Standardized SST-map features of the ten frozen encoders: (S, 10, 16)."""
        codes = torch.stack([enc(image) for enc in self.sst_encoders], 1).float().double()
        return _clip((codes - self.sst_feature_mean) / self.sst_feature_scale).float().double()

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, inputs, return_components=False):
        """Estimates for one target year. ``inputs`` is a ``TargetYearInputs``; arrays may have a sample axis S."""
        dev, f64 = self.weather_center.device, torch.float64
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=f64, device=dev)  # noqa: E731
        weather = t(inputs.weather)
        single = weather.ndim == 3
        if single:
            weather = weather[None]
        s = len(weather)
        sample = lambda a, nd: (t(a)[None] if t(a).ndim == nd else t(a)).expand((s,) + t(a).shape[-nd:])  # noqa: E731
        forecasts = sample(inputs.forecasts, 2)
        fields = sample(inputs.sst_fields, 3)
        indices = sample(inputs.sst_indices, 1)
        ref_yields = sample(inputs.reference_yields, 2)
        ids, n = self.prefecture_ids, weather.shape[1]
        calendar = t(calendar_encoding()).expand(n, -1, -1)

        # Standardized daily inputs (float32 as in training) and prefecture-mean summaries.
        x = torch.cat([(weather - self.weather_center) / self.weather_scale, calendar.expand(s, -1, -1, -1)], 3).float()
        xd = x.double()
        local_x = xd[..., list(LOCAL_WEATHER_CHANNELS) + [8, 9]]
        summaries = torch.stack([self._mean(weather[:, :, torch.as_tensor(mask, device=dev), ch].mean(2), 1)
                                 for mask, ch in summary_windows()], 2)
        environment = torch.cat([summaries, forecasts], 2)                                   # S, P, 10
        prefecture_x = torch.stack([x[:, ids == j].mean(1) for j in range(len(PREFECTURES))], 1)[..., :8]
        prefecture_x = torch.cat([prefecture_x, torch.zeros_like(prefecture_x)], 3)          # S, P, 121, 16
        bank_features = [filter_features(prefecture_x.reshape(-1, N_DAYS, 16), bank).reshape(s, len(PREFECTURES), -1)
                         for bank in self.banks()]

        # Reference years t-3, t-2, t-1: standardized with the statistics of year t.
        ref = t(inputs.reference_weather)                                                     # n, 3, 121, 8
        rx = torch.cat([(ref - self.weather_center[:, None]) / self.weather_scale,
                        calendar[:, None].expand(-1, 3, -1, -1)], 3).float().double()
        ref_local_x = rx[..., list(LOCAL_WEATHER_CHANNELS) + [8, 9]].reshape(n * 3, N_DAYS, 8)

        image, _ = self.sst_maps(fields)
        sst = self.sst_features(image)                                                        # S, 10, 16
        c_sst = self.sst_index(indices)                                                       # S
        env_std = _clip((environment - self.environment_center) / self.environment_scale)
        index_std = _clip((indices - self.index_mean) / self.index_scale)
        age = torch.tensor([-1.0, 0.0, 1.0], dtype=f64, device=dev)

        parts = []
        for member in self.members:
            reg_in = (t(inputs.regional_inputs) - member.regional_mean) / member.regional_scale
            reg, g = member.regional(reg_in)
            reg = reg.float().double()                                                        # n, 16
            ls, lm = self.yield_scale, self.yield_mean
            # Annual-weather network -> prefecture mean of the annual-weather predictions p_ann.
            eh, eq = member.annual(xd.reshape(s * n, N_DAYS, 10), reg.repeat(s, 1))
            eh, eq = eh.reshape(s, n, 16), eq.reshape(s, n)
            p_ann = self._mean((g + eq) * ls + lm, 1)
            # Fixed-filter prediction, weather-and-forecast correction, and base prefecture yield (eq. 6).
            reg_p = torch.stack([reg.float()[ids == j].mean(0) for j in range(len(PREFECTURES))]).double()
            p_fil = member.fixed_filter(bank_features, reg_p, self._mean(g, 0))
            c_wf = member.weather_forecast(environment)
            mu0 = WEIGHT_ANNUAL * p_ann + WEIGHT_FIXED_FILTER * p_fil + WEIGHT_WEATHER_FORECAST * c_wf + \
                WEIGHT_SST_INDEX * c_sst[:, None]
            # Local-weather network and the learned history correction -> base municipal deviation d0.
            lh, lq = member.local(local_x.reshape(s * n, N_DAYS, 8), reg.repeat(s, 1))
            lh, lq = lh.reshape(s, n, 16), lq.reshape(s, n)
            local_pred = (g + lq) * ls + lm
            rh, rq = member.local(ref_local_x, reg.repeat_interleave(3, 0))
            rh, rq = rh.reshape(n, 3, 16), rq.reshape(n, 3)
            ref_pred = (g[:, None] + rq) * ls + lm
            scale = member.residual_scale
            residual = (ref_yields - ref_pred) / scale                                        # S, n, 3
            ref_rep = self._center(rh, 0)                                                     # centered per reference year
            current = self._center(lh, 1)
            base = self._center(local_pred + residual.mean(2) * scale, 1)
            vec = torch.cat([current[:, :, None].expand(-1, -1, 3, -1), ref_rep.expand(s, -1, -1, -1),
                             residual[..., None], age.expand(s, n, 3)[..., None]], 3)          # S, n, 3, 34
            score = member.history_weighting((vec - member.history_weighting_mean) / member.history_weighting_scale)
            weight = torch.softmax(torch.tanh(score), 2)
            weighted = base + scale * self._center((weight * residual).sum(2) - residual.mean(2), 1)
            hist_x = ((vec - member.history_correction_mean) / member.history_correction_scale).float().double()
            d0 = weighted + scale * self._center(member.history_correction(hist_x).mean(2), 1)
            parts.append(dict(reg=reg, annual_rep=eh, p_ann=p_ann, p_fil=p_fil, c_wf=c_wf, mu0=mu0, d0=d0, hist_x=hist_x))

        all_mu0 = torch.stack([p["mu0"] for p in parts], 1)                                  # S, M, P
        all_ann = torch.stack([p["p_ann"] for p in parts], 1)
        half = N_WEATHER_MEMBERS // 2
        estimates, levels, deviations = [], [], []
        for member, p in zip(self.members, parts):
            # Eight prediction-state features (Supplementary Section S5), divided by 20 and clipped.
            state = torch.stack([p["mu0"] - member.level_center, WEIGHT_FIXED_FILTER * (p["p_fil"] - p["p_ann"]),
                                 WEIGHT_WEATHER_FORECAST * p["c_wf"], (WEIGHT_SST_INDEX * c_sst)[:, None].expand(-1, len(PREFECTURES)),
                                 p["mu0"] - all_mu0.mean(1), all_mu0.std(1, correction=0),
                                 all_mu0[:, half:].mean(1) - all_mu0[:, :half].mean(1), all_ann.std(1, correction=0)], 2)
            state = _clip(state / STATE_SCALE)
            current80 = torch.cat([p["annual_rep"], p["reg"].expand(s, -1, -1), env_std[:, ids], index_std[:, None].expand(-1, n, -1),
                                   state[:, ids]], 2)
            x80 = ((current80 - member.feature_mean) / member.feature_scale).float().double()
            hx = p["hist_x"]
            x147 = torch.cat([x80, hx[:, :, 0, :16], hx[:, :, :, 16:33].reshape(s, n, 51)], 2)   # municipal features
            mean = self._mean(x147, 1)
            sd = self._mean((x147 - mean[:, ids]) ** 2, 1).clamp_min(0).sqrt()
            x294 = torch.cat([mean, sd], 2)                                                    # prefecture features
            q = torch.stack([self._center(net(x147)[1], 1) for net in member.municipal_corrections]).mean(0)
            hidden, u = [], []
            for k, net in enumerate(member.prefecture_corrections):
                h, out = net(x294)
                hidden.append(_clip((h - member.hidden_mean[k]) / member.hidden_scale[k]))
                u.append(out)
            hidden = torch.stack(hidden).mean(0)
            v = torch.stack([net(torch.cat([hidden, sst[:, k, None].expand(-1, len(PREFECTURES), -1)], 2))[1]
                             for k, net in enumerate(member.sst_fusion)]).mean(0)
            mu_hat = p["mu0"] + torch.stack(u).mean(0) + v
            d_hat = p["d0"] + q
            levels.append(mu_hat)
            deviations.append(d_hat)
            estimates.append(mu_hat[:, ids] + d_hat)

        out = dict(estimate=torch.stack(estimates).mean(0), prefecture_level=torch.stack(levels).mean(0),
                   municipal_deviation=torch.stack(deviations).mean(0))
        if return_components:
            out.update(member_estimates=torch.stack(estimates, 1), base_prefecture_yield=all_mu0,
                       annual_weather_prediction=all_ann, fixed_filter_prediction=torch.stack([p["p_fil"] for p in parts], 1),
                       weather_forecast_correction=torch.stack([p["c_wf"] for p in parts], 1), sst_index_correction=c_sst,
                       base_municipal_deviation=torch.stack([p["d0"] for p in parts], 1), sst_features=sst)
        if single:
            out = {k: v[0] for k, v in out.items()}
        return {k: v.cpu().numpy() for k, v in out.items()}
