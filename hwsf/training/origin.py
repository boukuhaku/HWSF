"""Annual refitting of HWSF for one origin (target year) t, following Algorithm 1 of the paper.

``fit_origin`` trains every component with records of the years before t and returns
  * the fitted ``HWSF`` model of year t (if ``corrections=True``), and
  * the ``OriginRecord`` of year t: the predictions made at origin t for year t that later origins read
    (base prefecture-yield components for the weather-and-forecast targets, prediction-state features, and the
    prefecture targets; local representations and history-weighted deviations for the history correction network;
    base municipal deviations for the municipal targets).

Origins must be fitted in increasing order, starting in 2012 (``fit_origins``); the corrections are needed only for
the origins whose estimates are evaluated.
"""
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .. import constants as C
from ..calibration import (FILTER_BANK_SEEDS, SSTIndexCorrection, WeatherForecastCorrection, filter_features,
                           make_filter_bank)
from ..data import calendar_encoding, summary_windows
from ..model import HWSF
from ..prepare import LATITUDE_SAMPLES, LONGITUDE_SAMPLES, monthly_means
from . import fit as F
from .dataset import FIRST_FORECAST_YEAR, FIRST_SST_YEAR

MEMBER_SEEDS = (7301, 7302, 7303, 102301, 102302, 102303)   # weather members; regional seed = member seed - 300
FAMILIES = (MEMBER_SEEDS[:3], MEMBER_SEEDS[3:])
REGIONAL_STEPS = (400, 1200)
FIRST_RESPONSE_YEAR = 2011      # history weighting: responses with three reference years from 2008
FIRST_CORRECTION_YEAR = 2012    # prefecture correction and SST fusion
FIRST_MUNICIPAL_YEAR = 2014     # municipal correction loss
SST_FEATURE_FIRST_YEAR = 2012
CLIMATOLOGY_YEARS = range(1982, 2012)
REPLICA_SEED_STEP = 1000003


@dataclass
class OriginRecord:
    """Predictions made at origin t for year t (members in ``MEMBER_SEEDS`` order, prefectures in model order)."""
    year: int
    annual: np.ndarray                    # (6, 4) prefecture mean of the annual-weather predictions p_ann
    fixed_filter: np.ndarray              # (6, 4) p_fil
    weather_forecast: np.ndarray          # (6, 4) c_wf
    sst_index: float                      # c_SST
    base_level: np.ndarray                # (6, 4) mu0
    local_representation: np.ndarray     # (6, n, 16) centered local-weather representation of year t
    reference_representation: np.ndarray  # (6, n, 3, 16) centered representations of t-3, t-2, t-1
    weighted_deviation: np.ndarray        # (6, n) history-weighted municipal deviation
    base_deviation: np.ndarray            # (6, n) base municipal deviation d0

    def save(self, path):
        np.savez_compressed(path, **{f.name: np.asarray(getattr(self, f.name)) for f in fields(self)})

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            values = {f.name: z[f.name] for f in fields(cls)}
        values["year"], values["sst_index"] = int(values["year"]), float(values["sst_index"])
        return cls(**values)


# ---------------------------------------------------------------------------------------------- fold preparation
class Fold:
    """Training rows (years 2008..b-1, municipality-major) and statistics of a fold with 'before' year b."""

    def __init__(self, data, before, query_years=None):
        self.data, self.before = data, before
        n = len(data.area_code)
        self.train_years = np.arange(data.years[0], before)
        self.query_years = np.array([before] if query_years is None else query_years)
        rows = lambda years: [(i, int(y)) for i in range(n) for y in years]  # noqa: E731
        self.train_rows, self.query_rows = rows(self.train_years), rows(self.query_years)
        ix = lambda rs: (np.array([data.index(y) for _, y in rs]), np.array([i for i, _ in rs]))  # noqa: E731
        self.train_ix, self.query_ix = ix(self.train_rows), ix(self.query_rows)
        y = data.yields[self.train_ix]
        if np.isnan(y).any():
            raise ValueError("missing training yields")
        self.y_true = y
        self.yield_mean, self.yield_scale = float(y.mean()), max(float(y.std()), 1.0)
        self.y_std = (y - self.yield_mean) / self.yield_scale
        # Regional inputs: climate means over the training years, latitude, longitude; standardized over rows.
        climate = np.mean([monthly_means(data.weather[data.index(t)]) for t in self.train_years], 0)
        static = np.column_stack([climate, data.latitude, data.longitude])
        rows_static = static[self.train_ix[1]]
        self.static_mean = rows_static.mean(0)
        self.static_scale = np.maximum(rows_static.std(0), 1e-6)
        self.static_raw = static
        self.static = (static - self.static_mean) / self.static_scale
        # Daily standardization: municipality-by-day means over the training years, pooled channel SDs.
        w = data.weather[[data.index(t) for t in self.train_years]]                      # Y, n, 121, 8
        self.center = w.mean(0)
        self.scale = np.maximum((w - self.center).reshape(-1, 8).std(0), 1e-6)

    def inputs(self, ix):
        w = self.data.weather[ix] if isinstance(ix, tuple) else ix
        centers = self.center[ix[1]] if isinstance(ix, tuple) else self.center
        cal = calendar_encoding()
        x = np.concatenate([(w - centers) / self.scale, np.broadcast_to(cal, w.shape[:-1] + (2,))], -1)
        return x.astype(np.float32)


def local_channels(x):
    return x[..., list(C.LOCAL_WEATHER_CHANNELS) + [8, 9]]


def groups(rows, prefecture_ids):
    """Prefecture-year group of each row (prefecture-major, year-minor) and the sorted group keys."""
    keys = sorted({(int(prefecture_ids[i]), y) for i, y in rows})
    pos = {k: g for g, k in enumerate(keys)}
    return np.array([pos[(int(prefecture_ids[i]), y)] for i, y in rows]), keys


def select_regional_steps(data, origin, seeds):
    """Inner fold: fit on 2008..t-3, score the mean of the family's three seeds on t-2 and t-1 at 400 and 1200
    updates, and keep the better number of updates (ties -> 400)."""
    inner = Fold(data, origin - 2, query_years=[origin - 2, origin - 1])
    truth = data.yields[inner.query_ix]
    query = inner.static[inner.query_ix[1]]
    preds = {k: [] for k in REGIONAL_STEPS}
    for seed in seeds:
        _, rec = F.fit_regional(inner.static[inner.train_ix[1]], inner.y_std, seed - 300, max(REGIONAL_STEPS),
                                checkpoints=REGIONAL_STEPS, query=query)
        for k in REGIONAL_STEPS:
            preds[k].append(rec[k] * inner.yield_scale + inner.yield_mean)
    scores = {k: float(np.mean((np.mean(preds[k], 0) - truth) ** 2)) for k in REGIONAL_STEPS}
    return min(REGIONAL_STEPS, key=lambda k: (scores[k], k))


def environment(data, years):
    """Six weather summaries (prefecture means of municipal window means) and four forecasts: (len(years), 4, 10)."""
    ids = data.prefecture_ids
    out = []
    for y in years:
        w = data.weather[data.index(y)]
        s = np.stack([[w[ids == j][:, mask, ch].mean() for mask, ch in summary_windows()] for j in range(4)])
        out.append(np.concatenate([s, data.forecasts[data.index(y)]], 1))
    return np.array(out)


def sst_maps(data):
    """Clean encoder images (Z, 8, 10, 36) of all SST years, the 10 x 10 degree area weights, and the climatology."""
    f = data.sst_fields
    valid = np.isfinite(f)
    lat_w = np.cos(np.deg2rad(LATITUDE_SAMPLES))[:, None]
    block = lambda a: a.reshape(a.shape[:-2] + (10, 5, 36, 5)).sum((-3, -1))  # noqa: E731
    w = valid * lat_w
    den = block(w)
    coarse = block(np.where(valid, f, 0) * w) / np.maximum(den, 1e-12)
    clim_ix = [int(y - FIRST_SST_YEAR) for y in CLIMATOLOGY_YEARS]
    clim = coarse[clim_ix].mean(0)
    scale = np.maximum(coarse[clim_ix].std(0), 0.2)
    anomaly = np.clip((coarse - clim) / scale, -C.CLIP, C.CLIP) * (den > 0)
    ocean = valid[0, 0].reshape(10, 5, 36, 5).sum((1, 3)) / 25
    maps = np.concatenate([anomaly, np.broadcast_to(ocean, (len(f), 1, 10, 36)), np.zeros((len(f), 1, 10, 36))], 1)
    return maps.astype(np.float32), den[0, 0], clim, scale, ocean


# ---------------------------------------------------------------------------------------------- one origin
def fit_origin(data, origin, records, corrections=True, device=None, log=print):
    """Fit all components for target year ``origin``. ``records`` maps earlier origins (2012..t-1) to their
    ``OriginRecord``. Returns (model or None, record)."""
    device = device or F.default_device()
    F.configure()
    t = origin
    pid = data.prefecture_ids
    n, P = len(pid), len(C.PREFECTURES)
    counts = data.counts
    fold = Fold(data, t)
    LM, LS = fold.yield_mean, fold.yield_scale
    x_train, x_query = fold.inputs(fold.train_ix), fold.inputs(fold.query_ix)
    tgroups, tkeys = groups(fold.train_rows, pid)
    tmat = F.group_matrix(tgroups)
    qmat = F.group_matrix(pid)
    gw = counts[[p for p, _ in tkeys]]
    gw = gw / gw.sum()
    train_muni = np.array([i for i, _ in fold.train_rows])

    # Fixed-filter features: prefecture-mean standardized channels (float32 mean) plus eight zero channels.
    def prefecture_x(x, gid, ng):
        a = np.stack([x[gid == g].mean(0) for g in range(ng)])[..., :8]
        return np.concatenate([a, np.zeros_like(a)], -1)
    tq = torch.as_tensor(prefecture_x(x_train, tgroups, len(tkeys)), dtype=torch.float64)
    qq = torch.as_tensor(prefecture_x(x_query, pid, P), dtype=torch.float64)
    banks = [make_filter_bank(s) for s in FILTER_BANK_SEEDS]
    bank_train = [filter_features(tq, b).numpy() for b in banks]
    bank_query = [filter_features(qq, b).numpy() for b in banks]

    # Weather summaries and forecasts (2012..t), SST-index correction, SST indices.
    env_years = np.arange(FIRST_FORECAST_YEAR, t + 1)
    env = environment(data, env_years)                                        # (years, 4, 10)
    model = HWSF(list(data.prefecture)) if corrections else None
    c_mod = model.sst_index if corrections else SSTIndexCorrection()
    tr = [data.index(y) for y in fold.train_years]
    F.fit_sst_index(c_mod, data.power[tr], data.sst_indices[tr], data.prefecture_yields[tr], counts)
    c_sst = float(c_mod(torch.as_tensor(data.sst_indices[data.index(t)][None], dtype=torch.float64))[0])

    members = []
    steps = {}
    for fam in FAMILIES:
        k = select_regional_steps(data, t, fam)
        steps.update({s: k for s in fam})
    log(f"origin {t}: regional updates {steps}")
    for m, seed in enumerate(MEMBER_SEEDS):
        mem = {}
        # Stage 1: regional, annual-weather, local-weather networks with FiLM.
        reg_net, _ = F.fit_regional(fold.static[fold.train_ix[1]], fold.y_std, seed - 300, steps[seed])
        with torch.no_grad():
            h, g = reg_net(torch.as_tensor(fold.static))
        reg = h.numpy().astype(np.float32)                                     # (n, 16)
        g = g.numpy()
        reg_tr, g_tr = reg[train_muni], g[train_muni]
        annual = F.fit_weather(x_train, reg_tr, g_tr, fold.y_std, seed, True, device)
        local = F.fit_weather(local_channels(x_train), reg_tr, g_tr, fold.y_std, seed, False, device)
        local = F.fit_film(local, local_channels(x_train), reg_tr, g_tr, fold.y_std, seed, device)
        a_rep_tr, a_q_tr = F.weather_outputs(annual, x_train, reg_tr, device)
        a_rep_q, a_q_q = F.weather_outputs(annual, x_query, reg, device)
        l_rep_tr, l_q_tr = F.weather_outputs(local, local_channels(x_train), reg_tr, device)
        l_rep_q, l_q_q = F.weather_outputs(local, local_channels(x_query), reg, device)
        annual_pred_q = (g + a_q_q) * LS + LM
        local_pred_tr, local_pred_q = (g_tr + l_q_tr) * LS + LM, (g + l_q_q) * LS + LM
        p_ann = qmat @ annual_pred_q
        # Fixed-filter prediction.
        ff = F.new_fixed_filter_module()
        reg_p_tr = np.stack([reg_tr[tgroups == j].mean(0) for j in range(len(tkeys))]).astype(np.float32)
        reg_p_q = np.stack([reg[pid == j].mean(0) for j in range(P)]).astype(np.float32)
        g_p_tr, g_p_q = tmat @ g_tr, qmat @ g
        resid = tmat @ fold.y_std - g_p_tr
        preds = [F.fit_fixed_filter(ff, k, bank_train[k], bank_query[k], reg_p_tr, reg_p_q, resid, gw)
                 for k in range(len(banks))]
        ff.yield_mean.fill_(LM)
        ff.yield_scale.fill_(LS)
        p_fil = np.mean([(g_p_q + h_) * LS + LM for h_ in preds], 0)
        mem.update(seed=seed, reg_net=reg_net, reg=reg, g=g, annual=annual, local=local, ff=ff, p_ann=p_ann,
                   p_fil=p_fil, a_rep_tr=a_rep_tr, a_rep_q=a_rep_q, l_rep_tr=l_rep_tr, l_rep_q=l_rep_q,
                   local_pred_tr=local_pred_tr, local_pred_q=local_pred_q)
        members.append(mem)
        log(f"origin {t}: member {seed} base networks done")

    # Weather-and-forecast correction per family (targets use annual-weather predictions of earlier origins).
    wf_modules = []
    for fam in FAMILIES:
        mod = WeatherForecastCorrection(P)
        past = [y for y in range(FIRST_FORECAST_YEAR, t)]
        if past:
            order = [(j, y) for j in range(P) for y in past]                              # prefecture-major rows
            pred = np.array([env[y - FIRST_FORECAST_YEAR][j] for j, y in order])
            obs = np.array([data.yields[data.index(y)][pid == j].mean() for j, y in order])
            e_mean = np.array([np.mean(records[y].annual[[MEMBER_SEEDS.index(s) for s in fam], j]) for j, y in order])
            F.fit_weather_forecast(mod, pred, obs - e_mean, np.array([j for j, _ in order]), counts)
        wf_modules.append(mod)
    for mem in members:
        mod = wf_modules[0 if mem["seed"] in FAMILIES[0] else 1]
        mem["wf"] = mod
        mem["c_wf"] = mod(torch.as_tensor(env[-1][None], dtype=torch.float64))[0].numpy() if t > FIRST_FORECAST_YEAR else np.zeros(P)
        mem["mu0"] = C.WEIGHT_ANNUAL * mem["p_ann"] + C.WEIGHT_FIXED_FILTER * mem["p_fil"] + \
            C.WEIGHT_WEATHER_FORECAST * mem["c_wf"] + C.WEIGHT_SST_INDEX * c_sst

    # History weighting network (responses from 2011) and history correction network (origins 2012..t-1).
    resp = np.array([r for r, (_, y) in enumerate(fold.train_rows) if y >= FIRST_RESPONSE_YEAR])
    rowpos = {key: r for r, key in enumerate(fold.train_rows)}
    refs = np.array([[rowpos[(fold.train_rows[r][0], fold.train_rows[r][1] - k)] for k in (3, 2, 1)] for r in resp])
    qrefs = np.array([[rowpos[(i, t - k)] for k in (3, 2, 1)] for i in range(n)])
    rgroups, _ = groups([fold.train_rows[r] for r in resp], pid)
    rmat = F.group_matrix(rgroups)
    age = np.array([-1.0, 0.0, 1.0])
    for mem in members:
        z = F.center(mem["l_rep_tr"], tgroups, tmat)
        qz = F.center(mem["l_rep_q"], pid, qmat)
        res = (fold.y_true - mem["local_pred_tr"]) / LS

        def vector(cur, ref):
            return np.concatenate([np.repeat(cur[:, None], 3, 1), z[ref], res[ref][..., None],
                                   np.broadcast_to(age[None, :, None], (len(cur), 3, 1))], 2)
        tx, qx = vector(z[resp], refs), vector(qz, qrefs)
        mu, sd = tx.mean((0, 1)), np.maximum(tx.std((0, 1)), 1e-4)
        txs, qxs = ((tx - mu) / sd).astype(np.float32), ((qx - mu) / sd).astype(np.float32)
        base = F.center(mem["local_pred_tr"][resp] + res[refs].mean(1) * LS, rgroups, rmat)
        target = (F.center(fold.y_true[resp], rgroups, rmat) - base) / LS
        hw = F.fit_history_weighting(txs, res[refs].astype(np.float32), target, rgroups, mem["seed"], device)
        with torch.no_grad():
            r_q = torch.as_tensor(res[qrefs], dtype=torch.float32, device=device)
            a = torch.softmax(torch.tanh(hw(torch.as_tensor(qxs, device=device))), -1)
            delta = ((a * r_q).sum(-1) - r_q.mean(-1)).double().cpu().numpy()
        q_base = F.center(mem["local_pred_q"] + res[qrefs].mean(1) * LS, pid, qmat)
        weighted = q_base + LS * F.center(delta, pid, qmat)
        mem.update(z=z, qz=qz, res=res, hw=hw, hw_mean=mu, hw_scale=sd, weighted=weighted, qx_raw=qx)
        # History correction network.
        past = [y for y in range(FIRST_CORRECTION_YEAR, t)]
        m_ix = MEMBER_SEEDS.index(mem["seed"])
        if past:
            rows = [(i, y) for y in past for i in range(n)]                                # origin-major
            cur = np.concatenate([records[y].local_representation[m_ix] for y in past])
            refrep = np.concatenate([records[y].reference_representation[m_ix] for y in past])
            refres = np.array([[res[rowpos[(i, y - k)]] for k in (3, 2, 1)] for i, y in rows])
            hx = np.concatenate([np.repeat(cur[:, None], 3, 1), refrep, refres[..., None],
                                 np.broadcast_to(age[None, :, None], (len(rows), 3, 1))], 2)
            hmu, hsd = hx.mean((0, 1)), np.maximum(hx.std((0, 1)), 1e-4)
            hxs, hqs = ((hx - hmu) / hsd).astype(np.float32), ((qx - hmu) / hsd).astype(np.float32)
            hgroups, _ = groups(rows, pid)
            hmat = F.group_matrix(hgroups)
            obs = np.array([data.yields[data.index(y)][i] for i, y in rows])
            weighted_past = np.concatenate([records[y].weighted_deviation[m_ix] for y in past])
            htarget = (F.center(obs, hgroups, hmat) - weighted_past) / LS
            hc = F.fit_history_correction(hxs, htarget, hgroups, mem["seed"], device)
            with torch.no_grad():
                dq = hc(torch.as_tensor(hqs, device=device)).mean(-1).double().cpu().numpy()
            d0 = weighted + LS * F.center(dq, pid, qmat)
            mem.update(hc=hc, hc_mean=hmu, hc_scale=hsd, hist_rows=rows, hist_x=hxs, hist_q=hqs)
        else:
            d0 = weighted
            mem.update(hc=None)
        mem["d0"] = d0

    record = OriginRecord(
        year=t, annual=np.array([m["p_ann"] for m in members]), fixed_filter=np.array([m["p_fil"] for m in members]),
        weather_forecast=np.array([m["c_wf"] for m in members]), sst_index=c_sst,
        base_level=np.array([m["mu0"] for m in members]), local_representation=np.array([m["qz"] for m in members]),
        reference_representation=np.array([m["z"][qrefs] for m in members]),
        weighted_deviation=np.array([m["weighted"] for m in members]), base_deviation=np.array([m["d0"] for m in members]))
    if not corrections:
        return None, record
    records = {**records, t: record}
    _fit_corrections(data, t, fold, records, members, env, c_mod, model, device, log)
    return model, record


def _state(record, center):
    """Eight prediction-state features (P, 6, 8) of one origin, divided by 20 and clipped."""
    mu0, ann = record.base_level, record.annual
    pool = mu0.mean(0)
    half = len(MEMBER_SEEDS) // 2
    out = []
    for m in range(len(MEMBER_SEEDS)):
        out.append(np.stack([mu0[m] - center, C.WEIGHT_FIXED_FILTER * (record.fixed_filter[m] - ann[m]),
                             C.WEIGHT_WEATHER_FORECAST * record.weather_forecast[m],
                             np.full(len(center), C.WEIGHT_SST_INDEX * record.sst_index), mu0[m] - pool, mu0.std(0),
                             mu0[half:].mean(0) - mu0[:half].mean(0), ann.std(0)], 1))
    return np.clip(np.array(out) / C.STATE_SCALE, -C.CLIP, C.CLIP)                      # (6, P, 8)


def _fit_corrections(data, t, fold, records, members, env, c_mod, model, device, log):
    pid = data.prefecture_ids
    n, P = len(pid), len(C.PREFECTURES)
    counts = data.counts
    LM, LS = fold.yield_mean, fold.yield_scale
    years = np.arange(FIRST_CORRECTION_YEAR, t)
    rowpos = {key: r for r, key in enumerate(fold.train_rows)}
    rows = [(i, int(y)) for i in range(n) for y in years]                                  # key order
    rgroups, rkeys = groups(rows, pid)
    rmat = F.group_matrix(rgroups)
    qmat = F.group_matrix(pid)
    w = counts[[p for p, _ in rkeys]] / (n * len(years))
    mask = np.array([y >= FIRST_MUNICIPAL_YEAR for _, y in rows], float)
    obs = np.array([data.yields[data.index(y)][i] for i, y in rows])
    obs_mean = rmat @ obs
    # Environment (2012..t-1 standardization) and SST indices (2008..t-1 standardization).
    env_past = env[:-1]
    centers = env_past.mean(0)                                                           # (P, 10)
    cen = env_past - centers
    wenv = np.repeat((counts / n / len(years))[None], len(years), 0)
    escale = np.maximum(np.sqrt((wenv[..., None] * cen ** 2).sum((0, 1))), 0.05)
    env_std = np.clip((env - centers) / escale, -C.CLIP, C.CLIP)                          # (years+1, P, 10)
    idx_past = data.sst_indices[[data.index(y) for y in fold.train_years]]
    imean, iscale = idx_past.mean(0), np.maximum(idx_past.std(0), 0.05)
    idx_std = np.clip((data.sst_indices - imean) / iscale, -C.CLIP, C.CLIP)
    level_center = np.mean([records[y].base_level.mean(0) for y in years], 0)
    states = {y: _state(records[y], level_center) for y in list(years) + [t]}
    # SST encoders (shared by the members) and their standardized features of 2012..t.
    maps, area, clim, sscale, ocean = sst_maps(data)
    zy = lambda y: int(y - FIRST_SST_YEAR)  # noqa: E731
    enc_maps = maps[[zy(y) for y in range(FIRST_SST_YEAR, t)]]
    encoders, sst_norm, smean, sstd = [], [], [], []
    feat_years = np.arange(SST_FEATURE_FIRST_YEAR, t + 1)
    for r in range(C.N_SST_REPLICAS):
        enc = F.fit_sst_encoder(enc_maps, area, r, t, device)
        codes = F.sst_codes(enc, maps[[zy(y) for y in feat_years]], device).astype(np.float32).astype(float)
        mu, sd = codes[:-1].mean(0), np.maximum(codes[:-1].std(0), 0.05)
        sst_norm.append(np.clip((codes - mu) / sd, -C.CLIP, C.CLIP).astype(np.float32))
        smean.append(mu)
        sstd.append(sd)
        encoders.append(enc.cpu())
    log(f"origin {t}: SST encoders done")
    sst_by_year = {int(y): np.stack([s[k] for s in sst_norm]) for k, y in enumerate(feat_years)}   # (10, 16)

    for m_ix, mem in enumerate(members):
        seed = mem["seed"]
        # 80 current features: annual-weather and regional representations, environment, SST indices, state.
        tr_ix = np.array([rowpos[r] for r in rows])
        raw = np.concatenate([mem["a_rep_tr"][tr_ix], mem["reg"][[i for i, _ in rows]],
                              np.array([env_std[y - FIRST_FORECAST_YEAR][pid[i]] for i, y in rows]),
                              np.array([idx_std[data.index(y)] for _, y in rows]),
                              np.array([states[y][m_ix][pid[i]] for i, y in rows])], 1)
        qraw = np.concatenate([mem["a_rep_q"], mem["reg"], env_std[-1][pid], np.repeat(idx_std[data.index(t)][None], n, 0),
                               states[t][m_ix][pid]], 1)
        fmu, fsd = raw.mean(0), np.maximum(raw.std(0), 1e-4)
        x80, q80 = ((raw - fmu) / fsd).astype(np.float32), ((qraw - fmu) / fsd).astype(np.float32)
        # 67 history features from the standardized history-correction inputs (origin-major -> key order).
        hpos = {key: k for k, key in enumerate(mem["hist_rows"])}
        hx = mem["hist_x"][[hpos[r] for r in rows]]
        hq = mem["hist_q"]
        x147 = np.concatenate([x80, hx[:, 0, :16], hx[:, :, 16:33].reshape(len(rows), 51)], 1)
        q147 = np.concatenate([q80, hq[:, 0, :16], hq[:, :, 16:33].reshape(n, 51)], 1)
        agg = lambda x, gid, mat: np.concatenate([mat @ x.astype(float), np.sqrt(np.maximum(  # noqa: E731
            mat @ ((x.astype(float) - (mat @ x.astype(float))[gid]) ** 2), 0))], 1).astype(np.float32)
        x294, q294 = agg(x147, rgroups, rmat), agg(q147, pid, qmat)
        # Targets: prefecture-year residuals of the earlier-origin base level and municipal deviations.
        base_level = np.array([records[y].base_level[m_ix][p] for p, y in rkeys])
        r_target = (obs_mean - base_level) / C.LOSS_SCALE
        d0_past = np.array([records[y].base_deviation[m_ix][i] for i, y in rows])
        s_target = (F.center(obs, rgroups, rmat) - d0_past) / C.LOSS_SCALE
        municipal = [F.fit_municipal(x147, s_target, rgroups, mask, seed + REPLICA_SEED_STEP * r, device) for r in range(5)]
        prefecture = [F.fit_prefecture(x294, r_target, w, seed + REPLICA_SEED_STEP * r, device) for r in range(5)]
        hidden_tr, hidden_q, u_tr, hmean, hstd = [], [], [], [], []
        with torch.no_grad():
            for net in prefecture:
                h, u = net(torch.as_tensor(x294, device=device))
                hq_, _ = net(torch.as_tensor(q294, device=device))
                h, hq_ = h.double().cpu().numpy(), hq_.double().cpu().numpy()
                mu, sd = h.mean(0), np.maximum(h.std(0), 0.05)
                hidden_tr.append(np.clip((h - mu) / sd, -C.CLIP, C.CLIP))
                hidden_q.append(np.clip((hq_ - mu) / sd, -C.CLIP, C.CLIP))
                u_tr.append(u.double().cpu().numpy())
                hmean.append(mu)
                hstd.append(sd)
        h64 = np.mean(hidden_tr, 0).astype(np.float32)
        u_mean = np.mean(u_tr, 0)
        fusion = []
        for r in range(C.N_SST_REPLICAS):
            xs = np.concatenate([h64, np.array([sst_by_year[y][r] for _, y in rkeys])], 1)
            fusion.append(F.fit_sst_fusion(xs, u_mean, r_target, w, seed + REPLICA_SEED_STEP * r, device))
        log(f"origin {t}: member {seed} corrections done")
        # Copy the member into the model.
        mm = model.members[m_ix]
        mm.regional.load_state_dict(mem["reg_net"].state_dict())
        mm.regional_mean.copy_(torch.as_tensor(fold.static_mean))
        mm.regional_scale.copy_(torch.as_tensor(fold.static_scale))
        mm.annual.load_state_dict(mem["annual"].cpu().state_dict())
        mm.local.load_state_dict(mem["local"].cpu().state_dict())
        mm.fixed_filter.load_state_dict(mem["ff"].state_dict())
        mm.weather_forecast.load_state_dict(mem["wf"].state_dict())
        mm.history_weighting.load_state_dict(mem["hw"].cpu().state_dict())
        mm.history_weighting_mean.copy_(torch.as_tensor(mem["hw_mean"]))
        mm.history_weighting_scale.copy_(torch.as_tensor(mem["hw_scale"]))
        mm.history_correction.load_state_dict(mem["hc"].cpu().state_dict())
        mm.history_correction_mean.copy_(torch.as_tensor(mem["hc_mean"]))
        mm.history_correction_scale.copy_(torch.as_tensor(mem["hc_scale"]))
        mm.residual_scale.fill_(LS)
        mm.level_center.copy_(torch.as_tensor(level_center))
        mm.feature_mean.copy_(torch.as_tensor(fmu))
        mm.feature_scale.copy_(torch.as_tensor(fsd))
        for k in range(5):
            mm.municipal_corrections[k].load_state_dict(municipal[k].cpu().state_dict())
            mm.prefecture_corrections[k].load_state_dict(prefecture[k].cpu().state_dict())
        mm.hidden_mean.copy_(torch.as_tensor(np.array(hmean)))
        mm.hidden_scale.copy_(torch.as_tensor(np.array(hstd)))
        for k in range(C.N_SST_REPLICAS):
            mm.sst_fusion[k].load_state_dict(fusion[k].cpu().state_dict())

    model.year.fill_(t)
    model.weather_center.copy_(torch.as_tensor(fold.center))
    model.weather_scale.copy_(torch.as_tensor(fold.scale))
    model.yield_mean.fill_(LM)
    model.yield_scale.fill_(LS)
    model.environment_center.copy_(torch.as_tensor(centers))
    model.environment_scale.copy_(torch.as_tensor(escale))
    model.index_mean.copy_(torch.as_tensor(imean))
    model.index_scale.copy_(torch.as_tensor(iscale))
    model.sst_latitude.copy_(torch.as_tensor(LATITUDE_SAMPLES))
    model.sst_longitude.copy_(torch.as_tensor(LONGITUDE_SAMPLES))
    model.sst_climatology.copy_(torch.as_tensor(clim))
    model.sst_scale.copy_(torch.as_tensor(sscale))
    model.ocean_fraction.copy_(torch.as_tensor(ocean))
    model.sst_feature_mean.copy_(torch.as_tensor(np.array(smean)))
    model.sst_feature_scale.copy_(torch.as_tensor(np.array(sstd)))
    for k, enc in enumerate(encoders):
        model.sst_encoders[k].load_state_dict(enc.state_dict())
    model.municipalities = list(zip(data.prefecture.tolist(), data.area_code.tolist()))


def fit_origins(data, origins, output, evaluate_from=2019, device=None, log=print):
    """Fit origins in increasing order from 2012, saving records (``origin_YYYY.npz``) and, for origins from
    ``evaluate_from``, checkpoints (``hwsf_YYYY.pt``) to ``output``. Existing records are reused."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for t in range(FIRST_CORRECTION_YEAR, max(origins) + 1):
        path = output / f"origin_{t}.npz"
        need_model = t in origins and t >= evaluate_from and not (output / f"hwsf_{t}.pt").exists()
        if path.exists() and not need_model:
            records[t] = OriginRecord.load(path)
            continue
        model, record = fit_origin(data, t, records, corrections=need_model, device=device, log=log)
        record.save(path)
        records[t] = record
        if model is not None:
            model.save_checkpoint(output / f"hwsf_{t}.pt", model.municipalities)
            log(f"origin {t}: saved {output / f'hwsf_{t}.pt'}")
    return records
