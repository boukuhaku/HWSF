"""Fitting routines of the training stages (Algorithm 1; Supplementary Sections S2-S5).

Every network uses AdamW (weight decay 0.01) with global gradient-norm clipping at 5. The regional network is fitted
full-batch in float64 (learning rate 0.003); all other networks in float32 (learning rate 0.001 unless stated).
"""
import os

import numpy as np
import torch
from torch.nn import functional as F

from ..calibration import FixedFilterRegression
from ..constants import CLIP, LOSS_SCALE
from ..nn.networks import HistoryNetwork, RegionalNetwork, ResidualMLP, SSTDecoder, SSTEncoder, WeatherNetwork

MAX_NORM = 5.0


def configure(deterministic=True):
    """Deterministic GPU kernels, no TF32."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _f32(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)


def _update(params, opt):
    torch.nn.utils.clip_grad_norm_(params, MAX_NORM)
    opt.step()


def group_matrix(ids, n_groups=None):
    """Equal-weight group-mean matrix (G, N) of integer group ids."""
    ids = np.asarray(ids)
    n_groups = int(ids.max()) + 1 if n_groups is None else n_groups
    m = np.zeros((n_groups, len(ids)))
    for g in range(n_groups):
        m[g, ids == g] = 1.0 / np.sum(ids == g)
    return m


def center(x, ids, matrix):
    return x - (matrix @ x)[ids]


# ---------------------------------------------------------------------------------------------- regional network
def init_regional(seed):
    """Uniform(+-1/sqrt(fan_in)) initialization from NumPy default_rng(seed), weights before biases, layer by layer."""
    rng = np.random.default_rng(seed)
    net = RegionalNetwork().double()
    for layer, (a, b) in zip((net.hidden1, net.hidden2, net.output), ((26, 16), (16, 16), (16, 1))):
        lim = 1 / np.sqrt(a)
        w, bias = rng.uniform(-lim, lim, (a, b)), rng.uniform(-lim, lim, (1, b))
        layer.weight.data = torch.as_tensor(w.T.copy())
        layer.bias.data = torch.as_tensor(bias[0].copy())
    return net


def fit_regional(inputs, target, seed, steps, checkpoints=(), query=None):
    """Full-batch fit of the regional network on standardized inputs (n, 26) and standardized yields (n,).

    Returns the network after ``steps`` updates and the query baselines at each step in ``checkpoints``."""
    net = init_regional(seed)
    x, y = torch.as_tensor(inputs, dtype=torch.float64), torch.as_tensor(target, dtype=torch.float64)
    q = None if query is None else torch.as_tensor(query, dtype=torch.float64)
    opt = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.01, foreach=False)
    recorded = {}
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = ((net(x)[1] - y) ** 2).mean()
        loss.backward()
        _update(list(net.parameters()), opt)
        if step in checkpoints:
            with torch.no_grad():
                recorded[step] = net(q)[1].numpy().copy()
    return net, recorded


# ---------------------------------------------------------------------------------------------- weather networks
def fit_weather(x, regional, baseline, target, seed, depthwise, device, steps=600, batch_size=64):
    """Annual-weather (depthwise) or local-weather (full-channel) network with the regional network frozen.

    Loss: mean[(g + h - y)^2] + 0.1 mean(h^2) in standardized yield units, mini-batches of 64 drawn with
    NumPy default_rng(seed + 10000)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = WeatherNetwork(in_channels=x.shape[2], depthwise=depthwise).to(device)
    tx, ts, tg, ty = (_f32(a, device) for a in (x, regional, baseline, target))
    rng = np.random.default_rng(seed + 10000)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    params = list(net.parameters())
    for _ in range(steps):
        ii = torch.as_tensor(rng.choice(len(tx), batch_size, replace=False), device=device)
        opt.zero_grad(set_to_none=True)
        h = net(tx[ii], ts[ii])[1]
        loss = ((tg[ii] + h - ty[ii]) ** 2).mean() + 0.1 * (h * h).mean()
        loss.backward()
        _update(params, opt)
    return net


def fit_film(net, x, regional, baseline, target, seed, device, steps=200):
    """FiLM regional conditioning of the fitted local-weather network: 200 updates on batches of min(512, n) drawn
    with default_rng(seed + 127000); learning rates 1e-4 (existing weights) and 1e-3 (modulation maps); penalty
    0.1 mean[(h - h_initial)^2] towards the network before conditioning."""
    tx, ts, tg, ty = (_f32(a, device) for a in (x, regional, baseline, target))
    with torch.no_grad():
        h0 = net(tx, ts)[1].detach()
    net.add_film()
    batch = min(512, len(tx))
    rng = np.random.default_rng(seed + 127000)
    indices = [torch.as_tensor(rng.choice(len(tx), batch, replace=False), device=device) for _ in range(steps)]
    film = list(net.film.parameters())
    base = [p for n, p in net.named_parameters() if not n.startswith("film.")]
    opt = torch.optim.AdamW([dict(params=base, lr=1e-4), dict(params=film, lr=1e-3)], weight_decay=0.01)
    params = base + film
    for ii in indices:
        opt.zero_grad(set_to_none=True)
        h = net(tx[ii], ts[ii])[1]
        loss = ((tg[ii] + h - ty[ii]) ** 2).mean() + 0.1 * (h * h).mean() + 0.1 * ((h - h0[ii]) ** 2).mean()
        loss.backward()
        _update(params, opt)
    return net


@torch.no_grad()
def weather_outputs(net, x, regional, device, batch=4096):
    """16-element weather representation and scalar output (float64 from the float32 forward)."""
    hs, qs = [], []
    for a in range(0, len(x), batch):
        h, q = net(_f32(x[a:a + batch], device), _f32(regional[a:a + batch], device))
        hs.append(h.double().cpu().numpy())
        qs.append(q.double().cpu().numpy())
    return np.concatenate(hs), np.concatenate(qs)


# ---------------------------------------------------------------------------------------------- statistical calibration
def standardize_block(train, query, weights, mass, floor=0.05):
    """Weighted mean and SD (floor), clipping at +-6, and the block multiplier sqrt(mass / columns)."""
    mu = weights @ train
    sd = np.maximum(np.sqrt(weights @ ((train - mu) ** 2)), floor)
    factor = np.sqrt(mass / train.shape[1])
    return np.clip((train - mu) / sd, -CLIP, CLIP) * factor, np.clip((query - mu) / sd, -CLIP, CLIP) * factor, mu, sd


def fit_fixed_filter(module, k, train_features, query_features, train_regional, query_regional, residual, weights,
                     penalty=0.01):
    """One bank of the fixed-filter regression: weighted ridge (dual form, unpenalized intercept) on the standardized
    filter features and prefecture-mean regional features. Stores the equivalent primal coefficients in ``module``
    (a ``FixedFilterRegression``) and returns the query residual prediction."""
    xd, qd, dmu, dsd = standardize_block(train_features.astype(float), query_features.astype(float), weights, 0.75)
    xs, qs, smu, ssd = standardize_block(train_regional.astype(float), query_regional.astype(float), weights, 0.25)
    x, q = np.concatenate([xd, xs], 1), np.concatenate([qd, qs], 1)
    bias = float(weights @ residual)
    mean = weights @ x
    xc = x - mean
    alpha = np.linalg.solve(xc @ xc.T + penalty * np.diag(1 / weights), residual - bias)
    coef = xc.T @ alpha
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    module.coefficient[k], module.intercept[k], module.design_mean[k] = t(coef), bias, t(mean)
    module.dynamic_mean[k], module.dynamic_scale[k] = t(dmu), t(dsd)
    module.static_mean[k], module.static_scale[k] = t(smu), t(ssd)
    return bias + (q - mean) @ coef


def fit_weather_forecast(module, predictors, targets, prefecture_index, counts, penalty=0.1):
    """Weather-and-forecast correction: prefecture-centered predictors (unweighted prefecture means over the
    training years), weighted scale (floor 0.05), clipping, multipliers sqrt(0.5/6) and sqrt(0.5/4); weighted ridge
    on the standardized targets. Only 0.5 sigma_r x beta is used at query time."""
    n_pref = int(prefecture_index.max()) + 1
    centers = np.stack([predictors[prefecture_index == j].mean(0) for j in range(n_pref)])
    centered = predictors - centers[prefecture_index]
    w = counts[prefecture_index] / counts[prefecture_index].sum()
    scale = np.maximum(np.sqrt(w @ (centered ** 2)), 0.05)
    mass = np.r_[np.full(6, np.sqrt(0.5 / 6)), np.full(4, np.sqrt(0.5 / 4))]
    x = np.clip(centered / scale, -CLIP, CLIP) * mass
    mu = float(w @ targets)
    sd = max(1.0, float(np.sqrt(w @ ((targets - mu) ** 2))))
    y = (targets - mu) / sd
    xc = x - w @ x
    beta = np.linalg.solve(xc.T @ (w[:, None] * xc) + penalty * np.eye(x.shape[1]), xc.T @ (w * (y - w @ y)))
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    module.center.copy_(t(centers))
    module.scale.copy_(t(scale))
    module.mass.copy_(t(mass))
    module.coefficient.copy_(t(beta))
    module.target_scale.fill_(sd)


def fit_sst_index(module, power, indices, yields, prefecture_weights, penalty=0.01):
    """SST-index correction: weighted ridge of official prefecture yields on 20 NASA POWER weather means and the 30
    SST-index values (multipliers sqrt(0.75/20), sqrt(0.25/24), sqrt(0.125/6)); only the 30 SST coefficients times
    the yield scale are kept for the query.

    power: (Y, P, 20); indices: (Y, 30); yields: (Y, P) for the training years; prefecture_weights: (P,)."""
    n_years, n_pref = yields.shape
    w = np.repeat(prefecture_weights[:, None], n_years, 1).ravel()
    w = w / w.sum()
    raw = np.concatenate([power.transpose(1, 0, 2).reshape(-1, 20), np.tile(indices[:, :24], (n_pref, 1))], 1)
    target = yields.T.ravel()
    mean = w @ raw
    scale = np.maximum(np.sqrt(w @ ((raw - mean) ** 2)), 0.05)
    ym = float(w @ target)
    ys = max(float(np.sqrt(w @ ((target - ym) ** 2))), 1.0)
    mass = np.r_[np.full(20, np.sqrt(0.75 / 20)), np.full(24, np.sqrt(0.25 / 24))]
    x = np.clip((raw - mean) / scale, -CLIP, CLIP) * mass
    west = indices[:, 24:]
    wm, ws = west.mean(0), np.maximum(west.std(0), 0.05)
    x = np.column_stack([x, np.tile(np.clip((west - wm) / ws, -CLIP, CLIP) * np.sqrt(0.125 / 6), (n_pref, 1))])
    y = (target - ym) / ys
    xc = x - w @ x
    beta = np.linalg.solve(xc.T @ (w[:, None] * xc) + penalty * np.eye(x.shape[1]), xc.T @ (w * (y - w @ y)))
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    module.mean.copy_(t(np.r_[mean[20:], wm]))
    module.scale.copy_(t(np.r_[scale[20:], ws]))
    module.coefficient.copy_(t(beta[20:]))
    module.yield_scale.fill_(ys)


# ---------------------------------------------------------------------------------------------- history networks
def _history_net(seed, device):
    torch.manual_seed(seed)
    return HistoryNetwork().to(device)


def fit_history_weighting(inputs, residual, target, ids, seed, device, steps=200, penalty=0.1):
    """History weighting network: scores of the three reference years -> tanh -> softmax weights; the correction
    sum_k a_k r_k - mean_k r_k, centered within prefecture-years, regresses the target with a 0.1 penalty."""
    net = _history_net(seed, device)
    x, r, y = _f32(inputs, device), _f32(residual, device), _f32(target, device)
    m = _f32(group_matrix(ids), device)
    ix = torch.as_tensor(np.asarray(ids), device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    params = list(net.parameters())
    for _ in range(steps):
        opt.zero_grad(set_to_none=False)
        a = torch.softmax(torch.tanh(net(x)), -1)
        delta = (a * r).sum(-1) - r.mean(-1)
        delta = delta - (m @ delta)[ix]
        loss = ((delta - y) ** 2).mean() + penalty * (delta ** 2).mean()
        first = bool((net.output.weight.detach() == 0).all())
        loss.backward()
        if first:          # identical scores of all reference years: the output bias has an analytic zero gradient
            net.output.bias.grad.zero_()
        _update(params, opt)
    return net


def fit_history_correction(inputs, target, ids, seed, device, steps=200, penalty=1.0):
    """History correction network: mean of the three scalar outputs, centered within prefecture-years; penalty 1.
    The shared output bias cancels under centering, so its gradient is set to zero."""
    net = _history_net(seed, device)
    x, y = _f32(inputs, device), _f32(target, device)
    m = _f32(group_matrix(ids), device)
    ix = torch.as_tensor(np.asarray(ids), device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    params = list(net.parameters())
    for _ in range(steps):
        opt.zero_grad(set_to_none=False)
        delta = net(x).mean(-1)
        delta = delta - (m @ delta)[ix]
        loss = ((delta - y) ** 2).mean() + penalty * (delta ** 2).mean()
        loss.backward()
        net.output.bias.grad.zero_()
        _update(params, opt)
    return net


# ---------------------------------------------------------------------------------------------- correction networks
def _mlp(in_features, seed, device):
    torch.manual_seed(seed)
    return ResidualMLP(in_features).to(device)


def fit_municipal(x, target, ids, mask, seed, device, steps=200):
    """Municipal correction network, eq. (10): q = a - prefecture-year mean of a; loss mean over the masked rows of
    (q/20 - s/20)^2 + (q/20)^2."""
    net = _mlp(x.shape[1], seed, device)
    tx, ty, tm = _f32(x, device), _f32(target, device), _f32(mask, device)
    m = _f32(group_matrix(ids), device)
    ix = torch.as_tensor(np.asarray(ids), device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    params = list(net.parameters())
    for _ in range(steps):
        opt.zero_grad(set_to_none=False)
        a = net(tx)[1]
        q = (a - (m @ a)[ix]) / LOSS_SCALE
        loss = (tm * ((q - ty) ** 2 + q ** 2)).sum() / tm.sum()
        loss.backward()
        _update(params, opt)
    return net


def fit_prefecture(x, target, weights, seed, device, steps=200):
    """Prefecture correction network, eq. (9): weighted (u/20 - r/20)^2 + (u/20)^2."""
    net = _mlp(x.shape[1], seed, device)
    tx, ty, tw = _f32(x, device), _f32(target, device), _f32(weights, device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    params = list(net.parameters())
    for _ in range(steps):
        opt.zero_grad(set_to_none=False)
        u = net(tx)[1] / LOSS_SCALE
        loss = (tw * ((u - ty) ** 2 + u ** 2)).sum()
        loss.backward()
        _update(params, opt)
    return net


def fit_sst_fusion(x, parent, target, weights, seed, device, steps=200):
    """SST fusion network, eq. (11), with the prefecture correction u fixed:
    weighted ((u + v)/20 - r/20)^2 + ((u + v)/20)^2 + (v/20)^2."""
    net = _mlp(x.shape[1], seed, device)
    tx, tu, ty, tw = _f32(x, device), _f32(parent, device), _f32(target, device), _f32(weights, device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01, foreach=False)
    params = list(net.parameters())
    for _ in range(steps):
        opt.zero_grad(set_to_none=False)
        v = net(tx)[1]
        level = (tu + v) / LOSS_SCALE
        loss = (tw * ((level - ty) ** 2 + level ** 2 + (v / LOSS_SCALE) ** 2)).sum()
        loss.backward()
        _update(params, opt)
    return net


# ---------------------------------------------------------------------------------------------- SST encoder
def masked_reconstruction_batch(maps, area_weight, replica, origin):
    """Masked-reconstruction samples from clean encoder images (n, 8, 10, 36) of the years before the origin: each
    cell is hidden with probability 0.5 in all six months, and every mask is paired with its complement."""
    n = len(maps)
    rng = np.random.default_rng(1930001 + 1009 * replica + 17 * origin)
    drop = rng.random((n, 10, 36)) < 0.5
    flags = np.concatenate([drop, ~drop])
    x = np.concatenate([maps, maps]).copy()
    x[:, :6] *= ~flags[:, None]
    x[:, 7] = flags
    target = np.concatenate([maps[:, :6], maps[:, :6]])
    weights = np.broadcast_to(flags[:, None] * area_weight, target.shape).copy()
    weights /= weights.sum((1, 2, 3), keepdims=True)
    return x.astype(np.float32), target.astype(np.float32), weights.astype(np.float32)


def fit_sst_encoder(maps, area_weight, replica, origin, device, steps=300):
    """Masked-reconstruction pretraining of one SST encoder-decoder (300 full-batch updates); returns the encoder.
    The loss is the area-weighted mean squared reconstruction error over the hidden ocean cells."""
    torch.manual_seed(1930001 + 1009 * replica)
    encoder, decoder = SSTEncoder().to(device), SSTDecoder().to(device)
    x, target, weights = (_f32(a, device) for a in masked_reconstruction_batch(maps, area_weight, replica, origin))
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01, foreach=False)
    for _ in range(steps):
        opt.zero_grad(set_to_none=False)
        loss = (weights * (decoder(encoder(x)) - target) ** 2).sum((1, 2, 3)).mean()
        loss.backward()
        _update(params, opt)
    return encoder


@torch.no_grad()
def sst_codes(encoder, maps, device):
    return encoder(_f32(maps, device)).double().cpu().numpy()


def new_fixed_filter_module(n_banks=6):
    return FixedFilterRegression(n_banks)


def finalize(net):
    """Detach a fitted network to the CPU in float32 (the regional network stays in float64)."""
    return net.cpu().eval()


__all__ = [n for n in dir() if not n.startswith("_")]
_ = F  # imported for subclasses that use functional ops
