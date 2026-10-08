"""Unit tests of the training routines on small synthetic data."""
import numpy as np
import torch

from hwsf.calibration import FixedFilterRegression
from hwsf.training import fit as F


def test_fixed_filter_primal_equals_dual():
    rng = np.random.default_rng(0)
    n, m = 12, 30
    train, query = rng.normal(size=(n, m)), rng.normal(size=(3, m))
    reg_tr, reg_q = rng.normal(size=(n, 16)), rng.normal(size=(3, 16))
    resid, w = rng.normal(size=n), rng.uniform(1, 2, n)
    w /= w.sum()
    module = FixedFilterRegression(1, dynamic=m, static=16)
    primal = F.fit_fixed_filter(module, 0, train, query, reg_tr, reg_q, resid, w)
    # Dual form (kernel ridge with an unpenalized intercept) on the same standardized design.
    xd, qd, _, _ = F.standardize_block(train, query, w, 0.75)
    xs, qs, _, _ = F.standardize_block(reg_tr, reg_q, w, 0.25)
    x, q = np.c_[xd, xs], np.c_[qd, qs]
    bias, mean = w @ resid, w @ x
    alpha = np.linalg.solve((x - mean) @ (x - mean).T + 0.01 * np.diag(1 / w), resid - bias)
    dual = bias + (q - mean) @ (x - mean).T @ alpha
    np.testing.assert_allclose(primal, dual, atol=1e-10)


def test_masked_reconstruction_pairs_complementary_masks():
    rng = np.random.default_rng(1)
    maps = rng.normal(size=(4, 8, 10, 36)).astype(np.float32)
    maps[:, 6], maps[:, 7] = 0.5, 0.0
    area = rng.uniform(0, 1, (10, 36))
    x, target, weights = F.masked_reconstruction_batch(maps, area, replica=0, origin=2020)
    flags = x[:, 7].astype(bool)
    np.testing.assert_array_equal(flags[:4], ~flags[4:])                 # each mask paired with its complement
    assert np.all(x[:, :6][np.broadcast_to(flags[:, None], (8, 6, 10, 36))] == 0)   # hidden cells are zero
    np.testing.assert_allclose(weights.sum((1, 2, 3)), 1, rtol=1e-6)
    np.testing.assert_array_equal(target[:4], maps[:, :6])


def test_regional_network_reduces_loss():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 26))
    y = x[:, 0] - 0.5 * x[:, 1] + 0.1 * rng.normal(size=40)
    net0 = F.init_regional(7001)
    with torch.no_grad():
        before = float(((net0(torch.as_tensor(x))[1].numpy() - y) ** 2).mean())
    net, _ = F.fit_regional(x, y, 7001, 200)
    with torch.no_grad():
        after = float(((net(torch.as_tensor(x))[1].numpy() - y) ** 2).mean())
    assert after < 0.5 * before


def test_group_center():
    ids = np.array([0, 0, 1, 1, 1])
    m = F.group_matrix(ids)
    x = np.array([1.0, 3.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(F.center(x, ids, m), [-1, 1, -2, 0, 2])
