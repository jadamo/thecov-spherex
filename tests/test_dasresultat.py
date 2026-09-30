"""thecov.dasresultat: reading DasResultat randoms, shot noise, and the binned-multipole model."""
import numpy as np
import pytest

from thecov import GaussianCovariance
from thecov.dasresultat import read_dr_randoms, tracer_from_dr_randoms, shotnoise, model_from_binned

h5py = pytest.importorskip('h5py')


def sphere_randoms(rng, R, n, center=(0.0, 0.0, 1400.0)):
    x = rng.uniform(-R, R, size=(3 * n, 3))
    return x[np.sum(x * x, 1) < R * R][:n] + np.asarray(center)


def write_h5(path, cols):
    with h5py.File(path, 'w') as f:
        for name, data in cols.items():
            f.create_dataset(name, data=data)
    return path


def test_read_split_positions_and_weight_product(tmp_path):
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(50, 3))
    sys_w, fkp_w = rng.uniform(0.5, 1.5, 50), rng.uniform(0.5, 1.5, 50)
    path = write_h5(tmp_path / 'r.h5', {'position_x': pos[:, 0], 'position_y': pos[:, 1],
                                        'position_z': pos[:, 2], 'SYS_WEIGHT': sys_w,
                                        'fkp_weights': fkp_w, 'NZ': np.full(50, 1e-4)})
    r = read_dr_randoms(path)
    assert np.allclose(r['POSITION'], pos)
    assert np.allclose(r['WEIGHT'], sys_w * fkp_w)
    assert r['WEIGHT_COLUMNS'] == ('SYS_WEIGHT', 'fkp_weights')
    assert np.allclose(r['NZ'], 1e-4)


def test_read_legacy_columns_and_explicit_weights(tmp_path):
    rng = np.random.default_rng(2)
    pos = rng.normal(size=(20, 3))
    w, fkp_w = rng.uniform(0.5, 1.5, 20), rng.uniform(0.5, 1.5, 20)
    path = write_h5(tmp_path / 'r.hf5', {'Position': pos, 'WEIGHT': w, 'fkp_weights': fkp_w})
    r = read_dr_randoms(path)
    assert np.allclose(r['POSITION'], pos)
    assert np.allclose(r['WEIGHT'], w * fkp_w)
    assert 'NZ' not in r
    assert np.allclose(read_dr_randoms(path, weight_columns=('WEIGHT',))['WEIGHT'], w)
    assert np.allclose(read_dr_randoms(path, weight_columns=())['WEIGHT'], 1.0)
    with pytest.raises(KeyError):
        read_dr_randoms(path, weight_columns=('SYS_WEIGHT',))


def test_missing_positions(tmp_path):
    path = write_h5(tmp_path / 'r.h5', {'RA': np.zeros(5)})
    with pytest.raises(KeyError):
        read_dr_randoms(path)


def test_shotnoise_uniform_sphere(tmp_path):
    """Constant nbar and unit weights: int S / I = (1 + alpha) nbar V / (nbar^2 V) = (1 + alpha) / nbar."""
    rng = np.random.default_rng(3)
    R, n, nbar = 300.0, 20000, 3e-4
    pos = sphere_randoms(rng, R, n)
    alpha = nbar * 4 * np.pi / 3 * R ** 3 / n
    path = write_h5(tmp_path / 'r.h5', {'Position': pos, 'NZ': np.full(n, nbar)})
    A = tracer_from_dr_randoms(path, '0', alpha)
    cov = GaussianCovariance([A], np.linspace(0.01, 0.1, 5))
    assert shotnoise(cov, '0') == pytest.approx((1 + alpha) / nbar, rel=1e-12)
    cov.set_normalization('0', '0', 2 * cov.I('0', '0'))
    assert shotnoise(cov, '0') == pytest.approx(0.5 * (1 + alpha) / nbar, rel=1e-12)


def test_model_from_binned_extension_and_shotnoise():
    k_edges = np.linspace(0.01, 0.11, 6)
    k = 0.5 * (k_edges[1:] + k_edges[:-1])
    ells = [0, 2]
    pairs = [('0', '0'), ('0', '1'), ('1', '1')]
    # linear in k, so linear extension to the edges is exact
    psm = np.array([[(i + 1) * (1e4 - 5e4 * k), (i + 1) * 1e3 * np.ones_like(k)] for i in range(3)])
    sn = {'0': 100.0, '1': 300.0}
    model = model_from_binned(k_edges, k, psm, ells, pairs, shotnoise=sn)
    kk = np.linspace(k_edges[0], k_edges[-1], 50)
    assert np.allclose(model('0', '0', 0, kk), 1e4 - 5e4 * kk - 100.0)
    assert np.allclose(model('0', '1', 0, kk), 2 * (1e4 - 5e4 * kk))    # cross: no shot noise
    assert np.allclose(model('1', '0', 0, kk), model('0', '1', 0, kk))  # symmetric
    assert np.allclose(model('1', '1', 0, kk), 3 * (1e4 - 5e4 * kk) - 300.0)
    assert np.allclose(model('1', '1', 2, kk), 3e3)                     # ell = 2 untouched
    assert np.allclose(model('0', '0', 4, kk), 0.0)                     # not given -> zero
    const = model_from_binned(k_edges, k, psm, ells, pairs, extend='constant')
    assert const('0', '0', 0, k_edges[:1])[0] == pytest.approx(psm[0, 0, 0])
    assert const('0', '0', 0, k_edges[-1:])[0] == pytest.approx(psm[0, 0, -1])


def test_model_from_binned_validation():
    k_edges = np.linspace(0.01, 0.11, 6)
    k = 0.5 * (k_edges[1:] + k_edges[:-1])
    psm = np.ones((1, 1, 5))
    with pytest.raises(ValueError):
        model_from_binned(k_edges, k, np.ones((2, 1, 5)), [0], [('0', '0')])
    with pytest.raises(ValueError):
        model_from_binned(k_edges, k + 0.02, psm, [0], [('0', '0')])
    with pytest.raises(ValueError):
        model_from_binned(k_edges[:-1], k, psm, [0], [('0', '0')])
    with pytest.raises(ValueError):
        model_from_binned(k_edges, k, psm, [0], [('0', '0')], extend='spline')
