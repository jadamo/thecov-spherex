"""Adapters between DasResultat products and thecov.

DasResultat (the SPHEREx L4 power-spectrum pipeline) hands the covariance three things:

    randoms   : spatialized HDF5 catalogues, one per tracer and redshift bin, with the observer at
                the origin. Column names vary between pipeline versions: positions as 'Position'
                (N, 3) or 'position_x/y/z', a systematics weight 'SYS_WEIGHT' (older files:
                'WEIGHT'), an optional 'fkp_weights' and an optional 'NZ'.
    k binning : bin edges written by DasResultat's KBinner.
    multipoles: unwindowed P_L^{AB} at the bin centres, shape (nps, nl, nk) per redshift bin, with
                shot noise included in the monopole of each auto-spectrum.

This module turns them into Tracer and PowerSpectrumModel objects. It does not depend on
DasResultat, so the ordering of the tracer pairs (DasResultat's get_samples_ij) is passed in by the
caller. h5py is needed only by read_dr_randoms.

Typical use, per redshift bin:

    tracers = [tracer_from_dr_randoms(f, str(t), alpha[t]) for t, f in enumerate(random_files)]
    cov = GaussianCovariance(tracers, k_edges, ells=(0, 2, 4), L_max=4)
    sn = {t.name: shotnoise(cov, t.name) for t in tracers}
    cov.set_model(model_from_binned(k_edges, k, psm[iz], ells, pairs, shotnoise=sn))
    C, labels = cov.covariance(pairs)
"""
from __future__ import annotations

import numpy as np

from .kernels import PowerSpectrumModel
from .tracers import Tracer, Window

# Systematics weight: the first of these present in the file is used.
SYS_WEIGHT_COLUMNS = ('SYS_WEIGHT', 'WEIGHT')
# FKP weight, multiplied in when present ('fkp_weight' is the name used by some older files).
FKP_WEIGHT_COLUMNS = ('fkp_weights', 'fkp_weight')


def _default_weight_columns(columns) -> tuple:
    sys_w = [c for c in SYS_WEIGHT_COLUMNS if c in columns][:1]
    fkp_w = [c for c in FKP_WEIGHT_COLUMNS if c in columns][:1]
    return tuple(sys_w + fkp_w)


def read_dr_randoms(path, weight_columns=None) -> dict:
    """Read a DasResultat spatialized random catalogue into the dict Tracer expects.

    Parameters
    ----------
    path           : HDF5 file written by DasResultat's spatialize.py.
    weight_columns : columns whose product is the total weight w(x) applied to the density field.
                     Default: the systematics weight (SYS_WEIGHT, else WEIGHT) times the FKP weight
                     (fkp_weights) when present, and unit weights if the file has neither.

    Returns
    -------
    dict with 'POSITION' (N, 3), 'WEIGHT' (N,), 'NZ' (N,) if the file has it, and 'WEIGHT_COLUMNS',
    the columns that went into 'WEIGHT' (ignored by Tracer).
    """
    import h5py

    with h5py.File(path, 'r') as f:
        columns = set(f.keys())
        if 'POSITION' in columns:
            pos = f['POSITION'][:]
        elif 'Position' in columns:
            pos = f['Position'][:]
        elif {'position_x', 'position_y', 'position_z'} <= columns:
            pos = np.column_stack([f['position_x'][:], f['position_y'][:], f['position_z'][:]])
        else:
            raise KeyError(f"{path}: no Cartesian positions ('POSITION', 'Position' or "
                           f"'position_x/y/z'); columns are {sorted(columns)}")
        pos = np.asarray(pos, dtype=float)

        if weight_columns is None:
            weight_columns = _default_weight_columns(columns)
        missing = [c for c in weight_columns if c not in columns]
        if missing:
            raise KeyError(f"{path}: weight columns {missing} not found; columns are {sorted(columns)}")
        w = np.ones(len(pos))
        for c in weight_columns:
            w = w * np.asarray(f[c][:], dtype=float)

        randoms = {'POSITION': pos, 'WEIGHT': w, 'WEIGHT_COLUMNS': tuple(weight_columns)}
        if 'NZ' in columns:
            randoms['NZ'] = np.asarray(f['NZ'][:], dtype=float)
    return randoms


def tracer_from_dr_randoms(path, name, alpha, weight_columns=None, **tracer_kwargs) -> Tracer:
    """Tracer built from a DasResultat random catalogue (see read_dr_randoms).

    alpha is the weighted galaxy-to-random ratio; extra keyword arguments go to Tracer. Without an
    'NZ' column, Tracer estimates nbar from the randoms (and warns).
    """
    randoms = read_dr_randoms(path, weight_columns=weight_columns)
    return Tracer(name, randoms, alpha, **tracer_kwargs)


def shotnoise(cov, name) -> float:
    """Shot noise of the auto-spectrum of `name` as the covariance sees it:
    int S^A / I_AA = (1 + alpha) int nbar w^2 / I_AA, with I_AA as set on `cov` (so it follows
    set_normalization). This is the constant to remove from the monopole of a model that includes
    shot noise, since thecov adds it back through the shot-noise window."""
    return Window('S', cov._tracer(name)).integral() / cov.I(name, name)


def model_from_binned(k_edges, k, psm, ells, pairs, shotnoise=None, extend='linear') -> PowerSpectrumModel:
    """PowerSpectrumModel from multipoles tabulated at one k per bin.

    The shell kernels integrate over the full width of every bin, so the tabulation has to reach the
    outer edges k_edges[0] and k_edges[-1], while DasResultat provides the bin centres only. The
    table is therefore extended to the two outer edges, linearly from the two nearest points
    (extend='linear') or by holding the end values (extend='constant').

    Parameters
    ----------
    k_edges   : (nk + 1,) bin edges.
    k         : (nk,) k at which psm is tabulated, one inside each bin.
    psm       : (nps, nl, nk) multipoles for one redshift bin.
    ells      : (nl,) multipole of each index along axis 1.
    pairs     : nps tracer-name pairs (A, B), in the order of axis 0.
    shotnoise : optional {name: value}, subtracted from the monopole of the auto-spectrum of each
                listed tracer (see shotnoise()).
    """
    k_edges = np.asarray(k_edges, dtype=float)
    k = np.asarray(k, dtype=float)
    psm = np.asarray(psm, dtype=float)
    ells = [int(l) for l in ells]
    pairs = [tuple(str(x) for x in p) for p in pairs]
    if psm.shape != (len(pairs), len(ells), len(k)):
        raise ValueError(f"psm has shape {psm.shape}, expected (nps, nl, nk) = "
                         f"{(len(pairs), len(ells), len(k))}")
    if len(k_edges) != len(k) + 1:
        raise ValueError(f"{len(k_edges)} bin edges for {len(k)} k values")
    inside = (k > k_edges[:-1]) & (k < k_edges[1:])
    if not np.all(inside):
        raise ValueError(f"k values {k[~inside]} do not lie strictly inside their bins")
    if extend not in ('linear', 'constant'):
        raise ValueError("extend must be 'linear' or 'constant'")

    k_ext = np.concatenate([[k_edges[0]], k, [k_edges[-1]]])
    model = PowerSpectrumModel()
    for ips, (A, B) in enumerate(pairs):
        multipoles = {}
        for il, L in enumerate(ells):
            P = psm[ips, il].copy()
            if L == 0 and A == B and shotnoise is not None and A in shotnoise:
                P = P - shotnoise[A]
            if extend == 'constant' or len(k) < 2:
                lo, hi = P[0], P[-1]
            else:
                lo = P[0] + (P[1] - P[0]) / (k[1] - k[0]) * (k_edges[0] - k[0])
                hi = P[-1] + (P[-1] - P[-2]) / (k[-1] - k[-2]) * (k_edges[-1] - k[-1])
            multipoles[L] = (k_ext, np.concatenate([[lo], P, [hi]]))
        model.add((A, B), multipoles)
    return model
