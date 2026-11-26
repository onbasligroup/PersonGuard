from __future__ import annotations

import numpy as np
from scipy.fft import dct
from scipy.interpolate import BSpline


def _orthonormal_columns(M: np.ndarray, k: int) -> np.ndarray:
    Q, R = np.linalg.qr(M)
    keep = np.abs(np.diag(R)) > 1e-10 * max(1.0, np.abs(R).max())
    Q = Q[:, keep]
    if Q.shape[1] < k:
        raise ValueError(f"basis has rank {Q.shape[1]} < k={k}")
    s = np.sign(Q[np.argmax(np.abs(Q), axis=0), np.arange(Q.shape[1])])
    return (Q * s)[:, :k]


def dct_matrix(d: int) -> np.ndarray:
    return dct(np.eye(d), type=2, norm="ortho", axis=0).T


def dct_prefix(d: int, k: int) -> np.ndarray:
    return dct_matrix(d)[:, :k]


def dct_mask(d: int, idx) -> np.ndarray:
    return dct_matrix(d)[:, np.asarray(idx, dtype=int)]


def random_dc_mask_indices(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    others = rng.permutation(np.arange(1, d))[: k - 1]
    return np.concatenate([[0], np.sort(others)])


def random_mask_indices(d: int, k: int, seed: int) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).permutation(d)[:k])


def identity(d: int) -> np.ndarray:
    return np.eye(d)


def public_pca(public_residuals: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(public_residuals, dtype=np.float64)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    if Vt.shape[0] < k:
        raise ValueError("not enough public samples for requested PCA rank")
    return _orthonormal_columns(Vt[:k].T, k)


def haar_like(d: int, k: int) -> np.ndarray:
    cols = [np.ones(d)]
    level = 1
    while len(cols) < 4 * d and 2**level <= d:
        edges = np.linspace(0, d, 2**level + 1).round().astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            v = np.zeros(d)
            v[a:b] = 1.0
            cols.append(v)
        level += 1
    for j in range(d):
        v = np.zeros(d)
        v[j] = 1.0
        cols.append(v)
    return _orthonormal_columns(np.stack(cols, axis=1), k)


def bspline(d: int, k: int, degree: int = 3) -> np.ndarray:
    degree = min(degree, k - 1)
    x = np.linspace(0.0, 1.0, d)
    n_inner = k - degree - 1
    inner = np.linspace(0, 1, n_inner + 2)[1:-1] if n_inner > 0 else np.array([])
    t = np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])
    B = np.zeros((d, k))
    for j in range(k):
        c = np.zeros(k)
        c[j] = 1.0
        B[:, j] = BSpline(t, c, degree, extrapolate=False)(x)
    B = np.nan_to_num(B)
    B[-1, -1] = 1.0
    return _orthonormal_columns(B, k)


def make_basis(name: str, d: int, k: int, public_residuals=None, seed: int = 0) -> np.ndarray:
    if name == "dct":
        return dct_prefix(d, k)
    if name == "full":
        return identity(d)
    if name == "pca":
        return public_pca(public_residuals, k)
    if name == "haar":
        return haar_like(d, k)
    if name == "bspline":
        return bspline(d, k)
    if name == "random_dc":
        return dct_mask(d, random_dc_mask_indices(d, k, seed))
    if name == "random_nodc":
        return dct_mask(d, random_mask_indices(d, k, seed))
    raise ValueError(f"unknown basis {name}")
