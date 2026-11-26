from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .accounting import noise_multiplier


def clip_rows(X: np.ndarray, C: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    if np.isnan(C) or C < 0:
        raise ValueError("clip radius must be nonnegative")
    norms = np.linalg.norm(X, axis=-1)
    if not np.isfinite(C):
        return X.copy(), norms
    scale = np.minimum(1.0, C / np.maximum(norms, 1e-300))
    return X * scale[..., None], norms


@dataclass
class ReleaseResult:
    estimates: np.ndarray
    coeffs: np.ndarray
    q: np.ndarray
    sigma: float
    sensitivity: float
    clip_rate_inner: float
    clip_rate_outer: float
    info: dict = field(default_factory=dict)


def project_and_clip(S: np.ndarray, r: np.ndarray, A: np.ndarray, C: float, R: float = np.inf):
    X, outer_norms = clip_rows(S - r, R)
    Z, inner_norms = clip_rows(X @ A, C)
    return X, Z, outer_norms, inner_norms


def release(S: np.ndarray, r: np.ndarray, A: np.ndarray, C: float, eps: float, delta: float,
            rng: np.random.Generator, draws: int = 1, R: float = np.inf,
            sigma: float | None = None) -> ReleaseResult:
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2 or not len(S) or not np.isfinite(S).all():
        raise ValueError("S must contain finite person rows")
    A, r = np.asarray(A, float), np.asarray(r, float)
    if A.ndim != 2 or A.shape[0] != S.shape[1] or r.shape != (S.shape[1],):
        raise ValueError("incompatible reference/basis dimensions")
    if not np.isfinite(A).all() or not np.isfinite(r).all():
        raise ValueError("reference and basis must be finite")
    if not np.allclose(A.T @ A, np.eye(A.shape[1]), atol=1e-8):
        raise ValueError("basis columns must be orthonormal")
    n = S.shape[0]
    X, Z, on, inn = project_and_clip(S, r, A, C, R)
    q = Z.mean(axis=0)
    sens = 2.0 * min(C, R) / n
    if sigma is None:
        sigma = noise_multiplier(eps, delta) * sens
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("noise scale must be finite and nonnegative")
    coeffs = q[None, :] + sigma * rng.standard_normal((draws, A.shape[1]))
    est = r[None, :] + coeffs @ A.T
    return ReleaseResult(est, coeffs, q, float(sigma), float(sens),
                         float(np.mean(inn > C)), float(np.mean(on > R)) if np.isfinite(R) else 0.0)


def error_decomposition(S: np.ndarray, r: np.ndarray, A: np.ndarray, C: float, R: float,
                        sigma: float) -> dict:
    S = np.asarray(S, dtype=np.float64)
    d = S.shape[1]
    mu_raw = S.mean(axis=0)
    X, Z, _, _ = project_and_clip(S, r, A, C, R)
    mu_R = r + X.mean(axis=0)
    q_curve = r + Z.mean(axis=0) @ A.T
    proj_only = r + (X @ A).mean(axis=0) @ A.T
    biases = [mu_R - mu_raw, proj_only - mu_R, q_curve - proj_only]
    cross = 2 * sum(float(np.dot(biases[i], biases[j]))
                    for i in range(3) for j in range(i + 1, 3)) / d
    total = float(np.sum((q_curve - mu_raw) ** 2) / d)
    return {
        "bias_cross_mse": cross,
        "expected_total_mse": total + A.shape[1] * sigma**2 / d,
        "outer_clip_bias_mse": float(np.sum((mu_R - mu_raw) ** 2) / d),
        "projection_bias_mse": float(np.sum((proj_only - mu_R) ** 2) / d),
        "inner_clip_bias_mse": float(np.sum((q_curve - proj_only) ** 2) / d),
        "det_total_bias_mse": float(np.sum((q_curve - mu_raw) ** 2) / d),
        "noise_mse": float(A.shape[1] * sigma**2 / d),
    }


def james_stein_toward_reference(coeffs: np.ndarray, sigma: float) -> np.ndarray:
    k = coeffs.shape[-1]
    if k < 3:
        return coeffs
    nrm2 = np.sum(coeffs**2, axis=-1, keepdims=True)
    w = np.maximum(0.0, 1.0 - (k - 2) * sigma**2 / np.maximum(nrm2, 1e-300))
    return w * coeffs


def public_quantile_radius(public_vectors: np.ndarray, q: float) -> float:
    return float(np.quantile(np.linalg.norm(public_vectors, axis=1), q))
