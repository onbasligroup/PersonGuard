from __future__ import annotations

import numpy as np

DEFAULT_BANDS = [(0.5, 5.0), (5.0, 15.0), (15.0, 40.0), (40.0, 60.0)]


def _2d(x):
    x = np.asarray(x, dtype=np.float64)
    return x[None, :] if x.ndim == 1 else x


def curve_metrics(est: np.ndarray, target: np.ndarray, freqs: np.ndarray,
                  bands=DEFAULT_BANDS) -> dict[str, np.ndarray]:
    E = _2d(est)
    t = np.asarray(target, dtype=np.float64)
    diff = E - t
    out = {
        "rel_l2": np.linalg.norm(diff, axis=1) / max(np.linalg.norm(t), 1e-12),
        "rmse_db": 10.0 * np.sqrt(np.mean(diff**2, axis=1)),
        "max_abs_db": 10.0 * np.max(np.abs(diff), axis=1),
    }
    errs = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi) if hi < freqs.max() else (freqs >= lo) & (freqs <= hi)
        if m.any():
            errs.append(np.abs(E[:, m].mean(1) - t[m].mean()))
    out["band_mean_logpower_err_db"] = 10.0 * np.mean(errs, axis=0)
    Ec = E - E.mean(1, keepdims=True)
    tc = t - t.mean()
    out["shape_cosine"] = (Ec @ tc) / np.maximum(np.linalg.norm(Ec, axis=1) * np.linalg.norm(tc), 1e-12)
    out["cosine"] = (E @ t) / np.maximum(np.linalg.norm(E, axis=1) * np.linalg.norm(t), 1e-12)
    return out


def within_tolerance(est, target, tol_db: float) -> np.ndarray:
    return curve_metrics(est, target, np.arange(np.asarray(target).size))["rmse_db"] <= tol_db
