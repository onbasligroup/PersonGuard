from __future__ import annotations

import numpy as np
from scipy.stats import beta, rankdata

from smdp.accounting import noise_multiplier
from smdp.mechanism import clip_rows, release


def _scores(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("attack scores must be finite")
    if len(values) < 4:
        raise ValueError("at least four observations per world are needed")
    return values


def auc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    si, so = _scores(scores_in), _scores(scores_out)
    ranks = rankdata(np.concatenate([si, so]), method="average")
    n1, n0 = len(si), len(so)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _cal_eval(scores):
    scores = _scores(scores)
    cut = len(scores) // 2
    return scores[:cut], scores[cut:]


def _binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(beta.ppf(alpha / 2, k, n - k + 1)) if k else 0.0
    hi = float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def tpr_at_fpr_details(scores_in, scores_out, fpr: float, conf: float = 0.95) -> dict:
    if not 0 < fpr < 1 or not 0 < conf < 1:
        raise ValueError("fpr and conf must lie in (0, 1)")
    _, si = _cal_eval(scores_in)
    cal, so = _cal_eval(scores_out)
    threshold = float(np.quantile(cal, 1 - fpr, method="higher"))
    tp, fp = int(np.sum(si > threshold)), int(np.sum(so > threshold))
    tl, th = _binomial_ci(tp, len(si), 1 - conf)
    fl, fh = _binomial_ci(fp, len(so), 1 - conf)
    return {"tpr": tp / len(si), "tpr_ci_lo": tl, "tpr_ci_hi": th,
            "fpr_observed": fp / len(so), "fpr_ci_lo": fl, "fpr_ci_hi": fh,
            "threshold": threshold, "n_in_eval": len(si), "n_out_eval": len(so),
            "n_out_calibration": len(cal), "fpr_resolved": len(so) * fpr >= 20}


def tpr_at_fpr(scores_in, scores_out, fpr: float) -> float:
    return float(tpr_at_fpr_details(scores_in, scores_out, fpr)["tpr"])


def eps_lower_bound_details(scores_in, scores_out, delta: float, conf: float = 0.95,
                            n_thresholds: int = 99) -> dict:
    if not 0 <= delta < 1 or not 0 < conf < 1 or n_thresholds < 1:
        raise ValueError("invalid delta, confidence, or threshold count")
    ci, si = _cal_eval(scores_in)
    co, so = _cal_eval(scores_out)
    thresholds = np.unique(np.quantile(np.concatenate([ci, co]),
                                       np.linspace(0.01, 0.99, n_thresholds)))
    a = (1 - conf) / (4 * len(thresholds))
    best = 0.0
    for threshold in thresholds:
        ki, ko = int(np.sum(si > threshold)), int(np.sum(so > threshold))
        pi_lo = float(beta.ppf(a, ki, len(si) - ki + 1)) if ki else 0.0
        pi_hi = float(beta.ppf(1 - a, ki + 1, len(si) - ki)) if ki < len(si) else 1.0
        po_lo = float(beta.ppf(a, ko, len(so) - ko + 1)) if ko else 0.0
        po_hi = float(beta.ppf(1 - a, ko + 1, len(so) - ko)) if ko < len(so) else 1.0
        for num, den in ((pi_lo, po_hi), (po_lo, pi_hi),
                         (1 - pi_hi, 1 - po_lo), (1 - po_hi, 1 - pi_lo)):
            if num > delta and den > 0:
                best = max(best, float(np.log((num - delta) / den)))
    return {"eps_lb": best, "confidence": conf, "n_thresholds": len(thresholds),
            "n_in_calibration": len(ci), "n_out_calibration": len(co),
            "n_in_evaluation": len(si), "n_out_evaluation": len(so),
            "confidence_method": "independent_calibration_bonferroni_clopper_pearson"}


def eps_lower_bound(scores_in, scores_out, delta: float, conf: float = 0.95) -> float:
    return float(eps_lower_bound_details(scores_in, scores_out, delta, conf)["eps_lb"])


def summarize_attack(scores_in, scores_out) -> dict:
    si, so = _scores(scores_in), _scores(scores_out)
    result = {"auc": auc(si, so), "scores_in": si, "scores_out": so}
    for fpr, tag in ((0.01, "1pct"), (0.001, "0.1pct")):
        details = tpr_at_fpr_details(si, so, fpr)
        result[f"tpr_at_{tag}_fpr"] = details.pop("tpr")
        result.update({f"{tag}_{key}": value for key, value in details.items()})
    return result


class MeanReleaseAttack:

    def __init__(self, public_X: np.ndarray, n: int, sigma: float, ridge: float = 1e-6):
        X = np.asarray(public_X, dtype=float)
        if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all():
            raise ValueError("a finite public reference with >=2 people is required")
        self.m = X.mean(axis=0)
        Sig = np.atleast_2d(np.cov(X, rowvar=False))
        k = Sig.shape[0]
        reg = ridge * max(float(np.trace(Sig)) / k, 1e-12)
        self.W = np.linalg.pinv(Sig / n + (sigma**2 + reg) * np.eye(k))

    def score(self, x_t: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(Y) - self.m) @ self.W @ (x_t - self.m)


def canary_audit(r: np.ndarray, A: np.ndarray, C: float, R: float, n: int,
                 sigma: float | None, trials: int, rng: np.random.Generator,
                 eps: float = 1.0, delta: float = 1e-5) -> dict:
    r, A = np.asarray(r, float), np.asarray(A, float)
    if n < 1 or trials < 4 or not np.isfinite(C) or C <= 0 or (sigma is not None and sigma < 0):
        raise ValueError("canary needs positive finite C, n>=1, trials>=4, sigma>=0")
    c = min(C, R)
    D_in = np.repeat(r[None], n, axis=0)
    D_out = D_in.copy()
    D_in[-1] = r + c * A[:, 0]
    D_out[-1] = r - c * A[:, 0]
    a = release(D_in, r, A, C, eps, delta, rng, draws=trials, R=R, sigma=sigma)
    b = release(D_out, r, A, C, eps, delta, rng, draws=trials, R=R, sigma=sigma)
    result = summarize_attack(a.coeffs[:, 0], b.coeffs[:, 0])
    result.update({"audit_pair": "fixed_replace_one_plus_minus_first_basis_direction",
                   "audit_calls_shared_release": True, "sigma": float(a.sigma),
                   "noise_calibration": "production_default" if sigma is None else "explicit_override",
                   "sensitivity": float(a.sensitivity)})
    return result


def run_membership(pool_S: np.ndarray, public_S: np.ndarray, r: np.ndarray, A: np.ndarray | None,
                   C: float, R: float, n: int, sigma: float, trials: int, rng: np.random.Generator,
                   canary: bool = False) -> dict:
    pool_S, public_S = np.asarray(pool_S, float), np.asarray(public_S, float)
    if n < 1 or len(pool_S) < n + 1 or trials < 4:
        raise ValueError("membership needs n>=1, >=n+1 eligible people, and >=4 trials")
    if canary:
        if A is None:
            raise ValueError("canary audit requires a clipped basis release")
        return canary_audit(r, A, C, R, n, sigma, trials, rng)
    if A is None:
        pool, pub = pool_S, public_S
    else:
        pool = clip_rows(clip_rows(pool_S - r, R)[0] @ A, C)[0]
        pub = clip_rows(clip_rows(public_S - r, R)[0] @ A, C)[0]
    att = MeanReleaseAttack(pub, n, sigma)
    s_in, s_out = [], []
    for _ in range(trials):
        perm = rng.choice(len(pool), n + 1, replace=False)
        target_idx, neighbour_idx = perm[n], perm[n - 1]
        in_idx = np.concatenate([perm[:n - 1], [target_idx]])
        out_idx = np.concatenate([perm[:n - 1], [neighbour_idx]])
        if A is None:
            yi, yo = pool_S[in_idx].mean(0), pool_S[out_idx].mean(0)
        else:
            yi = release(pool_S[in_idx], r, A, C, 1.0, 1e-5, rng, R=R, sigma=sigma).coeffs[0]
            yo = release(pool_S[out_idx], r, A, C, 1.0, 1e-5, rng, R=R, sigma=sigma).coeffs[0]
        s_in.append(att.score(pool[target_idx], yi)[0])
        s_out.append(att.score(pool[target_idx], yo)[0])
    return summarize_attack(s_in, s_out)


def eligible_window_people(windows: np.ndarray, cap: int) -> np.ndarray:
    W = np.asarray(windows)
    if W.ndim != 3 or cap < 1 or cap > W.shape[1]:
        raise ValueError("windows must have shape (people, max_windows, d) and a valid cap")
    return np.flatnonzero(np.isfinite(W[:, :cap]).all(axis=(1, 2)))


def run_window_membership(pool_windows: np.ndarray, public_windows: np.ndarray,
                          r: np.ndarray, A: np.ndarray, C: float, R: float,
                          n: int, eps: float, delta: float, trials: int,
                          rng: np.random.Generator, cap: int) -> dict:
    eligible = eligible_window_people(pool_windows, cap)
    ref_ok = eligible_window_people(public_windows, cap)
    W = np.asarray(pool_windows, float)[eligible, :cap]
    P = np.asarray(public_windows, float)[ref_ok, :cap]
    if len(W) < n + 1 or len(P) < 2 or trials < 4:
        raise ValueError("not enough exact-cap people for window-level membership")
    d = W.shape[-1]
    def coefficients(V):
        X = clip_rows(V.reshape(-1, d) - r, R)[0]
        return clip_rows(X @ A, C)[0].reshape(len(V), cap, -1).mean(1)
    pool, pub = coefficients(W), coefficients(P)
    sigma = noise_multiplier(eps, delta) * 2 * min(C, R) / (n * cap)
    att = MeanReleaseAttack(pub, n, sigma)
    s_in, s_out = [], []
    for _ in range(trials):
        perm = rng.choice(len(W), n + 1, replace=False)
        target_idx = perm[n]
        in_idx = np.concatenate([perm[:n - 1], [target_idx]])
        out_idx = perm[:n]
        yi = release(W[in_idx].reshape(n * cap, d), r, A, C, eps, delta, rng, R=R).coeffs[0]
        yo = release(W[out_idx].reshape(n * cap, d), r, A, C, eps, delta, rng, R=R).coeffs[0]
        s_in.append(att.score(pool[target_idx], yi)[0])
        s_out.append(att.score(pool[target_idx], yo)[0])
    result = summarize_attack(s_in, s_out)
    result.update({"protected_unit": "window", "windows_per_person": cap,
                   "n_eligible_people": len(W), "sigma": sigma,
                   "target": "mean_of_individually_clipped_window_coefficients"})
    return result


def linkage_reid(half1: np.ndarray, half2: np.ndarray, gallery_size: int, rng: np.random.Generator,
                 public_mean: np.ndarray, public_std: np.ndarray, repeats: int = 20) -> dict:
    H1, H2 = np.asarray(half1, float), np.asarray(half2, float)
    ok = np.isfinite(H1).all(1) & np.isfinite(H2).all(1)
    H1, H2 = H1[ok], H2[ok]
    std = np.maximum(np.asarray(public_std), 1e-9)
    H1, H2 = (H1 - public_mean) / std, (H2 - public_mean) / std
    H1, H2 = H1 - H1.mean(1, keepdims=True), H2 - H2.mean(1, keepdims=True)
    n1, n2 = np.linalg.norm(H1, axis=1), np.linalg.norm(H2, axis=1)
    usable = (n1 > 1e-12) & (n2 > 1e-12)
    H1, H2 = H1[usable] / n1[usable, None], H2[usable] / n2[usable, None]
    g = min(int(gallery_size), len(H1))
    if g < 1 or repeats < 1:
        return {"gallery": 0, "n_eligible": len(H1), "top1": np.nan, "top5": np.nan,
                "chance_top1": np.nan, "chance_top5": np.nan}
    top1, top5 = [], []
    for _ in range(repeats):
        idx = rng.choice(len(H1), g, replace=False)
        order = np.argsort(-(H1[idx] @ H2[idx].T), axis=1)
        top1.append(np.mean(order[:, 0] == np.arange(g)))
        top5.append(np.mean(np.any(order[:, :min(5, g)] == np.arange(g)[:, None], axis=1)))
    return {"gallery": g, "n_eligible": len(H1), "top1": float(np.mean(top1)),
            "top5": float(np.mean(top5)), "chance_top1": 1 / g, "chance_top5": min(5, g) / g}


def canary_audit_windows(r: np.ndarray, A: np.ndarray, C: float, R: float, n: int, windows: int,
                         eps: float, delta: float, trials: int, rng: np.random.Generator,
                         unit: str = "window") -> dict:
    r, A = np.asarray(r, float), np.asarray(A, float)
    if n < 2 or windows < 1 or trials < 4 or not np.isfinite(C) or C <= 0:
        raise ValueError("window canary needs n>=2, windows>=1, trials>=4 and positive finite C")
    if unit not in ("window", "person"):
        raise ValueError("unit must be 'window' or 'person'")
    c = min(C, R)
    rows = n * windows if unit == "window" else n
    D_in = np.repeat(r[None], rows, axis=0)
    D_out = D_in.copy()
    last = slice(rows - windows, rows) if unit == "window" else slice(rows - 1, rows)
    D_in[last] = r + c * A[:, 0]
    D_out[last] = r - c * A[:, 0]
    a = release(D_in, r, A, C, eps, delta, rng, draws=trials, R=R)
    b = release(D_out, r, A, C, eps, delta, rng, draws=trials, R=R)
    result = summarize_attack(a.coeffs[:, 0], b.coeffs[:, 0])
    result.update({"audit_pair": "replace_one_person_contributing_all_windows",
                   "release_unit": unit, "windows_per_person": int(windows),
                   "rows_released": int(rows), "sigma": float(a.sigma), "sensitivity": float(a.sensitivity),
                   "audit_calls_shared_release": True})
    return result
