from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smdp.accounting import noise_multiplier
from smdp.bases import make_basis
from smdp.mechanism import clip_rows, release as gaussian_release


@dataclass
class Candidate:
    name: str
    basis: str
    k: int
    A: np.ndarray | None
    C: float
    quantile: float
    seed: int = 0

    def T(self, X: np.ndarray) -> np.ndarray:
        if self.A is None:
            return np.zeros_like(X)
        Z, _ = clip_rows(X @ self.A, self.C)
        return Z @ self.A.T


def build_candidates(X_ref: np.ndarray, bases: list[str], ks: list[int], quantiles: list[float],
                     random_seeds: list[int] = (), include_public_only: bool = True) -> list[Candidate]:
    d = X_ref.shape[1]
    out = []
    if include_public_only:
        out.append(Candidate("public_only", "none", 0, None, 0.0, 0.0))
    for b in bases:
        kk = [d] if b == "full" else [k for k in ks if k <= d]
        seeds = random_seeds if b.startswith("random") else [0]
        for k in kk:
            for s in seeds:
                A = make_basis(b, d, k, public_residuals=X_ref, seed=s)
                norms = np.linalg.norm(X_ref @ A, axis=1)
                for q in quantiles:
                    C = float(np.quantile(norms, q))
                    if C > 0:
                        tag = f"{b}_k{k}_q{q:g}" + (f"_s{s}" if b.startswith("random") else "")
                        out.append(Candidate(tag, b, k, A, C, q, s))
    return out


def bias_norm(cand: Candidate, X: np.ndarray) -> float:
    return float(np.linalg.norm((X - cand.T(X)).mean(axis=0)))


def sigma_of(cand: Candidate, n: int, R: float, z: float) -> float:
    return 0.0 if cand.A is None else z * 2.0 * min(cand.C, R) / n


def score(cand: Candidate, X: np.ndarray, n: int, R: float, z: float) -> float:
    s = sigma_of(cand, n, R, z)
    return float(np.sqrt(bias_norm(cand, X) ** 2 + cand.k * s * s))


def public_tune(cands: list[Candidate], groups, n: int, R: float, z: float,
                by_basis: bool = True) -> dict[str, Candidate]:
    if isinstance(groups, np.ndarray):
        groups = [groups]
    best: dict[str, tuple[float, Candidate]] = {}
    for c in cands:
        key = (c.basis + (f"_s{c.seed}" if c.basis.startswith("random") else "")) if by_basis else "all"
        sc = float(np.mean([score(c, G, n, R, z) for G in groups]))
        if key not in best or sc < best[key][0]:
            best[key] = (sc, c)
    return {k: v[1] for k, v in best.items()}


def stress_cohorts(X: np.ndarray, meta_rows, columns: list[str], n: int, n_random: int,
                   min_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    groups = []
    for col in columns:
        if col not in meta_rows:
            continue
        vals = np.array([str(v) for v in meta_rows[col].tolist()], dtype=str)
        for v in np.unique(vals):
            m = vals == v
            if v not in ("nan", "", "None", "<NA>") and m.sum() >= min_size:
                groups.append(X[m])
    for _ in range(n_random):
        groups.append(X[rng.choice(len(X), min(n, len(X)), replace=False)])
    return groups or [X]


def exponential_select(cands: list[Candidate], X: np.ndarray, n: int, R: float, z_rel: float,
                       eps_sel: float, rng: np.random.Generator) -> tuple[Candidate, np.ndarray]:
    if not cands or n != len(X) or n < 1:
        raise ValueError("selection requires candidates and public n equal to the number of people")
    if not np.isfinite(R) or R <= 0 or not np.isfinite(eps_sel) or eps_sel <= 0:
        raise ValueError("selection requires positive finite R and eps_sel")
    if not np.isfinite(X).all() or np.any(np.linalg.norm(X, axis=1) > R * (1 + 1e-10)):
        raise ValueError("selection inputs must already be outer-clipped to public R")
    r = np.array([score(c, X, n, R, z_rel) for c in cands])
    logits = -eps_sel * n * r / (4.0 * R)
    g = rng.gumbel(size=len(cands))
    return cands[int(np.argmax(logits + g))], r


def release_candidate(cand: Candidate, X: np.ndarray, r_ref: np.ndarray, n: int, R: float,
                      z_rel: float, rng: np.random.Generator, draws: int = 1) -> np.ndarray:
    if cand.A is None:
        return np.repeat(r_ref[None, :], draws, axis=0)
    if len(X) != n:
        raise ValueError("public n must equal the number of people")
    s = sigma_of(cand, n, R, z_rel)
    out = gaussian_release(X, np.zeros(X.shape[1]), cand.A, cand.C, 1., 1e-5,
                           rng, draws=draws, R=R, sigma=s)
    return r_ref[None, :] + out.coeffs @ cand.A.T


def laplace_bound(value: np.ndarray, b: float, alpha: float, rng, upper: bool = True) -> np.ndarray:
    if not 0 < alpha < .5 or not np.isfinite(b) or b < 0:
        raise ValueError("Laplace bound requires 0 < alpha < .5 and finite nonnegative scale")
    noisy = value + rng.laplace(0.0, b, size=np.shape(value))
    off = b * np.log(1.0 / (2.0 * alpha))
    return np.maximum(0.0, noisy + off) if upper else np.maximum(0.0, noisy - off)


def certify_and_decide(Y: np.ndarray, r_ref: np.ndarray, mu_R: np.ndarray, n: int, R: float,
                       eps_cert: float, alpha: float, rng: np.random.Generator,
                       mode: str = "beat_public", tol: float | None = None,
                       W: np.ndarray | None = None) -> dict:
    Y, r_ref, mu_R = np.asarray(Y, float), np.asarray(r_ref, float), np.asarray(mu_R, float)
    if Y.ndim != 2 or Y.shape[1:] != r_ref.shape or mu_R.shape != r_ref.shape:
        raise ValueError("Y must be (draws,d), reference and clipped target must have shape (d,)")
    if not all(np.isfinite(a).all() for a in (Y, r_ref, mu_R)):
        raise ValueError("certificate inputs must be finite")
    if n < 1 or not np.isfinite(R) or R < 0 or not np.isfinite(eps_cert) or eps_cert <= 0:
        raise ValueError("certificate needs n>=1, finite R>=0 and positive eps_cert")
    if not 0 < alpha < .5:
        raise ValueError("certificate alpha must lie in (0,.5)")
    if mode == "tolerance" and (tol is None or not np.isfinite(tol) or tol < 0):
        raise ValueError("tolerance mode needs a finite nonnegative tolerance")
    if W is not None and (np.ndim(W) != 2 or np.shape(W)[1] != Y.shape[1] or not np.isfinite(W).all()):
        raise ValueError("public workload must be a finite matrix with d columns")
    Wop = 1.0 if W is None else float(np.linalg.norm(W, 2))
    proj = (lambda v: v) if W is None else (lambda v: v @ W.T)
    err_Y = np.linalg.norm(proj(Y - mu_R), axis=-1)
    if mode == "beat_public":
        e1 = e2 = eps_cert / 2.0
        b1 = 2 * R * Wop / (n * e1)
        b2 = 2 * R * Wop / (n * e2)
        U = laplace_bound(err_Y, b1, alpha, rng, upper=True)
        err_pub = np.linalg.norm(proj(r_ref - mu_R))
        Lpub = laplace_bound(np.full(len(Y), err_pub), b2, alpha, rng, upper=False)
        accept = U < Lpub
    elif mode == "tolerance":
        b1 = 2 * R * Wop / (n * eps_cert)
        U = laplace_bound(err_Y, b1, alpha, rng, upper=True)
        Lpub = np.full(len(Y), np.nan)
        accept = U <= tol
    else:
        raise ValueError(mode)
    published = np.where(accept[:, None], Y, r_ref[None, :])
    return {"published": published, "U": U, "L_pub": Lpub, "accepted": accept,
            "covered": err_Y <= U, "err_Y_vs_muR": err_Y,
            "certificate_target": "outer_clipped_empirical_mean",
            "certificate_applies_to": "candidate_before_abstention",
            "joint_false_acceptance_bound": 2 * alpha if mode == "beat_public" else alpha}


def band_workload(freqs: np.ndarray, bands) -> np.ndarray:
    W = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            row = np.zeros(len(freqs))
            row[m] = 1.0 / m.sum()
            W.append(row)
    return np.array(W)


def noise_z(eps: float, delta: float) -> float:
    return noise_multiplier(eps, delta)


def public_gate_biases(cands: list[Candidate], groups, n: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    for G in groups:
        idx = rng.choice(len(G), n, replace=len(G) < n)
        X = G[idx]
        out.append([bias_norm(c, X) for c in cands])
    return np.asarray(out, dtype=float)


def public_gate_decision(cands: list[Candidate], biases: np.ndarray, n: int, R: float, z_rel: float,
                         eps_sel: float) -> dict:
    if biases.ndim != 2 or biases.shape[1] != len(cands):
        raise ValueError("gate biases must be (groups, candidates)")
    sig = np.array([sigma_of(c, n, R, z_rel) for c in cands])
    k = np.array([c.k for c in cands], dtype=float)
    pub = np.array([c.A is None for c in cands])
    if not pub.any():
        raise ValueError("the selection candidate list must contain public_only")
    r = np.sqrt(biases ** 2 + k[None] * sig[None] ** 2)
    logits = -eps_sel * n * r / (4.0 * R)
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    expected = float(np.mean((w * r).sum(axis=1)))
    public = float(np.mean(r[:, pub].min(axis=1)))
    return {"gate_on": bool(expected >= public), "gate_expected_selection_score": expected,
            "gate_public_score": public, "gate_groups": int(len(biases))}
