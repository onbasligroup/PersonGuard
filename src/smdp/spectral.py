from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly, welch, get_window

LN10 = np.log(10.0)

LEADS12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEADS8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass(frozen=True)
class SpectralConfig:
    fs: float = 250.0
    window_sec: float = 8.0
    amp_bound_mv: float = 5.0
    nperseg: int = 512
    noverlap: int = 256
    fmin: float = 0.5
    fmax: float = 60.0
    floor: float = 1e-12
    csd_nperseg: int = 256
    csd_noverlap: int = 128
    csd_fmax: float = 40.0
    csd_load: float = 1e-3
    max_flat_frac: float = 0.5

    @property
    def win_len(self) -> int:
        return int(round(self.fs * self.window_sec))


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return x
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=-1)


def normalise(x: np.ndarray, cfg: SpectralConfig) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), np.clip(x / cfg.amp_bound_mv, -1.0, 1.0), np.nan)


def window_ok(w: np.ndarray, cfg: SpectralConfig) -> bool:
    w = np.atleast_2d(w)
    if not np.isfinite(w).all():
        return False
    for ch in w:
        flat = np.mean(np.abs(np.diff(ch)) < 1e-7)
        sat = np.mean(np.abs(ch) >= 0.999)
        if flat > cfg.max_flat_frac or sat > cfg.max_flat_frac or ch.std() < 1e-5:
            return False
    return True


def split_windows(x: np.ndarray, cfg: SpectralConfig, max_windows: int | None = None,
                  center_if_short: bool = True) -> list[np.ndarray]:
    n = x.shape[-1]
    L = cfg.win_len
    if n < L:
        return []
    count = n // L
    if count == 1 and center_if_short:
        a = (n - L) // 2
        starts = [a]
    else:
        starts = [i * L for i in range(count)]
    out = []
    for s in starts:
        w = x[..., s:s + L]
        if window_ok(w, cfg):
            out.append(w)
        if max_windows is not None and len(out) >= max_windows:
            break
    return out


def freq_grid(cfg: SpectralConfig, nperseg: int | None = None, fmax: float | None = None) -> np.ndarray:
    nperseg = nperseg or cfg.nperseg
    fmax = cfg.fmax if fmax is None else fmax
    f = np.fft.rfftfreq(nperseg, d=1.0 / cfg.fs)
    return f[(f >= cfg.fmin) & (f <= fmax)]


def log_psd(w: np.ndarray, cfg: SpectralConfig) -> np.ndarray:
    f, p = welch(w, fs=cfg.fs, window="hann", nperseg=cfg.nperseg, noverlap=cfg.noverlap,
                 detrend="constant", scaling="density")
    m = (f >= cfg.fmin) & (f <= cfg.fmax)
    return np.log10(np.maximum(p[m], cfg.floor))


def csd_matrix(w: np.ndarray, cfg: SpectralConfig) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    n, step = cfg.csd_nperseg, cfg.csd_nperseg - cfg.csd_noverlap
    win = get_window("hann", n)
    scale = 1.0 / (cfg.fs * np.sum(win**2))
    starts = range(0, w.shape[1] - n + 1, step)
    segs = np.stack([w[:, s:s + n] for s in starts], axis=0)
    segs = segs - segs.mean(axis=-1, keepdims=True)
    V = np.fft.rfft(segs * win, axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / cfg.fs)
    m = (f >= cfg.fmin) & (f <= cfg.csd_fmax)
    V = V[:, :, m]
    S = np.einsum("slf,smf->flm", V, V.conj()) / V.shape[0] * scale
    S *= 2.0
    return S


def hermitian_logm(S: np.ndarray, load: float = 1e-3, floor: float = 1e-30) -> np.ndarray:
    L = S.shape[-1]
    tr = np.real(np.trace(S, axis1=-2, axis2=-1))[..., None, None]
    S = S + load * (tr / L) * np.eye(L)
    lam, U = np.linalg.eigh(S)
    lam = np.log10(np.maximum(lam, floor))
    return (U * lam[..., None, :]) @ np.swapaxes(U.conj(), -1, -2)


def hermitian_expm(H: np.ndarray) -> np.ndarray:
    H = 0.5 * (H + np.swapaxes(H.conj(), -1, -2))
    lam, U = np.linalg.eigh(H)
    return (U * (10.0 ** lam)[..., None, :]) @ np.swapaxes(U.conj(), -1, -2)


def herm_to_vec(H: np.ndarray) -> np.ndarray:
    L = H.shape[-1]
    iu = np.triu_indices(L, 1)
    d = np.real(np.diagonal(H, axis1=-2, axis2=-1))
    up = H[..., iu[0], iu[1]]
    return np.concatenate([d, np.sqrt(2) * up.real, np.sqrt(2) * up.imag], axis=-1)


def vec_to_herm(v: np.ndarray, L: int) -> np.ndarray:
    iu = np.triu_indices(L, 1)
    p = len(iu[0])
    d, re, im = v[..., :L], v[..., L:L + p], v[..., L + p:L + 2 * p]
    H = np.zeros(v.shape[:-1] + (L, L), dtype=np.complex128)
    idx = np.arange(L)
    H[..., idx, idx] = d
    up = (re + 1j * im) / np.sqrt(2)
    H[..., iu[0], iu[1]] = up
    H[..., iu[1], iu[0]] = up.conj()
    return H


def diag_coords(L: int) -> np.ndarray:
    return np.arange(L)


def limb_transform() -> np.ndarray:
    T = np.zeros((12, 8))
    I, II = 0, 1
    T[0, I] = 1
    T[1, II] = 1
    T[2, II], T[2, I] = 1, -1
    T[3, I], T[3, II] = -0.5, -0.5
    T[4, I], T[4, II] = 1, -0.5
    T[5, II], T[5, I] = 1, -0.5
    for j in range(6):
        T[6 + j, 2 + j] = 1
    return T


def coherence(S: np.ndarray) -> np.ndarray:
    d = np.real(np.diagonal(S, axis1=-2, axis2=-1))
    return np.abs(S) ** 2 / (d[..., :, None] * d[..., None, :])
