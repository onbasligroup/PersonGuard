from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, stat=np.median,
                         B: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    groups = [values[clusters == c] for c in uniq]
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = stat(np.concatenate([groups[i] for i in pick]))
    a = (1 - level) / 2
    return float(stat(values)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    adj, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def paired_tests(df: pd.DataFrame, metric: str, reference: str, pair_cols: list[str],
                 method_col: str = "method") -> pd.DataFrame:
    piv = df.pivot_table(index=pair_cols, columns=method_col, values=metric, aggfunc="mean")
    rows, p = [], {}
    for m in piv.columns:
        if m == reference:
            continue
        sub = piv[[m, reference]].dropna()
        if len(sub) < 5:
            continue
        diff = sub[m] - sub[reference]
        try:
            pv = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        except ValueError:
            pv = 1.0
        p[m] = pv
        rows.append({"method": m, "reference": reference, "metric": metric, "n_pairs": len(sub),
                     "median_diff": float(diff.median()),
                     "win_rate": float(np.mean(diff < 0)), "p": pv})
    out = pd.DataFrame(rows)
    if len(out):
        adj = holm(p)
        out["p_holm"] = out["method"].map(adj)
    return out
