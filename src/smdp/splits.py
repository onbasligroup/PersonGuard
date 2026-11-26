from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def stable_unit(person_id: str, salt: str = "smdp") -> float:
    h = hashlib.sha256(f"{salt}:{person_id}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)


def select(meta: pd.DataFrame, query: str | None) -> np.ndarray:
    if not query:
        return np.arange(len(meta))
    return np.flatnonzero(meta.eval(query).to_numpy())


def hash_split(meta: pd.DataFrame, idx: np.ndarray, fractions: dict[str, float],
               salt: str = "smdp") -> dict[str, np.ndarray]:
    u = np.array([stable_unit(str(meta.person_id.iloc[i]), salt) for i in idx])
    out, lo = {}, 0.0
    names = list(fractions)
    for j, name in enumerate(names):
        hi = 1.0 if j == len(names) - 1 else lo + fractions[name]
        out[name] = idx[(u >= lo) & (u < hi)]
        lo = hi
    return out


def assert_disjoint(meta: pd.DataFrame, *parts: np.ndarray) -> None:
    seen: set[str] = set()
    for p in parts:
        ids = set(meta.person_id.iloc[p].astype(str))
        if seen & ids:
            raise RuntimeError(f"person overlap between splits: {list(seen & ids)[:5]}")
        seen |= ids


def has_label(meta: pd.DataFrame, label: str) -> np.ndarray:
    lab = meta.get("labels", pd.Series([""] * len(meta))).fillna("").astype(str)
    return lab.str.split("|").apply(lambda s: label in s).to_numpy()


def single_label(meta: pd.DataFrame, priority: list[str]) -> pd.Series:
    lab = meta.get("labels", pd.Series([""] * len(meta))).fillna("").astype(str)

    def pick(s: str):
        L = [x for x in s.split("|") if x]
        if L == ["NORM"]:
            return "NORM"
        for c in priority:
            if c != "NORM" and c in L:
                return c
        return np.nan

    return lab.apply(pick)


def reliable_person_mask(meta: pd.DataFrame) -> np.ndarray:
    if "person_id_reliable" not in meta:
        return np.zeros(len(meta), dtype=bool)
    v = meta["person_id_reliable"].astype("string").fillna("").str.strip().str.lower()
    return v.isin(["true", "1", "1.0"]).to_numpy()
