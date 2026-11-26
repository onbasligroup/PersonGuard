from __future__ import annotations

import json
import hashlib
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import ITERATORS, Person, load_record, pick_single_lead
from .spectral import (LEADS8, SpectralConfig, csd_matrix, freq_grid, herm_to_vec,
                       hermitian_logm, log_psd, normalise, resample_to, split_windows)
from .wfdbio import select_leads

FEATURE_SCHEMA = 2
CAPS = (1, 4, 16, 64)


def person_features(p: Person, cfg: SpectralConfig, caps=CAPS, csd_cap: int = 16,
                    want_csd: bool = True, retain_windows: bool = True) -> dict | None:
    max_w = max(caps)
    single, single_rec = [], []
    csds = []
    read_errors = []
    per_rec_cap = max(1, math.ceil(max_w / max(1, len(p.records))))
    for ri, spec in enumerate(p.records):
        try:
            rec = load_record(spec)
        except Exception as exc:
            read_errors.append({"record": spec.path, "error": str(exc)})
            continue
        try:
            x = pick_single_lead(rec, p.single_lead)
            x = normalise(resample_to(x, rec.fs, cfg.fs), cfg)
            for w in split_windows(x, cfg, max_windows=per_rec_cap):
                if len(single) < max_w:
                    single.append(log_psd(w, cfg))
                    single_rec.append(ri)
        except KeyError:
            pass
        if want_csd and len(csds) < csd_cap:
            try:
                r8 = select_leads(rec, LEADS8)
                x8 = normalise(resample_to(r8.signal, rec.fs, cfg.fs), cfg)
                for w in split_windows(x8, cfg, max_windows=csd_cap - len(csds)):
                    csds.append(csd_matrix(w, cfg))
            except KeyError:
                pass
    if not single and not csds:
        return {"error": p.person_id, "trace": "No usable windows", "record_errors": read_errors,
                "qc_excluded": not read_errors}
    out = {"person_id": p.person_id, "dataset": p.dataset, "site": p.site,
           "person_id_reliable": p.person_id_reliable, "n_windows": len(single),
           "n_records_used": len(set(single_rec)), "n_csd_windows": len(csds),
           "n_record_read_errors": len(read_errors), **p.meta}
    arrays = {}
    if single:
        S = np.stack(single)
        if retain_windows:
            padded = np.full((max_w, S.shape[1]), np.nan, dtype=np.float32)
            padded[:len(S)] = S
            arrays["PG_windows"] = padded
            arrays["PG_n_windows"] = np.asarray(len(S), dtype=np.int32)
        for L in caps:
            arrays[f"PG_L{L}"] = S[:L].mean(0)
        recs = np.array(single_rec)
        if len(set(single_rec)) >= 2:
            cut = sorted(set(single_rec))[len(set(single_rec)) // 2]
            h1, h2 = S[recs < cut], S[recs >= cut]
        elif len(S) >= 2:
            h1, h2 = S[: len(S) // 2], S[len(S) // 2:]
        else:
            h1 = h2 = None
        if h1 is not None:
            arrays["PG_half1"], arrays["PG_half2"] = h1.mean(0), h2.mean(0)
    if csds:
        Sm = np.mean(np.stack(csds), axis=0)
        arrays["B_logcsd"] = herm_to_vec(hermitian_logm(Sm, load=cfg.csd_load)).astype(np.float32)
        diag = np.real(np.diagonal(Sm, axis1=-2, axis2=-1))
        loaded_diag = diag + cfg.csd_load * diag.mean(-1, keepdims=True)
        arrays["B_logpsd8"] = np.log10(np.maximum(loaded_diag, 1e-30)).astype(np.float32)
    return {"meta": out, "arrays": arrays}


def _work(args):
    p, cfg_dict, caps, csd_cap, want_csd, retain_windows = args
    try:
        return person_features(p, SpectralConfig(**cfg_dict), caps, csd_cap, want_csd, retain_windows)
    except Exception:
        return {"error": p.person_id, "trace": traceback.format_exc(limit=2)}


def run_shard(dataset: str, root: str, out_dir: str, shard: int = 0, num_shards: int = 1,
              workers: int | None = None, cfg: SpectralConfig | None = None, caps=CAPS,
              csd_cap: int = 16, want_csd: bool = True, limit: int | None = None,
              iterator_kwargs: dict | None = None, name: str | None = None,
              tag: str | None = None, retain_windows: bool = True) -> Path:
    cfg = cfg or SpectralConfig()
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("shard must be between zero and num_shards - 1")
    def digest_file(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
        return h.hexdigest()
    input_files = {}
    for fname in ("manifest.csv", "record_list.csv", "ptbxl_database.csv", "scp_statements.csv", "SHA256SUMS.txt"):
        path = Path(root) / fname
        if path.is_file(): input_files[fname] = digest_file(path)
    for key, val in (iterator_kwargs or {}).items():
        if key.endswith("_csv") and val:
            input_files[key] = digest_file(val)
    signature = {"schema": FEATURE_SCHEMA, "dataset": dataset, "root": str(Path(root).resolve()),
                 "spectral": asdict(cfg), "caps": list(caps), "csd_cap": csd_cap,
                 "want_csd": want_csd, "retain_windows": retain_windows, "input_files": input_files,
                 "num_shards": num_shards, "shard": shard,
                 "limit": limit, "iterator_kwargs": iterator_kwargs or {}, "tag": tag}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    out = Path(out_dir) / (name or dataset)
    out.mkdir(parents=True, exist_ok=True)
    target = out / (f"shard_{tag}.npz" if tag else f"shard_{shard:05d}_of_{num_shards:05d}.npz")
    if target.exists():
        with np.load(target, allow_pickle=False) as old:
            if "fingerprint" not in old or str(old["fingerprint"]) != fingerprint:
                raise ValueError(f"Stale or incompatible shard {target}; use a fresh output folder")
        summary_path = target.with_suffix(".summary.json")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("failed_persons", 0) or summary.get("record_read_errors", 0):
                raise ValueError(f"Cached shard {target} contains read failures; repair inputs and remove this shard to retry")
        print(f"[skip] compatible shard {target}")
        return target
    persons = []
    for i, p in enumerate(ITERATORS[dataset](root, **(iterator_kwargs or {}))):
        if limit is not None and i >= limit:
            break
        if i % num_shards == shard:
            persons.append(p)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    jobs = [(p, asdict(cfg), caps, csd_cap, want_csd, retain_windows) for p in persons]
    metas, arrays, errors = [], {}, []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_work, j) for j in jobs]
        for k, f in enumerate(as_completed(futs)):
            res = f.result()
            if res is None:
                continue
            if "error" in res:
                errors.append(res)
                continue
            metas.append(res["meta"])
            for key, v in res["arrays"].items():
                arrays.setdefault(key, {})[res["meta"]["person_id"]] = v
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(jobs)} persons")
    metas.sort(key=lambda m: str(m["person_id"]))
    ids = [m["person_id"] for m in metas]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate person IDs in one shard")
    packed = {}
    for key, d in arrays.items():
        sample = next(iter(d.values()))
        A = np.full((len(ids),) + sample.shape, np.nan, dtype=np.float32)
        for i, pid in enumerate(ids):
            if pid in d:
                A[i] = d[pid]
        packed[key] = A
    tmp = target.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, person_id=np.array(ids), meta=json.dumps(metas, default=str),
                        fingerprint=fingerprint, feature_schema=FEATURE_SCHEMA,
                        extraction=json.dumps(signature, sort_keys=True), **packed)
    os.replace(tmp, target)
    (out / f"errors_{tag or f'{shard:05d}'}.json").write_text(json.dumps(errors, indent=1))
    (target.with_suffix(".summary.json")).write_text(json.dumps(
        {"schema": FEATURE_SCHEMA, "submitted": len(jobs), "usable": len(ids),
         "qc_excluded_persons": sum(bool(e.get("qc_excluded")) for e in errors),
         "failed_persons": sum(not e.get("qc_excluded") for e in errors),
         "record_read_errors": sum(m["n_record_read_errors"] for m in metas)
                               + sum(len(e.get("record_errors", [])) for e in errors)}, indent=2))
    print(f"[done] {target}: {len(ids)} persons, {len(errors)} errors")
    return target


def merge_shards(dataset_dir: str | Path, cfg: SpectralConfig | None = None, allow_read_errors: bool = False) -> Path:
    cfg = cfg or SpectralConfig()
    d = Path(dataset_dir)
    shards = sorted(p for p in d.glob("shard_*.npz") if not p.name.endswith(".tmp.npz"))
    if not shards:
        raise FileNotFoundError(f"no finished shards in {d}")
    failures = {"qc_excluded_persons": 0, "failed_persons": 0, "record_read_errors": 0, "shards_without_summary": 0}
    for s in shards:
        sp = s.with_suffix(".summary.json")
        if not sp.exists():
            failures["shards_without_summary"] += 1
            continue
        sm = json.loads(sp.read_text())
        for key in ("qc_excluded_persons", "failed_persons", "record_read_errors"):
            failures[key] += int(sm.get(key, 0))
    if (failures["failed_persons"] or failures["record_read_errors"]) and not allow_read_errors:
        raise ValueError(f"Unresolved read failures in {d}: {failures}. Inspect errors_*.json; "
                         "repair the inputs, or merge with allow_read_errors=True and report them as exclusions")
    metas, loaded = [], []
    extraction = None
    seen_shards = set()
    tagged = False
    for s in shards:
        z = np.load(s, allow_pickle=False)
        if "feature_schema" not in z or int(z["feature_schema"]) != FEATURE_SCHEMA:
            raise ValueError(f"Old feature schema in {s}; regenerate into a fresh output folder")
        sig = json.loads(str(z["extraction"]))
        seen_shards.add(sig["shard"])
        tagged = tagged or sig["tag"] is not None
        common = {k: v for k, v in sig.items() if k not in ("root", "iterator_kwargs", "shard", "tag", "limit")}
        common["iterator_kwargs"] = {k: v for k, v in sig.get("iterator_kwargs", {}).items() if k != "top_dirs"}
        if extraction is None:
            extraction = common
        elif extraction != common:
            raise ValueError("Cannot merge shards with different feature settings")
        m = json.loads(str(z["meta"]))
        metas += m
        loaded.append((len(m), {k: z[k] for k in z.files if k.startswith(("PG_", "B_"))}))
    if extraction and not tagged and seen_shards != set(range(extraction["num_shards"])):
        raise ValueError("Incomplete shard set; finish every shard before merging")
    meta = pd.DataFrame(metas)
    if meta.empty:
        raise ValueError(f"No usable persons in {d}; inspect shard summary/error files")
    if meta.person_id.duplicated().any():
        raise ValueError("Duplicate persons across shards; do not mix overlapping extractions")
    cfg = SpectralConfig(**extraction["spectral"])
    shapes = {}
    for _, arrs in loaded:
        for k, v in arrs.items():
            shapes.setdefault(k, v.shape[1:])
    arrays = {}
    for k, shp in shapes.items():
        arrays[k] = np.concatenate([arrs.get(k, np.full((n,) + shp, np.nan, np.float32))
                                    for n, arrs in loaded])
    order = np.argsort(meta.person_id.astype(str).to_numpy(), kind="stable")
    meta = meta.iloc[order].reset_index(drop=True)
    arrays = {k: v[order] for k, v in arrays.items()}
    n = len(meta)
    meta.to_csv(d / "meta.csv", index=False)
    np.savez(d / "features.npz", freqs_PG=freq_grid(cfg), freqs_B=freq_grid(cfg, cfg.csd_nperseg, cfg.csd_fmax),
             feature_schema=FEATURE_SCHEMA, **arrays)
    (d / "feature_manifest.json").write_text(json.dumps({**extraction, "merge_exclusions": failures,
                                                          "read_errors_allowed": bool(allow_read_errors)}, indent=2))
    print(f"[merged] {d}: {n} persons, arrays={list(arrays)}, exclusions={failures}")
    return d / "features.npz"


def load_features(dataset_dir: str | Path, include_keys: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    d = Path(dataset_dir)
    meta = pd.read_csv(d / "meta.csv", dtype={"person_id": str})
    with np.load(d / "features.npz", allow_pickle=False) as z:
        if "feature_schema" not in z or int(z["feature_schema"]) != FEATURE_SCHEMA:
            raise ValueError(f"Incompatible old features in {d}; regenerate with this repository")
        arrays = {k: z[k] for k in z.files if k != "feature_schema"
                  and (include_keys is None or k in include_keys or k.startswith("freqs_"))}
    if meta.person_id.duplicated().any():
        raise ValueError(f"Duplicate person IDs in {d}")
    if any(len(v) != len(meta) for k, v in arrays.items() if k.startswith(("PG_", "B_"))):
        raise ValueError(f"Metadata and feature row counts differ in {d}")
    return meta, arrays


def load_many(features_root: str | Path, names: list[str], include_keys: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    metas, arrs = [], []
    for nm in names:
        d = Path(features_root) / nm
        if not (d / "features.npz").exists():
            print(f"[warn] {d} has no features.npz -- skipped")
            continue
        m, a = load_features(d, include_keys=include_keys)
        if arrs and any(not np.array_equal(a[k], arrs[0][k]) for k in ("freqs_PG", "freqs_B")):
            raise ValueError("Datasets have different frequency grids")
        metas.append(m)
        arrs.append(a)
    if not metas:
        raise FileNotFoundError("no feature folders found")
    meta = pd.concat(metas, ignore_index=True)
    if meta.person_id.duplicated().any():
        raise ValueError("Person IDs overlap across loaded feature folders")
    out = {"freqs_PG": arrs[0]["freqs_PG"], "freqs_B": arrs[0]["freqs_B"]}
    keys = {k for a in arrs for k in a if k.startswith(("PG_", "B_"))}
    for k in keys:
        shape = next(a[k].shape[1:] for a in arrs if k in a)
        out[k] = np.concatenate([a[k] if k in a else np.full((len(m),) + shape, np.nan, np.float32)
                                 for m, a in zip(metas, arrs)])
    return meta, out
