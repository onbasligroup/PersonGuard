from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "pg")]

from smdp.accounting import BudgetLedger, gdp_mu, gdp_to_eps
from smdp.features import load_many
from smdp.mechanism import (clip_rows, error_decomposition, james_stein_toward_reference,
                            release as gaussian_release)
from smdp.metrics import curve_metrics
from smdp.splits import assert_disjoint, hash_split, reliable_person_mask, select
from reliability import (build_candidates, certify_and_decide, exponential_select,
                         noise_z, public_gate_biases, public_gate_decision, public_tune,
                         stress_cohorts)
from attacks import (canary_audit, canary_audit_windows, eps_lower_bound, eps_lower_bound_details,
                     linkage_reid, run_membership, run_window_membership)


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


def resolve_scenario(meta: pd.DataFrame, sc: dict) -> tuple[np.ndarray, np.ndarray]:
    if "pool" in sc:
        idx = select(meta, sc["pool"])
        parts = hash_split(meta, idx, {"public": sc.get("split_fraction_public", 0.3), "private": 1.0},
                           salt=f"scenario:{sc['name']}")
        pub, priv = parts["public"], parts["private"]
    else:
        pub, priv = select(meta, sc["public"]), select(meta, sc["private"])
    reliable = reliable_person_mask(meta)
    priv = priv[reliable[priv]]
    priv_ids = set(meta.person_id.iloc[priv].astype(str))
    pub = np.array([i for i in pub if str(meta.person_id.iloc[i]) not in priv_ids], dtype=int)
    assert_disjoint(meta, pub, priv)
    return pub, priv


def coeff_release(cand, X, eps, delta, R, rng, draws):
    res = gaussian_release(X, np.zeros(X.shape[1]), cand.A, cand.C, eps, delta, rng, draws, R=R)
    return res.coeffs, res.sigma, res.sensitivity


def privacy_fields(delta, sensitivity=0., sigma=0., eps_sel=0., eps_cert=0., nonprivate=False):
    if nonprivate:
        return {"accounted_eps": np.inf, "accounted_delta": 0., "privacy_status": "NONPRIVATE_REFERENCE",
                "privacy_ledger": "nonprivate evaluation baseline", "accounting_scope": "one_invocation"}
    ledger = BudgetLedger(delta)
    if eps_sel:
        ledger.add_pure("configuration_selection", eps_sel)
    if sigma > 0 and sensitivity > 0:
        ledger.add_gaussian("coefficient_release", sensitivity, sigma)
    if eps_cert:
        ledger.add_pure("accuracy_certificate", eps_cert)
    e, dd = ledger.total()
    return {"accounted_eps": e, "accounted_delta": dd, "privacy_status": "PER_INVOCATION_BOUND",
            "privacy_ledger": ledger.describe(), "accounting_scope": "one_invocation"}


def validate_config(cfg):
    fsel = cfg["selection"]["eps_frac_sel"]
    fcert = cfg["certificate"]["eps_frac_cert"]
    if not (0 < fsel < 1 and 0 < fcert < 1 and fsel + fcert < 1):
        raise ValueError("selection and certificate budget fractions must be positive and sum to < 1")
    if not 0 < cfg["certificate"]["alpha"] < .5:
        raise ValueError("certificate alpha must be in (0, .5)")
    if cfg["certificate"]["mode"] == "tolerance" and cfg["certificate"].get("tol", -1) < 0:
        raise ValueError("tolerance mode needs a nonnegative tol in L2 log10 units")


def run_utility(cfg, meta, feats, scen_ids, caps, out):
    validate_config(cfg)
    freqs = feats["freqs_PG"]
    d = len(freqs)
    bands = [tuple(b) for b in cfg["bands_hz"]]
    delta = float(cfg["delta"])
    draws, cohorts = int(cfg["draws"]), int(cfg["cohorts"])
    sel, cert = cfg["selection"], cfg["certificate"]
    for si in scen_ids:
        sc = cfg["scenarios"][si]
        pub_all, priv_all = resolve_scenario(meta, sc)
        for L in caps:
            S_all = feats[f"PG_L{L}"].astype(np.float64)
            ok = np.isfinite(S_all).all(1)
            pub, priv = pub_all[ok[pub_all]], priv_all[ok[priv_all]]
            if len(pub) < 20 or len(priv) < min(cfg["n_grid"]):
                print(f"[skip] {sc['name']} L={L}: public={len(pub)} reliable_private={len(priv)}")
                continue
            t0 = time.time()
            halves = hash_split(meta, pub, {"ref": cfg["public_split"]["ref"], "val": 1.0}, salt="pubsplit")
            P_ref, P_val, Q = S_all[halves["ref"]], S_all[halves["val"]], S_all[priv]
            r = P_ref.mean(0)
            R = max(float(np.quantile(np.linalg.norm(P_ref - r, axis=1), cfg["outer_R_quantile"])), 1e-12)
            X_ref, _ = clip_rows(P_ref - r, R)
            X_val, _ = clip_rows(P_val - r, R)
            meta_val = meta.iloc[halves["val"]].reset_index(drop=True)
            tcfg = cfg.get("tuning", {})
            zero = np.zeros(d)
            fixed = build_candidates(X_ref, cfg["bases_fixed"], cfg["k_grid"], cfg["C_quantiles"], include_public_only=False)
            dc_only = [c for c in fixed if c.basis == "dct" and c.k == 1]
            unc = build_candidates(P_ref, ["full", "dct"], [7], cfg["C_quantiles"], include_public_only=False)
            sel_c = build_candidates(X_ref, sel["bases"], sel["k_grid"], sel["C_quantiles"])
            rows, decomp = [], []
            for n in cfg["n_grid"]:
                if n > len(Q):
                    print(f"[skip-cell] {sc['name']} L={L} n={n}: only {len(Q)} reliable people")
                    continue
                cohort_idx = [np.random.default_rng(seed_of(cfg["seed"], sc["name"], L, n, c)).choice(len(Q), n, replace=False)
                              for c in range(cohorts)]
                groups = lambda values: stress_cohorts(values, meta_val, tcfg.get("columns", []), n,
                    tcfg.get("random_cohorts", 10), tcfg.get("min_size", 30),
                    np.random.default_rng(seed_of(cfg["seed"], "tune", sc["name"], L, n)))
                G_val, G_unc = groups(X_val), groups(P_val)
                gate_cfg = sel.get("gate", {})
                gate_b = (public_gate_biases(sel_c, G_val, n, np.random.default_rng(seed_of(cfg["seed"], "gate", sc["name"], L, n)))
                          if gate_cfg.get("enabled", False) else None)
                for eps in cfg["eps_grid"]:
                    z = noise_z(eps, delta)
                    e_sel, e_cert = eps * sel["eps_frac_sel"], eps * cert["eps_frac_cert"]
                    gates = {}
                    if gate_b is not None:
                        for gtag, gcert in (("private_select", False), ("private_select_cert", True)):
                            g_erel = eps - e_sel - (e_cert if gcert else 0.)
                            gates[gtag] = public_gate_decision(sel_c, gate_b, n, R, noise_z(g_erel, delta), e_sel)
                    tuned = public_tune(fixed, G_val, n, R, z)
                    t_dc = public_tune(dc_only, G_val, n, R, z, by_basis=False)["all"]
                    rnd = build_candidates(X_ref, ["random_dc"], [tuned["dct"].k], cfg["C_quantiles"],
                                           random_seeds=cfg["random_seeds"], include_public_only=False)
                    t_rnd = public_tune(rnd, G_val, n, R, z)
                    t_unc_full = public_tune([c for c in unc if c.basis == "full"], G_unc, n, np.inf, z, False)["all"]
                    t_unc_dct = public_tune([c for c in unc if c.basis == "dct"], G_unc, n, np.inf, z, False)["all"]
                    for c, ci in enumerate(cohort_idx):
                        S = Q[ci]
                        mu = S.mean(0)
                        X, outer_norms = clip_rows(S - r, R)
                        mu_R = r + X.mean(0)
                        outer_bias_db = float(10 * np.linalg.norm(mu_R - mu) / np.sqrt(d))
                        pub_rmse = curve_metrics(r, mu, freqs, bands)["rmse_db"][0]
                        n_windows = np.minimum(np.asarray(meta.iloc[priv[ci]].n_windows, dtype=float), L)

                        def append(name, E, extra):
                            m = curve_metrics(E, mu, freqs, bands)
                            mr = curve_metrics(E, mu_R, freqs, bands)
                            for j in range(len(E)):
                                row = {"scenario": sc["name"], "L": L, "n": n, "eps": eps, "delta": delta,
                                       "cohort": c, "draw": int(extra.get("draw_offset", 0)) + j, "method": name,
                                       "R": R, "target_R": R, "draws": 1, "beats_public": float(m["rmse_db"][j] < pub_rmse),
                                       "rmse_db_vs_muR": float(mr["rmse_db"][j]), "outer_clip_bias_rmse_db": outer_bias_db,
                                       "outer_clip_rate": float(np.mean(outer_norms > R)),
                                       "mean_windows_used": float(n_windows.mean()),
                                       "certificate_target": "outer_clipped_empirical_mean"}
                                for key, v in m.items():
                                    row[key] = float(v[j])
                                row.update({k: v for k, v in extra.items() if k != "draw_offset"})
                                rows.append(row)

                        append("public_only", np.repeat(r[None], draws, 0),
                               {"k": 0, "C": 0., "R": 0., "sigma": 0., **privacy_fields(delta)})
                        variants = [("workshop_full_uncentred", t_unc_full, S, zero, np.inf),
                                    ("workshop_dct7_uncentred", t_unc_dct, S, zero, np.inf),
                                    ("dc_only_centred", t_dc, X, r, R),
                                    *[(f"{b}_centred", tuned[b], X, r, R) for b in cfg["bases_fixed"] if b in tuned],
                                    *[(f"random_dc_centred_{key}", cnd, X, r, R) for key, cnd in t_rnd.items()]]
                        for name, cand, Xc, rr, RR in variants:
                            rng = np.random.default_rng(seed_of(cfg["seed"], sc["name"], L, n, eps, c, name))
                            co, s, sensitivity = coeff_release(cand, Xc, eps, delta, RR, rng, draws)
                            p = privacy_fields(delta, sensitivity, s)
                            append(name, rr[None] + co @ cand.A.T,
                                   {"k": cand.k, "C": cand.C, "R": RR, "sigma": s, **p})
                            if name == "dct_centred":
                                append("dct_centred_js", r[None] + james_stein_toward_reference(co, s) @ cand.A.T,
                                       {"k": cand.k, "C": cand.C, "R": R, "sigma": s, **p})
                                nf = r + clip_rows(X @ cand.A, cand.C)[0].mean(0) @ cand.A.T
                                append("noiseless_dct_clipped", np.repeat(nf[None], draws, 0),
                                       {"k": cand.k, "C": cand.C, "R": R, "sigma": 0.,
                                        **privacy_fields(delta, nonprivate=True)})
                                decomp.append({"scenario": sc["name"], "L": L, "n": n, "eps": eps, "cohort": c,
                                               "k": cand.k, "C": cand.C, "R": R,
                                               **error_decomposition(S, r, cand.A, cand.C, R, s)})
                        for tag, use_cert in (("private_select", False), ("private_select_cert", True)):
                            erel = eps - e_sel - (e_cert if use_cert else 0.)
                            zz = noise_z(erel, delta)
                            produced = []
                            for draw in range(draws):
                                rng = np.random.default_rng(seed_of(cfg["seed"], sc["name"], L, n, eps, c, tag, draw))
                                chosen, _ = exponential_select(sel_c, X, n, R, zz, e_sel, rng)
                                if chosen.A is None:
                                    Y, s, sensitivity = r[None].copy(), 0., 0.
                                else:
                                    co, s, sensitivity = coeff_release(chosen, X, erel, delta, R, rng, 1)
                                    Y = r[None] + co @ chosen.A.T
                                branch = privacy_fields(delta, sensitivity, s, e_sel, e_cert if use_cert else 0.)
                                total_eps = e_sel + erel + (e_cert if use_cert else 0.)
                                extra = {"k": chosen.k, "C": chosen.C, "R": R, "chosen": chosen.name,
                                         "sigma": s, "draw_offset": draw,
                                         "accounted_eps": total_eps, "accounted_delta": delta,
                                         "privacy_status": "PER_INVOCATION_BOUND",
                                         "privacy_accounting": "full_adaptive_algorithm",
                                         "privacy_ledger": (f"selection pure {e_sel:g} + gaussian release ({erel:g},{delta:g})"
                                                            + (f" + certificate pure {e_cert:g}" if use_cert else "")),
                                         "accounting_scope": "one_invocation",
                                         "branch_realized_eps": branch["accounted_eps"],
                                         "branch_realized_delta": branch["accounted_delta"]}
                                if use_cert:
                                    res = certify_and_decide(Y, r, mu_R, n, R, e_cert, cert["alpha"], rng,
                                                            mode=cert["mode"], tol=cert.get("tol"))
                                    extra.update({"accept_rate": float(res["accepted"][0]), "coverage": float(res["covered"][0]),
                                                  "U_rmse_db": float(10 * res["U"][0] / np.sqrt(d)),
                                                  "certificate_applies_to": "candidate_before_abstention",
                                                  "candidate_rmse_db_vs_muR": float(10 * res["err_Y_vs_muR"][0] / np.sqrt(d))})
                                    Y = res["published"]
                                append(tag, Y, extra)
                                produced.append((Y, extra))
                            if tag in gates:
                                g = gates[tag]
                                for draw, (Yd, ed) in enumerate(produced):
                                    if g["gate_on"]:
                                        ge = {"k": 0, "C": 0., "R": R, "chosen": "public_gate", "sigma": 0., "draw_offset": draw,
                                              "accounted_eps": 0., "accounted_delta": 0.,
                                              "privacy_status": "OUTPUT_DEPENDS_ONLY_ON_PUBLIC_DATA",
                                              "privacy_accounting": "public_gate_reference_zero_cost",
                                              "privacy_ledger": "public gate on: released output is the public reference; the benchmark still evaluates the ungated method separately",
                                              "accounting_scope": "one_invocation"}
                                        Yg = r[None].copy()
                                    else:
                                        ge, Yg = dict(ed), Yd
                                    ge.update(g)
                                    append(tag + "_gated", Yg, ge)
            tag = f"{sc['name']}_L{L}"
            pd.DataFrame(rows).to_csv(out / f"utility_{tag}.csv.gz", index=False)
            pd.DataFrame(decomp).to_csv(out / f"decomposition_{tag}.csv.gz", index=False)
            print(f"[utility] {tag}: {len(rows)} draw rows in {time.time() - t0:.1f}s (public={len(pub)}, private={len(priv)})")


def run_attacks(cfg, meta, feats, out):
    ac = cfg["attacks"]
    delta = float(cfg["delta"])
    L = int(ac.get("cap", 1))
    S_all = feats[f"PG_L{L}"].astype(np.float64)
    ok = np.isfinite(S_all).all(1)
    pool = select(meta, ac["pool"])
    pool = pool[ok[pool] & reliable_person_mask(meta)[pool]]
    if ac.get("public"):
        pub = select(meta, ac["public"])
        pub = np.setdiff1d(pub[ok[pub]], pool)
        link_part = np.array([], dtype=int)
    else:
        lf = float(ac.get("linkage", {}).get("holdout_fraction", 0.)) if ac.get("linkage") else 0.
        parts = hash_split(meta, pool, {"linkage": lf, "reference": .3 * (1 - lf), "pool": 1.0}, salt="attack")
        pub, pool, link_part = parts["reference"], parts["pool"], parts["linkage"]
    assert_disjoint(meta, pub, pool, link_part)
    if len(pub) < 10 or len(pool) < min(ac["n_grid"]) + 2:
        raise ValueError("attack pool needs reliable person IDs and enough disjoint reference/private people")
    P, Pool = S_all[pub], S_all[pool]
    r = P.mean(0)
    R = max(float(np.quantile(np.linalg.norm(P - r, axis=1), cfg["outer_R_quantile"])), 1e-12)
    Xp, _ = clip_rows(P - r, R)
    half = len(Xp) // 2
    cands = build_candidates(Xp[:half], ["dct"], cfg["k_grid"], cfg["C_quantiles"], include_public_only=False)
    rows, audit = [], []
    score_store = {}
    conf = float(ac.get("confidence", .95))
    planned_cells = max(1, len(ac["n_grid"]) * len(ac["eps_grid"]))
    cell_conf = 1 - (1 - conf) / planned_cells
    def summarize(res):
        return {k: v for k, v in res.items() if not k.startswith("scores")}
    for n in ac["n_grid"]:
        if n + 2 > len(Pool):
            continue
        rng = np.random.default_rng(seed_of(cfg["seed"], "attack", n))
        res = run_membership(Pool, P, r, None, np.inf, np.inf, n, 0., ac["trials"], rng)
        rows.append({"n": n, "mechanism": "non_private_person_mean", "eps": np.inf, "cap": L,
                     "estimand": "mean_person_window_logpsd", **summarize(res)})
        for eps in ac["eps_grid"]:
            z = noise_z(eps, delta)
            rng_t = np.random.default_rng(seed_of(cfg["seed"], "attack-tune", n, eps))
            G = [Xp[half:][rng_t.choice(len(Xp) - half, min(n, len(Xp) - half), replace=False)] for _ in range(10)]
            t = public_tune(cands, G, n, R, z)["dct"]
            sig = z * 2 * min(t.C, R) / n
            res = run_membership(Pool, P, r, t.A, t.C, R, n, sig, ac["trials"], rng)
            rows.append({"n": n, "mechanism": "person_level_dp", "eps": eps, "cap": L,
                         "k": t.k, "sigma": sig, "estimand": "clipped_person_mean_coefficients",
                         **summarize(res), "eps_lb": eps_lower_bound(res["scores_in"], res["scores_out"], delta, cell_conf),
                         "eps_lb_confidence_cell": cell_conf})
            audited = canary_audit(r, t.A, t.C, R, n, None, ac["audit_trials"], rng, eps=eps, delta=delta)
            det = eps_lower_bound_details(audited["scores_in"], audited["scores_out"], delta, cell_conf)
            lb = float(det["eps_lb"])
            score_store[f"n{n}_eps{eps:g}_in"] = audited["scores_in"].astype(np.float32)
            score_store[f"n{n}_eps{eps:g}_out"] = audited["scores_out"].astype(np.float32)
            audit.append({"n": n, "eps_theory": eps, "delta": delta, "eps_lb_canary": lb,
                          "n_thresholds": det["n_thresholds"], "n_in_evaluation": det["n_in_evaluation"],
                          "n_out_evaluation": det["n_out_evaluation"], "sigma": audited["sigma"], "k": t.k, "C": t.C, "R": R,
                          "noise_calibration": audited["noise_calibration"],
                          "scores_file": "audit_scores.npz",
                          "auc_canary": audited["auc"], "confidence_cell": cell_conf,
                          "confidence_family": conf, "planned_cells": planned_cells,
                          "neighboring_pair": "fixed_replace_one_person", "release_path": "smdp.mechanism.release"})
            W = feats.get("PG_windows")
            wcap = int(ac.get("window_cap", 2))
            if W is not None and W.shape[1] >= wcap:
                WP, WQ = W[pub, :wcap].astype(float), W[pool, :wcap].astype(float)
                WP = WP[np.isfinite(WP).all((1, 2))]
                WQ = WQ[np.isfinite(WQ).all((1, 2))]
                if len(WP) >= 5 and len(WQ) >= n + 2:
                    wref = WP.reshape(-1, WP.shape[-1])
                    wr = wref.mean(0)
                    wR = max(float(np.quantile(np.linalg.norm(wref - wr, axis=1), cfg["outer_R_quantile"])), 1e-12)
                    wx, _ = clip_rows(wref - wr, wR)
                    wcands = build_candidates(wx, ["dct"], [t.k], [.85], include_public_only=False)
                    wt = wcands[0]
                    wres = run_window_membership(WQ, WP, wr, wt.A, wt.C, wR, n, eps, delta,
                                                 ac["trials"], rng, wcap)
                    rows.append({"n": n, "mechanism": "window_level_dp_actual", "eps": eps, "cap": wcap,
                                 "k": wt.k, "sigma": z * 2 * min(wt.C, wR) / (n * wcap),
                                 "estimand": "mean_of_individually_clipped_window_coefficients",
                                 "privacy_unit": "one_window", **summarize(wres),
                                 "eps_lb": eps_lower_bound(wres["scores_in"], wres["scores_out"], delta, cell_conf)})
                    mres = run_membership(WQ.mean(1), WP.mean(1), wr, wt.A, wt.C, wR, n,
                                          z * 2 * min(wt.C, wR) / n, ac["trials"], rng)
                    rows.append({"n": n, "mechanism": "person_level_dp_window_matched", "eps": eps, "cap": wcap,
                                 "k": wt.k, "sigma": z * 2 * min(wt.C, wR) / n,
                                 "estimand": "clipped_person_window_mean_coefficients", "privacy_unit": "one_person",
                                 **summarize(mres)})
                else:
                    print(f"[skip-window-attack] n={n}, cap={wcap}: eligible public={len(WP)}, private={len(WQ)}")
            else:
                print("[skip-window-attack] PG_windows missing or cap exceeds cache; regenerate schema-2 features")
            print(f"[attack] n={n} eps={eps}: simultaneous canary eps_lb={lb:.3f}")
    pd.DataFrame(rows).to_csv(out / "attacks.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "audit.csv", index=False)
    np.savez_compressed(out / "audit_scores.npz", **score_store)
    lk = ac.get("linkage")
    if lk and "PG_half1" in feats:
        idx = select(meta, lk["pool"])
        idx = idx[reliable_person_mask(meta)[idx]]
        idx = np.intersect1d(idx, link_part) if len(link_part) else np.setdiff1d(np.setdiff1d(idx, pub), pool)
        assert_disjoint(meta, idx, pub, pool)
        H1, H2 = feats["PG_half1"][idx].astype(float), feats["PG_half2"][idx].astype(float)
        lrows = []
        for g in lk["gallery"]:
            if g > len(idx):
                print(f"[skip-linkage] gallery {g} exceeds {len(idx)} holdout people")
                continue
            rr = linkage_reid(H1, H2, g, np.random.default_rng(seed_of("link", g)), P.mean(0), P.std(0))
            rr["gallery_disjoint_from_attack_and_reference"] = True
            rr["linkage_holdout_people"] = int(len(idx))
            lrows.append(rr)
        cols = ["gallery", "n_eligible", "top1", "top5", "chance_top1", "chance_top5",
                "gallery_disjoint_from_attack_and_reference", "linkage_holdout_people", "status"]
        done = {int(x["gallery"]) for x in lrows}
        for g in lk["gallery"]:
            if int(g) not in done:
                lrows.append({"gallery": int(g), "n_eligible": int(len(idx)), "linkage_holdout_people": int(len(idx)),
                              "gallery_disjoint_from_attack_and_reference": True, "status": "skipped_gallery_exceeds_holdout"})
        for x in lrows:
            x.setdefault("status", "ok")
        pd.DataFrame(lrows).reindex(columns=cols).to_csv(out / "linkage.csv", index=False)
    (out / "attack_protocol.json").write_text(json.dumps({"cohort_size_public": True,
        "privacy_unit": "person_except_explicit_window_baseline", "same_population_reference": True,
        "canary_family_confidence": conf, "planned_canary_cells": planned_cells,
        "evaluation_outputs_contain_unnoised_diagnostics": True,
        "linkage_holdout_fraction": float(ac.get("linkage", {}).get("holdout_fraction", 0.)) if ac.get("linkage") else None,
        "membership_trials_per_world": int(ac["trials"]), "canary_scores_saved": True}, indent=2))


def run_window_audit(cfg, meta, feats, out):
    wa = cfg["window_audit"]
    delta = float(cfg["delta"])
    W = feats.get("PG_windows")
    if W is None:
        raise ValueError("PG_windows missing: rebuild schema-2 features")
    idx = select(meta, wa["pool"])
    idx = idx[reliable_person_mask(meta)[idx]]
    ref = hash_split(meta, idx, {"reference": float(wa.get("reference_fraction", .5)), "rest": 1.0}, salt="window-audit")["reference"]
    WP = W[ref].astype(float)
    win = WP.reshape(-1, WP.shape[-1])
    win = win[np.isfinite(win).all(1)]
    if len(win) < 50:
        raise ValueError(f"window audit needs public windows; found {len(win)}")
    r = win.mean(0)
    R = max(float(np.quantile(np.linalg.norm(win - r, axis=1), cfg["outer_R_quantile"])), 1e-12)
    x, _ = clip_rows(win - r, R)
    cand = build_candidates(x, ["dct"], [int(wa.get("k", 5))], [float(wa.get("C_quantile", .85))],
                            include_public_only=False)[0]
    max_w = int(W.shape[1])
    wins = [int(w) for w in wa["windows"] if int(w) <= max_w]
    conf = float(wa.get("confidence", .95))
    cells = max(1, len(wins) * len(wa["n_grid"]) * len(wa["eps_grid"]) * 2)
    cell_conf = 1 - (1 - conf) / cells
    rows, store = [], {}
    for n in wa["n_grid"]:
        for eps in wa["eps_grid"]:
            for w in wins:
                for unit in ("window", "person"):
                    rng = np.random.default_rng(seed_of(cfg["seed"], "window-audit", n, eps, w, unit))
                    res = canary_audit_windows(r, cand.A, cand.C, R, int(n), w, float(eps), delta,
                                               int(wa["trials"]), rng, unit=unit)
                    det = eps_lower_bound_details(res["scores_in"], res["scores_out"], delta, cell_conf)
                    store[f"n{n}_eps{eps:g}_w{w}_{unit}_in"] = res["scores_in"].astype(np.float32)
                    store[f"n{n}_eps{eps:g}_w{w}_{unit}_out"] = res["scores_out"].astype(np.float32)
                    rows.append({"n": n, "eps_theory": eps, "delta": delta, "windows_per_person": w,
                                 "release_unit": unit, "eps_lb_canary": float(det["eps_lb"]),
                                 "auc_canary": res["auc"], "sigma": res["sigma"], "sensitivity": res["sensitivity"],
                                 "rows_released": res["rows_released"], "k": cand.k, "C": cand.C, "R": R,
                                 "person_guarantee_eps_at_delta": (gdp_to_eps(w * gdp_mu(float(eps), delta), delta)
                                                                   if unit == "window" else float(eps)),
                                 "person_guarantee_delta": delta,
                                 "person_guarantee_gdp_mu": (w if unit == "window" else 1) * gdp_mu(float(eps), delta),
                                 "person_guarantee_basis": ("gaussian_group_sensitivity_W_times_window" if unit == "window"
                                                            else "person_level_calibration"),
                                 "confidence_cell": cell_conf, "confidence_family": conf, "planned_cells": cells,
                                 "n_thresholds": det["n_thresholds"], "n_out_evaluation": det["n_out_evaluation"],
                                 "public_windows": int(len(win)), "max_windows_available": max_w,
                                 "audit_pair": res["audit_pair"]})
                    print(f"[window-audit] n={n} eps={eps} W={w} unit={unit}: eps_lb={det['eps_lb']:.3f}")
    pd.DataFrame(rows).to_csv(out / "window_audit.csv", index=False)
    np.savez_compressed(out / "window_audit_scores.npz", **store)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/pg.yaml"))
    ap.add_argument("--part", choices=["all", "utility", "attacks", "window_audit"], default="all")
    ap.add_argument("--scenario", type=int)
    ap.add_argument("--cap", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    validate_config(cfg)
    out = Path(cfg["out_dir"])
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    froot = Path(cfg["features_root"])
    keys = {f"PG_L{L}" for L in cfg["caps"]}
    if a.cap is not None:
        keys.add(f"PG_L{int(a.cap)}")
    if a.part in ("all", "attacks") and "attacks" in cfg:
        keys |= {f"PG_L{int(cfg['attacks'].get('cap', 1))}", "PG_windows", "PG_half1", "PG_half2"}
    if a.part in ("all", "window_audit") and "window_audit" in cfg:
        keys |= {"PG_windows"}
    meta, feats = load_many(froot if froot.is_absolute() else ROOT / froot, cfg["dataset_dirs"],
                            include_keys=sorted(keys))
    meta["institution"] = np.where(meta["dataset"] == "cinc2021", meta["site"], meta["dataset"])
    if "age" in meta:
        meta["age_band"] = pd.cut(pd.to_numeric(meta["age"], errors="coerce"), [0, 40, 55, 65, 75, 200],
                                  labels=["<40", "40-54", "55-64", "65-74", "75+"]).astype(str)
    print(f"loaded {len(meta)} people; reliable IDs={reliable_person_mask(meta).sum()}")
    print(f"analytic Gaussian: z(1,1e-5)={noise_z(1.,1e-5):.3f}")
    if a.part in ("all", "utility"):
        run_utility(cfg, meta, feats, [a.scenario] if a.scenario is not None else list(range(len(cfg["scenarios"]))),
                    [a.cap] if a.cap is not None else cfg["caps"], out)
    if a.part in ("all", "attacks") and "attacks" in cfg:
        run_attacks(cfg, meta, feats, out)
    if a.part in ("all", "window_audit") and "window_audit" in cfg:
        run_window_audit(cfg, meta, feats, out)


if __name__ == "__main__":
    main()
