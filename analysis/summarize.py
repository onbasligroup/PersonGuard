from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from smdp.stats import cluster_bootstrap_ci, paired_tests


MAIN_METHODS = {
    "public_only": ("Public only", "#555555"),
    "dct_centred": ("DCT", "#0072B2"),
    "pca_centred": ("PCA", "#E69F00"),
    "bspline_centred": ("B-spline", "#009E73"),
    "private_select": ("Private selection", "#CC79A7"),
    "private_select_cert": ("Selection + certificate", "#D55E00"),
    "private_select_gated": ("Gated selection", "#56B4E9"),
    "private_select_cert_gated": ("Gated selection + certificate", "#F0E442"),
}
ORACLE_METHODS = {
    "noiseless_dct_clipped": ("Noiseless clipped DCT (nonprivate)", "#882255"),
    "nonprivate_dct_projection": ("Noiseless clipped DCT (nonprivate, legacy name)", "#666633"),
}
BASELINE_METHODS = {
    "public_only": MAIN_METHODS["public_only"],
    "dct_centred": MAIN_METHODS["dct_centred"],
    "workshop_dct7_uncentred": ("Workshop DCT7 (uncentred)", "#D55E00"),
}
SCENARIO_LABELS = {
    "in_dist_ptbxl": "No shift: PTB-XL",
    "institution_china_to_ptbxl": "Institution: China to PTB-XL",
    "device_clinical_to_wearable": "Device: clinical to wearable",
    "age_shift_ptbxl": "Age shift: PTB-XL",
    "prevalence_shift_ptbxl": "Prevalence shift: PTB-XL",
}


def summarize(df, keys, metric, cluster):
    rows=[]
    for vals,g in df.groupby(keys,dropna=False):
        valid=np.isfinite(pd.to_numeric(g[metric],errors="coerce"))
        g=g.loc[valid]
        if g.empty: continue
        med,lo,hi=cluster_bootstrap_ci(g[metric].to_numpy(),g[cluster].to_numpy(),B=1000)
        vals=vals if isinstance(vals,tuple) else (vals,)
        rows.append(dict(zip(keys,vals),median=med,ci_lo=lo,ci_hi=hi,n_rows=len(g),n_clusters=g[cluster].nunique()))
    return pd.DataFrame(rows)


def select_plot_slices(df, plot_n=1000, plot_L=None):
    if plot_n <= 0 or (plot_L is not None and plot_L <= 0):
        raise ValueError("Plot cohort size and window cap must be positive")
    groups = df.groupby("scenario", sort=True, dropna=False) if "scenario" in df else [("All available data", df)]
    selected = []
    for scenario, group in groups:
        row = {"scenario": scenario, "requested_n": plot_n, "requested_L": plot_L}
        for key, requested in [("L", plot_L), ("n", plot_n)]:
            if key not in group:
                continue
            values = np.sort(group[key].dropna().unique())
            if not len(values):
                raise ValueError(f"No finite {key} values for scenario {scenario}")
            if key == "L":
                chosen = values[-1] if requested is None else requested
                if chosen not in values:
                    raise ValueError(f"Requested L={chosen} absent in scenario {scenario}; available: {values.tolist()}")
            else:
                chosen = values[np.argmin(np.abs(values - requested))]
            row[key] = chosen
            group = group[group[key] == chosen]
        deltas = sorted(group.delta.dropna().unique()) if "delta" in group else []
        row["delta"] = ", ".join(f"{value:g}" for value in deltas) if deltas else "unavailable"
        row["n_fallback"] = bool("n" in row and row["n"] != plot_n)
        selected.append(row)
    return pd.DataFrame(selected)


def _slice_rows(frame, record):
    chosen = frame
    for key in ["scenario", "L", "n"]:
        if key in frame and key in record:
            chosen = chosen[chosen[key] == record[key]]
    return chosen


def _target_label(metric):
    if metric == "rmse_db":
        return "raw empirical mean mu", "RMSE (dB)"
    if metric == "rmse_db_vs_muR":
        return "outer-clipped empirical mean mu_R", "RMSE (dB)"
    return f"benchmark target for {metric}", metric


def _utility_figure(summary, slices, metric, view="main"):
    cols = min(3, len(slices))
    rows = int(np.ceil(len(slices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.5 * rows + 1.55), squeeze=False)
    if view == "diagnostic":
        methods = {"dct_centred": MAIN_METHODS["dct_centred"], **ORACLE_METHODS}
    elif view == "baseline":
        methods = BASELINE_METHODS
    else:
        methods = MAIN_METHODS
    handles = {}
    target, ylabel = _target_label(metric)
    for ax, record in zip(axes.flat, slices.to_dict(orient="records")):
        panel = _slice_rows(summary, record)
        for method, (label, color) in methods.items():
            group = panel[panel.method == method].sort_values("eps")
            if group.empty:
                continue
            linestyle = "--" if method in ORACLE_METHODS else "-"
            line, = ax.plot(group.eps, group["median"], label=label, color=color,
                            linestyle=linestyle, marker="o", markersize=3.5, linewidth=1.6)
            ax.fill_between(group.eps.to_numpy(), group.ci_lo.to_numpy(), group.ci_hi.to_numpy(),
                            color=color, alpha=0.12)
            handles.setdefault(method, line)
        scenario = SCENARIO_LABELS.get(record["scenario"], str(record["scenario"]).replace("_", " "))
        info = [f"{key}={record[key]:g}" for key in ["n", "L"] if key in record]
        info.append(f"delta={record['delta']}")
        if record["n_fallback"]:
            info.append(f"nearest to n={record['requested_n']:g}")
        ax.set_title(scenario + "\n" + ", ".join(info), fontsize=10, pad=9)
        ax.set(xlabel="epsilon", ylabel=ylabel, xscale="log")
        eps = np.sort(panel.eps.unique())
        if len(eps):
            ax.set_xticks(eps, labels=[f"{value:g}" for value in eps])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.2)
    for ax in list(axes.flat)[len(slices):]:
        ax.set_visible(False)
    headings = {"main": "PersonGuard utility", "diagnostic": "Evaluation diagnostic: private DCT and nonprivate noise floor",
                "baseline": "Baseline comparison: centered DCT and workshop DCT7"}
    heading = headings[view]
    fig.suptitle(f"{heading}\nTarget: {target}", fontsize=14, y=0.99)
    if handles:
        lines = list(handles.values())
        fig.legend(lines, [line.get_label() for line in lines], loc="lower center",
                   bbox_to_anchor=(0.5, 0.10 if rows == 1 else 0.065),
                   ncol=min(3, len(lines)), fontsize=9, frameon=False)
    caption = "Medians with 95% whole-cohort bootstrap intervals (1,000 resamples, seed 0).\n"
    caption += "Experimental intervals describe benchmark variability; they are not release certificates."
    if view == "baseline":
        caption += "\nDCT rank and radius can differ across methods; this comparison does not isolate centering."
    fig.text(0.5, 0.012, caption,
             ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.30 if rows == 1 else 0.23,
                        top=0.74 if rows == 1 else 0.84, hspace=0.64, wspace=0.28)
    return fig


def utility_figures(df, summary, metric, plot_n=1000, plot_L=None):
    slices = select_plot_slices(df, plot_n, plot_L)
    if slices.empty:
        raise ValueError("No scenarios available for plotting")
    figures = {"PG_utility": _utility_figure(summary, slices, metric)}
    plotted = pd.concat([_slice_rows(summary, row) for row in slices.to_dict(orient="records")])
    if plotted.method.isin(ORACLE_METHODS).any():
        figures["PG_utility_diagnostics"] = _utility_figure(summary, slices, metric, view="diagnostic")
    if plotted.method.eq("workshop_dct7_uncentred").any():
        figures["PG_utility_baseline_comparison"] = _utility_figure(summary, slices, metric, view="baseline")
    slices["metric"] = metric
    slices["target"] = _target_label(metric)[0]
    return figures, slices


def save_utility_figures(df, summary, metric, out, plot_n=1000, plot_L=None):
    figures, slices = utility_figures(df, summary, metric, plot_n, plot_L)
    for name, fig in figures.items():
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    slices.to_csv(out / "PG_utility_plot_slices.csv", index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",default=None,help="folder holding utility_*.csv.gz; defaults to the config out_dir")
    ap.add_argument("--config",default=str(ROOT/"configs/pg.yaml"))
    ap.add_argument("--output-dir", type=Path, help="Output directory; defaults to RESULTS/paper")
    ap.add_argument("--plot-n", type=int, default=1000,
                    help="Representative cohort size; absent values use nearest observed n per scenario (ties use smaller n)")
    ap.add_argument("--plot-L", type=int, default=None,
                    help="Window cap for every scenario; defaults to largest observed L per scenario")
    ap.add_argument("--plot-only", action="store_true",
                    help="Reuse saved RESULTS/paper summary CIs; leave summary tables and paired tests unchanged")
    a=ap.parse_args()
    if a.results is None:
        import yaml
        od=Path(yaml.safe_load(Path(a.config).read_text())["out_dir"])
        data=od if od.is_absolute() else ROOT/od
        res=data
    else:
        res=Path(a.results)
        data=res/"pg" if (res/"pg").is_dir() and not list(res.glob("utility*.csv.gz")) else res
    print(f"reading results from {data}")
    out=a.output_dir if a.output_dir is not None else res/"paper"
    out.mkdir(parents=True,exist_ok=True)
    files=sorted(data.glob("utility*.csv.gz"))
    if not files: raise SystemExit(f"No utility result files in {data}")
    df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    cells=[k for k in ("scenario","L","n","eps") if k in df]
    keys=cells+["method"]
    metrics=[k for k in ("rmse_db","rmse_db_vs_muR","logE_rmse_db","airm","psd12_rmse_db","coh12_mae",
                        "own_logE_rmse_db","own_psd12_rmse_db","own_marginal_rmse_db") if k in df]
    if a.plot_only:
        if "eps" not in df or not metrics:
            raise SystemExit("Plot-only requires epsilon and a supported metric")
        path = res/"paper"/f"PG_{metrics[0]}_ci.csv"
        if not path.exists():
            raise SystemExit(f"Plot-only requires existing cohort-bootstrap summary: {path}")
        save_utility_figures(df, pd.read_csv(path), metrics[0], out, a.plot_n, a.plot_L)
        print(f"Figures only: {out}; reused {path}")
        return
    summary_tables = {}
    for metric in metrics:
        tab=summarize(df,keys,metric,"cohort")
        summary_tables[metric] = tab
        tab.to_csv(out/f"PG_{metric}_ci.csv",index=False)
        tab.head(40).to_latex(out/f"PG_{metric}_preview.tex",index=False,float_format="%.4g")
        print(f"{metric}: {len(tab)} cells")
    reference="dct_centred"
    if reference in df.method.unique() and metrics:
        tests=[]
        for vals,g in df.groupby(cells,dropna=False):
            t=paired_tests(g,metrics[0],reference,["cohort"])
            if not t.empty:
                vals=vals if isinstance(vals,tuple) else (vals,)
                for k,v in zip(cells,vals): t[k]=v
                tests.append(t)
        if tests: pd.concat(tests,ignore_index=True).to_csv(out/"PG_paired_tests_holm.csv",index=False)
    for name in ("window_audit.csv","attacks.csv","audit.csv","linkage.csv","subgroups_contrasts.csv","sites_fusion.csv","classification.csv","linear_controls.csv","sites_status.csv","subgroups_draws.csv.gz"):
        p=data/name
        if p.exists() and p.stat().st_size>1:
            d=pd.read_csv(p)
            d.to_csv(out/("PG_"+name),index=False)
    if "eps" in df and metrics:
        save_utility_figures(df, summary_tables[metrics[0]], metrics[0], out, a.plot_n, a.plot_L)
    (out/"INTERPRETATION.txt").write_text(
        "Experimental intervals summarize repeated cohorts/noise. They are not private release certificates. "
        "Evaluation files may contain exact private truth and must remain inside the trusted benchmark. "
        "Separate benchmark repetitions/methods do not share one epsilon budget.\n")
    print(f"Tables and figures: {out}")

if __name__=="__main__": main()
