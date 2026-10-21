"""
Summarize TabPFN permutation-test results into one table.

Run from /projectnb/jamlab/ixc751 (same place the jobs ran):
    python summarize_tabpfn_results.py                       # all-features runs (tabpfn/)
    python summarize_tabpfn_results.py tabpfn_rf_infold_top20  # RF top-20 runs
    EXPECTED_PAIRS=132 python summarize_tabpfn_results.py tabpfn_newblup   # new BLUP outcomes

Reads   tabpfn/*_results.txt        (only runs with 1000 permutations by default)
        tabpfn/*_permscores.npy     (null distribution, if present)
Writes  tabpfn/summary_tabpfn_results.csv
        tabpfn/summary_tabpfn_pvalue_heatmap.png
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

RESULTS_DIR = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "tabpfn"
REQUIRED_N_PERM = 1000   # set to None to include runs with any number of permutations
EXPECTED_PAIRS = int(os.getenv("EXPECTED_PAIRS", "60"))   # e.g. EXPECTED_PAIRS=132 for the new outcomes
ALPHA = 0.05


def parse_results_file(path):
    fields = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                fields[key.strip()] = val.strip()
    return fields


def fdr_bh(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(ranked, 1.0)
    return adj


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_results.txt")))
    rows, skipped = [], []

    for path in files:
        d = parse_results_file(path)
        n_perm = int(d["Number of permutations"]) if "Number of permutations" in d else None
        if REQUIRED_N_PERM is not None and n_perm != REQUIRED_N_PERM:
            skipped.append(os.path.basename(path))
            continue

        score = float(d["Original MSE Score"])          # negative MSE
        mse = -score
        shape = re.findall(r"\d+", d.get("Dataset Shape", ""))
        n_subj = int(shape[0]) if shape else np.nan

        # Null distribution summary from saved permutation scores
        null_mean_mse = np.nan
        npy = path.replace("_results.txt", "_permscores.npy")
        if os.path.exists(npy):
            null_mean_mse = -float(np.mean(np.load(npy)))

        rows.append({
            "outcome": d["Outcome"],
            "predictor": d["Predictor"].replace("_400Parcels_bilateral", ""),
            "n_subjects": n_subj,
            "n_perm": n_perm,
            "cv_mse": mse,
            "null_mean_mse": null_mean_mse,
            # % reduction in MSE vs. the average permuted (chance) model; >0 = better than chance
            "mse_improvement_vs_null_pct": 100 * (null_mean_mse - mse) / null_mean_mse,
            "p_perm": float(d["Permutation p-value"]),
            "runtime_min": float(d.get("Runtime (min)", "nan")),
        })

    if not rows:
        sys.exit("No matching results files found. Run this from /projectnb/jamlab/ixc751.")

    df = pd.DataFrame(rows)
    n_tests = len(df)
    df["p_fdr"] = fdr_bh(df["p_perm"])
    df["p_bonferroni"] = np.minimum(df["p_perm"] * n_tests, 1.0)
    # FDR within each outcome (useful if each outcome is treated as its own family of tests)
    df["p_fdr_within_outcome"] = df.groupby("outcome")["p_perm"].transform(fdr_bh)
    df["sig_uncorrected"] = df["p_perm"] < ALPHA
    df["sig_fdr"] = df["p_fdr"] < ALPHA
    df["sig_bonferroni"] = df["p_bonferroni"] < ALPHA
    df = df.sort_values(["p_perm", "outcome", "predictor"]).reset_index(drop=True)

    out_csv = os.path.join(RESULTS_DIR, "summary_tabpfn_results.csv")
    df.to_csv(out_csv, index=False)

    # ---------------- Console report ----------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(f"Results included: {n_tests} / {EXPECTED_PAIRS} expected pairs "
          f"(n_perm = {REQUIRED_N_PERM})")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) from other runs (different/no n_perm).")
    if n_tests < EXPECTED_PAIRS:
        print(f"NOTE: {EXPECTED_PAIRS - n_tests} pair(s) missing - rerun this script when all are done.")
    print(f"Smallest possible p with {REQUIRED_N_PERM} permutations: "
          f"{1 / (REQUIRED_N_PERM + 1):.4f}; Bonferroni threshold: {ALPHA / n_tests:.5f}\n")

    print(f"Uncorrected p < {ALPHA}: {df['sig_uncorrected'].sum()}")
    print(f"FDR (BH, all tests) < {ALPHA}: {df['sig_fdr'].sum()}")
    print(f"Bonferroni < {ALPHA}: {df['sig_bonferroni'].sum()}\n")

    cols = ["outcome", "predictor", "n_subjects", "cv_mse", "null_mean_mse",
            "mse_improvement_vs_null_pct", "p_perm", "p_fdr", "p_fdr_within_outcome", "p_bonferroni"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nSaved table: {out_csv}")

    # ---------------- Heatmap ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        pmat = df.pivot(index="predictor", columns="outcome", values="p_perm")
        fdr_mat = df.pivot(index="predictor", columns="outcome", values="p_fdr")
        annot = pmat.map(lambda p: f"{p:.3f}") if hasattr(pmat, "map") else pmat.applymap(lambda p: f"{p:.3f}")
        annot = annot.where(~(fdr_mat < ALPHA), annot + "*")

        # many outcomes -> put outcomes on the rows so the figure stays readable
        outcomes_on_rows = pmat.shape[1] > pmat.shape[0]
        if outcomes_on_rows:
            pmat, fdr_mat, annot = pmat.T, fdr_mat.T, annot.T
        fig_h = max(4, 0.45 * len(pmat.index) + 1.5)
        fig_w = max(6, 1.3 * len(pmat.columns) + 4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(-np.log10(pmat), annot=annot, fmt="", cmap="viridis",
                    cbar_kws={"label": "-log10(permutation p)"}, ax=ax)
        ax.set_title(f"{RESULTS_DIR}: permutation p-values ({REQUIRED_N_PERM} perms)\n* = FDR < {ALPHA}")
        ax.set_xlabel("Predictor" if outcomes_on_rows else "Outcome")
        ax.set_ylabel("Outcome" if outcomes_on_rows else "Predictor")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_png = os.path.join(RESULTS_DIR, "summary_tabpfn_pvalue_heatmap.png")
        fig.savefig(out_png, dpi=200)
        print(f"Saved heatmap: {out_png}")
    except Exception as e:
        print(f"(Heatmap skipped: {e})")


if __name__ == "__main__":
    main()
