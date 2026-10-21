"""
Combine the per-pair TabPFN SHAP outputs into a few organized files.

Run from /projectnb/jamlab/ixc751:
    python summarize_tabpfn_shap.py                      # reads shap_tabpfn_all/
    python summarize_tabpfn_shap.py shap_tabpfn_rf_fulldata_top20

Reads   <folder>/shap_<predictor>_400Parcels_bilateral_<outcome>_regions.csv
        <folder>/shap_<predictor>_400Parcels_bilateral_<outcome>_networks.csv
Writes  <folder>/summary/
            shap_summary.xlsx                 (everything below, one sheet each)
            shap_top_regions_per_pair.csv     (top N regions for every pair)
            shap_networks_long.csv            (every pair x network)
            shap_network_relative_matrix.csv  (network x pair, 1.0 = average network)
            shap_region_consistency.csv       (regions that rank high in many pairs)
            shap_network_heatmap.png
            shap_region_matrix.csv            (all 200 regions x pairs, 1.0 = average region)
            shap_region_heatmap_all200.png    (every region, grouped by network)
            shap_region_heatmap_top.png       (readable: regions in the top 10 of any pair)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

FOLDER = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "shap_tabpfn_all"
TOP_N = 10            # regions listed per pair
CONSISTENCY_TOP = 20  # a region "counts" for a pair if its rank is <= this
SUFFIX = "_400Parcels_bilateral_"


def parse_name(path, kind):
    """shap_<predictor>_400Parcels_bilateral_<outcome>_regions.csv -> (predictor, outcome)"""
    base = os.path.basename(path)[len("shap_"):-len(f"_{kind}.csv")]
    predictor, outcome = base.split(SUFFIX)
    return predictor, outcome


def load_all(kind):
    frames = []
    for path in sorted(glob.glob(os.path.join(FOLDER, f"shap_*_{kind}.csv"))):
        predictor, outcome = parse_name(path, kind)
        d = pd.read_csv(path)
        d.insert(0, "pair", f"{outcome} x {predictor}")
        d.insert(1, "outcome", outcome)
        d.insert(2, "predictor", predictor)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    regions = load_all("regions")
    networks = load_all("networks")
    if regions.empty:
        sys.exit(f"No SHAP files found in {FOLDER}/. Run from /projectnb/jamlab/ixc751.")

    pairs = regions["pair"].unique()
    print(f"Found {len(pairs)} pairs in {FOLDER}/:")
    for p in pairs:
        print("  ", p)

    out_dir = os.path.join(FOLDER, "summary")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- 1. top regions per pair ----------------
    top = (regions.sort_values(["pair", "rank"]).groupby("pair").head(TOP_N)
           .copy())
    top["stable_3of4_folds"] = top["folds_in_top20_of_4"] >= 3
    top["direction"] = np.where(top["direction_r"] > 0, "higher value -> higher prediction",
                                np.where(top["direction_r"] < 0,
                                         "higher value -> lower prediction", "n/a"))
    top_cols = ["pair", "outcome", "predictor", "rank", "feature", "network",
                "mean_abs_shap", "direction_r", "direction", "folds_in_top20_of_4",
                "stable_3of4_folds"]
    top = top[top_cols]
    top.to_csv(os.path.join(out_dir, "shap_top_regions_per_pair.csv"), index=False)

    # ---------------- 2. networks, long + relative matrix ----------------
    # per-region importance divided by the pair's average over networks:
    # 1.0 = average network, 2.0 = twice as important per region as average.
    # This makes pairs with different outcome scales comparable and removes
    # the bias toward networks that simply contain more regions.
    networks["relative_importance"] = (
        networks["mean_abs_shap_per_region"]
        / networks.groupby("pair")["mean_abs_shap_per_region"].transform("mean"))
    networks["rank_by_per_region"] = (
        networks.groupby("pair")["mean_abs_shap_per_region"].rank(ascending=False).astype(int))
    networks = networks.sort_values(["pair", "rank_by_per_region"])
    networks.to_csv(os.path.join(out_dir, "shap_networks_long.csv"), index=False)

    matrix = networks.pivot(index="network", columns="pair", values="relative_importance")
    matrix["mean_across_pairs"] = matrix.mean(axis=1)
    matrix = matrix.sort_values("mean_across_pairs", ascending=False)
    matrix.to_csv(os.path.join(out_dir, "shap_network_relative_matrix.csv"))

    # ---------------- 3. regions that recur across pairs ----------------
    regions["in_top"] = regions["rank"] <= CONSISTENCY_TOP
    cons = (regions.groupby(["feature", "network"])
            .agg(n_pairs_in_top=("in_top", "sum"),
                 mean_rank=("rank", "mean"),
                 mean_abs_shap=("mean_abs_shap", "mean"),
                 pairs=("pair", lambda s: "; ".join(
                     s[regions.loc[s.index, "in_top"]])))
            .reset_index()
            .sort_values(["n_pairs_in_top", "mean_rank"], ascending=[False, True]))
    cons = cons[cons["n_pairs_in_top"] > 0]
    cons.to_csv(os.path.join(out_dir, "shap_region_consistency.csv"), index=False)

    # ---------------- 4. Excel workbook ----------------
    try:
        xlsx = os.path.join(out_dir, "shap_summary.xlsx")
        with pd.ExcelWriter(xlsx) as xw:
            top.to_excel(xw, sheet_name="top_regions_per_pair", index=False)
            networks.to_excel(xw, sheet_name="networks_long", index=False)
            matrix.to_excel(xw, sheet_name="network_relative_matrix")
            cons.to_excel(xw, sheet_name="region_consistency", index=False)
            regions.pivot(index="feature", columns="pair", values="mean_abs_shap") \
                .to_excel(xw, sheet_name="region_raw_mean_abs_shap")
        print(f"Saved workbook: {xlsx}")
    except Exception as e:
        print(f"(Excel workbook skipped: {e}. The CSVs were still written.)")

    # ---------------- 5. heatmap ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        m = matrix.drop(columns="mean_across_pairs")
        m.columns = [c.replace(" x ", "\n") for c in m.columns]
        fig, ax = plt.subplots(figsize=(max(7, 1.3 * m.shape[1] + 3),
                                        max(5, 0.35 * m.shape[0] + 2)))
        sns.heatmap(m, annot=True, fmt=".1f", cmap="viridis",
                    cbar_kws={"label": "relative importance per region (1 = average network)"},
                    ax=ax)
        ax.set_title("TabPFN SHAP: network importance (per region, relative)")
        ax.set_xlabel("")
        ax.set_ylabel("Network")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        png = os.path.join(out_dir, "shap_network_heatmap.png")
        fig.savefig(png, dpi=200)
        print(f"Saved heatmap: {png}")
    except Exception as e:
        print(f"(Heatmap skipped: {e})")

    # ---------------- 4b. region-level matrix (each of the 200 regions) ----------------
    # relative importance = region's mean |SHAP| / average region in that pair
    # (1.0 = average region, 3.0 = three times the average).
    rmat = regions.pivot(index="feature", columns="pair", values="mean_abs_shap")
    rrel = rmat / rmat.mean(axis=0)
    rrank = regions.pivot(index="feature", columns="pair", values="rank")
    net_of = regions.drop_duplicates("feature").set_index("feature")["network"]
    short = lambda f: f.replace("17networks_LH_", "").replace("17networks_RH_", "")

    rtab = rrel.copy()
    rtab.insert(0, "network", net_of.reindex(rtab.index))
    rtab["mean_relative"] = rrel.mean(axis=1)
    rtab["best_rank"] = rrank.min(axis=1).astype(int)
    rtab["n_pairs_in_top20"] = (rrank <= CONSISTENCY_TOP).sum(axis=1)
    rtab = rtab.sort_values("mean_relative", ascending=False)
    rtab.to_csv(os.path.join(out_dir, "shap_region_matrix.csv"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cols = list(rrel.columns)
        nice = [c.replace(" x ", "\n") for c in cols]
        vmax = float(np.nanpercentile(rrel.values, 99))

        # (a) all regions, grouped by network (networks ordered by mean importance)
        net_order = rtab.groupby("network")["mean_relative"].mean().sort_values(
            ascending=False).index.tolist()
        rtab["_n"] = rtab["network"].map({n: i for i, n in enumerate(net_order)})
        allr = rtab.sort_values(["_n", "mean_relative"], ascending=[True, False])
        fig, ax = plt.subplots(figsize=(8, max(10, 0.16 * len(allr))))
        sns.heatmap(allr[cols], cmap="viridis", vmin=0, vmax=vmax, ax=ax,
                    yticklabels=[short(f) for f in allr.index],
                    cbar_kws={"label": "relative importance (1 = average region; capped at 99th pct)"})
        ax.set_xticklabels(nice, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=4.5)
        # network separators
        pos = 0
        for n in net_order:
            k = int((allr["network"] == n).sum())
            pos += k
            ax.axhline(pos, color="white", lw=1.2)
        ax.set_ylabel("Region (grouped by network, strongest network first)")
        ax.set_xlabel("")
        ax.set_title("TabPFN SHAP: all regions")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "shap_region_heatmap_all200.png"), dpi=220)
        plt.close("all")

        # (b) readable version: regions in the top 10 of at least one pair
        keep = rrank.index[(rrank <= TOP_N).any(axis=1)]
        topr = rtab.loc[keep].sort_values(["_n", "mean_relative"], ascending=[True, False])
        fig, ax = plt.subplots(figsize=(8.5, max(5, 0.27 * len(topr) + 2)))
        sns.heatmap(topr[cols], annot=True, fmt=".1f", cmap="viridis", vmin=0, vmax=vmax,
                    ax=ax, annot_kws={"size": 7},
                    yticklabels=[f"{short(f)}" for f in topr.index],
                    cbar_kws={"label": "relative importance (1 = average region)"})
        ax.set_xticklabels(nice, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel(f"Regions in the top {TOP_N} of any pair")
        ax.set_xlabel("")
        ax.set_title("TabPFN SHAP: top regions")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "shap_region_heatmap_top.png"), dpi=220)
        plt.close("all")
        print("Saved region heatmaps (all200 + top)")
    except Exception as e:
        print(f"(Region heatmaps skipped: {e})")

    # ---------------- console report ----------------
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== Top 3 regions per pair ===")
    print(top.groupby("pair").head(3)[["pair", "rank", "feature", "mean_abs_shap",
                                       "direction_r", "folds_in_top20_of_4"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\n=== Top 3 networks per pair (per-region importance) ===")
    print(networks[networks.rank_by_per_region <= 3][
        ["pair", "rank_by_per_region", "network", "relative_importance"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n=== Regions in the top {CONSISTENCY_TOP} of 2+ pairs ===")
    print(cons[cons.n_pairs_in_top >= 2].head(15)[
        ["feature", "network", "n_pairs_in_top", "mean_rank"]]
          .to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nAll files are in {out_dir}/")


if __name__ == "__main__":
    main()
