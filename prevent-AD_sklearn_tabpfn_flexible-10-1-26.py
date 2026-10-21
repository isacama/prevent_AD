import os

# OVERRIDE HOME BEFORE ANY IMPORTS (Fixes pathlib.Path.home() resolution)
os.environ["HOME"] = "/projectnb/jamlab/ixc751"
os.environ["TABPFN_CACHE_DIR"] = "/projectnb/jamlab/ixc751/.cache/tabpfn"
os.environ["HF_HOME"] = "/projectnb/jamlab/ixc751/.cache/huggingface"
os.environ["TORCH_HOME"] = "/projectnb/jamlab/ixc751/.cache/torch"
os.environ["XDG_CACHE_HOME"] = "/projectnb/jamlab/ixc751/.cache"
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_FUAxz474RhUoxPRT3m43I0Lo6Z2sSxpPtMniCddooQI"
os.environ["PYTHONWARNINGS"] = "ignore"

# Create directories explicitly
os.makedirs("/projectnb/jamlab/ixc751/.cache/tabpfn", exist_ok=True)

# NOW DO ALL IMPORTS
import glob
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    permutation_test_score,
    GridSearchCV,
    HalvingGridSearchCV,
    KFold,
)
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3)

data_dir = "/projectnb/jamlab/prevent_AD_dataset/prevent_AD_dataset/"


import sys

def main():
    df_outcomes = pd.read_csv(data_dir + "df_longitudinal_outcomes.csv", index_col=0)
    df_predictors = sorted(glob.glob(data_dir + "*_400Parcels_bilateral.csv"))
    outcomes = df_outcomes.columns[1:].tolist()

    all_pairs = [(outcome, pred) for outcome in outcomes for pred in df_predictors]

    lsb_index = os.getenv("LSB_JOBINDEX")

    if lsb_index is not None:
        pair_idx = int(lsb_index) - 1
        if pair_idx < len(all_pairs):
            outcome_var, df_predictor = all_pairs[pair_idx]
            print(f"--- Running Array Task {lsb_index}/{len(all_pairs)}: {outcome_var} x {df_predictor} ---")
            predict_longitudinal_outcome(
                df_predictor, df_outcomes, outcome_var, model_type="tabpfn", multioutput=False, n_perm=100
            )
        else:
            print(f"Index {lsb_index} is out of range for total pairs ({len(all_pairs)}). Exiting.")
    else:
        print("No LSB_JOBINDEX found. Running sequentially...")
        for outcome_var, df_predictor in all_pairs:
            predict_longitudinal_outcome(
                df_predictor, df_outcomes, outcome_var, model_type="tabpfn", multioutput=False, n_perm=100
            )  # I changed the number of permutations from 1000 to 100 


def predict_longitudinal_outcome(
    df_predictor,
    df_outcomes,
    outcome_var,
    model_type,  # Parameter name
    multioutput=False,
    n_perm=100, # I changed the number of permutations from 1000 to 100 
):
    predictor_name = df_predictor.split(".")[0].split("/")[-1]
    print("\n\n")
    print(f"Outcome: {outcome_var}")
    print(f"Predictor: {predictor_name}")

    df_predictor = pd.read_csv(df_predictor, index_col=0)
    df_model = pd.merge(
        df_outcomes[["subject", outcome_var]],
        df_predictor,
        on="subject",
        how="outer",
    ).dropna(how="any")

    print("Dataset shape:", df_model.shape)

    X = df_model.iloc[:, 2:].to_numpy(dtype=np.float32)
    y = df_model[outcome_var].to_numpy(dtype=np.float32)

    cv_strategy = KFold(n_splits=4, shuffle=True, random_state=4)

    # ------------------------------------------------------------------
    # 1. MODEL DEFINITION
    # ------------------------------------------------------------------
    if model_type == "ridge":
        scaler = StandardScaler()
        regression = RidgeCV(
            alphas=np.logspace(-4, 5, 50),
            scoring="neg_mean_squared_error",
            alpha_per_target=multioutput,
        )
        final_model = make_pipeline(scaler, regression)

    elif model_type == "xgboost":
        regression = XGBRegressor(
            objective="reg:squarederror",
            random_state=4,
            n_jobs=1,
            tree_method="hist",
        )
        pipeline = Pipeline([("scaler", StandardScaler()), ("xgb", regression)])
        param_grid = {
            "xgb__n_estimators": [50, 100, 200],
            "xgb__max_depth": [2, 3, 5],
            "xgb__learning_rate": [0.01, 0.05, 0.1],
            "xgb__subsample": [0.8, 1.0],
            "xgb__colsample_bytree": [0.5, 0.8, 1.0],
        }
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_strategy,
            scoring="neg_mean_squared_error",
            n_jobs=4,
            verbose=1,
        )
        grid_search.fit(X, y)
        final_model = grid_search.best_estimator_

    elif model_type == "svr":
        regression = LinearSVR(dual="auto", tol=1e-3)
        pipeline = Pipeline([("scaler", StandardScaler()), ("svr", regression)])
        param_grid = {
            "svr__C": [0.01, 0.1, 1.0, 10],
            "svr__epsilon": [0.001, 0.01, 0.1, 0.5, 0.75],
        }
        grid_search = HalvingGridSearchCV(
            pipeline,
            param_grid,
            cv=cv_strategy,
            scoring="neg_mean_squared_error",
        )
        grid_search.fit(X, y)
        final_model = grid_search.best_estimator_

    elif model_type == "tabpfn":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- DETECTED DEVICE FOR TABPFN: {device} ---")

        tabpfn_cache = os.getenv("TABPFN_CACHE_DIR", "/projectnb/jamlab/ixc751/.cache/tabpfn")
        os.makedirs(tabpfn_cache, exist_ok=True)

        # Fast evaluation settings
        final_model = TabPFNRegressor(
            device=device,
            model_path=tabpfn_cache,
            n_estimators=1,               # Single model pass (max speed)
            ignore_prefiltering=True,     # Skip redundant pre-processing per fold
            show_progress_bar=False,)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # 2. PERMUTATION TESTING
    # ------------------------------------------------------------------
    # Using 'threading' allows multi-threaded CUDA streams on a single H100
    # ------------------------------------------------------------------
    # 2. PERMUTATION TESTING
    # ------------------------------------------------------------------
    import joblib

    print(f"Starting permutation test (n_perm={n_perm}, n_jobs=4)...")
    
    # Use threading backend so all 4 workers safely execute on the GPU concurrently
    with joblib.parallel_backend("threading", n_jobs=4):
        score, permutation_scores, pvalue = permutation_test_score(
            estimator=final_model,
            X=X,
            y=y,
            scoring="neg_mean_squared_error",
            cv=cv_strategy,
            n_permutations=n_perm,
            verbose=1,
            random_state=4,
        )

    print(f"Original score: {score:.4f}")
    print(f"Permutation p-value: {pvalue:.4f}")

    # Save output plots and metrics
    output_dir = "xgBoost" if model_type == "xgboost" else model_type.lower()
    os.makedirs(output_dir, exist_ok=True)

    g = sns.displot(permutation_scores, kind="hist", kde=True)
    g.refline(x=score, color="red", linewidth=2)
    g.fig.suptitle(f"{predictor_name} predicting {outcome_var}", y=1.02)
    plot_filename = f"{output_dir}/baseline_{predictor_name}_{model_type}_{outcome_var}.png"
    g.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close("all")

    results_filename = f"{output_dir}/baseline_{predictor_name}_{model_type}_{outcome_var}_results.txt"
    with open(results_filename, "w") as f:
        f.write(f"Predictor: {predictor_name}\n")
        f.write(f"Outcome: {outcome_var}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Dataset Shape: {df_model.shape}\n")
        f.write(f"Original MSE Score: {score:.6f}\n")
        f.write(f"Permutation p-value: {pvalue:.6f}\n")

    print(f"Saved plot to: {plot_filename}")
    print(f"Saved metrics to: {results_filename}")

    # ------------------------------------------------------------------
    # 3. POST-PROCESSING & FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    if model_type == "ridge":
        final_model.fit(X, y)
        ridge_obj = final_model.named_steps["ridgecv"]
        print("Best Parameters (Alpha):", ridge_obj.alpha_)

    elif model_type == "xgboost":
        final_model.fit(X, y)
        best_xgb = final_model.named_steps["xgb"]

        importance = pd.Series(
            best_xgb.feature_importances_, index=df_model.columns[2:]
        ).sort_values(ascending=False)

        print("\nTop 20 Features:\n", importance.head(20))

    elif model_type == "svr":
        final_model.fit(X, y)
        print("SVR Model fitted successfully.")

    elif model_type == "tabpfn":
        final_model.fit(X, y)
        print("TabPFN Model fitted successfully.")

    return


if __name__ == "__main__":
    main()
