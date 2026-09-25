import glob
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3)

data_dir = '/projectnb/jamlab/prevent_AD_dataset/prevent_AD_dataset/'


def main():
    print("Starting script...")

    # Load outcomes and predictors
    df_outcomes = pd.read_csv(
        data_dir + 'df_longitudinal_outcomes.csv', index_col=0
    )
    df_predictors = sorted(glob.glob(data_dir + '*_400Parcels_bilateral.csv'))

    outcomes = df_outcomes.columns[1:]
    model = 'xgboost'  # Can change model you are testing

    for outcome in outcomes:
        for df in df_predictors:
            predict_longitudinal_outcome(
                df, df_outcomes, outcome, model, multioutput=False, n_perm=1000
            )


def predict_longitudinal_outcome(
    df_predictor, df_outcomes, outcome_var, model, multioutput=False, n_perm=2500
):

    predictor_name = df_predictor.split('.')[0].split('/')[-1]
    print("\n\n")
    print(f"Outcome: {outcome_var}")
    print(f"Predictor: {predictor_name}")

    df_predictor = pd.read_csv(df_predictor, index_col=0)
    df_model = pd.merge(
        df_outcomes[['subject', outcome_var]],
        df_predictor,
        on='subject',
        how='outer',
    )
    df_model = df_model.dropna(how='any')
    print("Dataset shape:", df_model.shape)

    X = df_model.iloc[:, 2:]
    X = np.asfortranarray(X, dtype=np.float64)
    y = np.asfortranarray(df_model[outcome_var])

    # ------------------------------------------------------------------
    # CONSISTENT CV STRATEGY (Used for both tuning and permutation testing)
    # ------------------------------------------------------------------
    cv_strategy = KFold(n_splits=4, shuffle=True, random_state=4)

    # ------------------------------------------------------------------
    # 1. MODEL DEFINITION & HYPERPARAMETER TUNING
    # ------------------------------------------------------------------
    if model == 'ridge':
        scaler = StandardScaler()
        if multioutput:
            regression = RidgeCV(
                alphas=np.logspace(-4, 5, 50),
                scoring='neg_mean_squared_error',
                alpha_per_target=True,
            )
        else:
            regression = RidgeCV(
                alphas=np.logspace(-4, 5, 50),
                scoring='neg_mean_squared_error',
            )

        final_model = make_pipeline(scaler, regression)

    elif model == 'xgboost':
        regression = XGBRegressor(
            objective='reg:squarederror',
            random_state=4,
            n_jobs=1,  # Keep at 1 so GridSearchCV handles parallelization
            tree_method='hist',
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', regression)
        ])

        param_grid = {
            'xgb__n_estimators': [50, 100, 200],
            'xgb__max_depth': [2, 3, 5],
            'xgb__learning_rate': [0.01, 0.05, 0.1],
            'xgb__subsample': [0.8, 1.0],
            'xgb__colsample_bytree': [0.5, 0.8, 1.0],
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_strategy,
            scoring='neg_mean_squared_error',
            n_jobs=16,
            verbose=1,
        )

        print("Finding optimal XGBoost hyperparameters...")
        grid_search.fit(X, y)

        print("Best Parameters:", grid_search.best_params_)
        print("Best CV Score:", grid_search.best_score_)

        final_model = grid_search.best_estimator_

    elif model == 'svr':
        regression = LinearSVR(dual='auto', tol=1e-3)
        pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('svr', regression)
        ])

        param_grid = {
            'svr__C': [0.01, 0.1, 1.0, 10],
            'svr__epsilon': [0.001, 0.01, 0.1, 0.5, 0.75],
        }

        grid_search = HalvingGridSearchCV(
            pipeline,
            param_grid,
            cv=cv_strategy,
            scoring='neg_mean_squared_error',
        )

        print("Finding optimal SVR hyperparameters...")
        grid_search.fit(X, y)
        final_model = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # 2. PERMUTATION TESTING
    # ------------------------------------------------------------------
    score, permutation_scores, pvalue = permutation_test_score(
        estimator=final_model,
        X=X,
        y=y,
        scoring='neg_mean_squared_error',
        cv=cv_strategy,
        n_permutations=n_perm,
        n_jobs=16,
        verbose=1,
        random_state=4,
    )

    print(f'Original score: {score:.4f}')
    print(f'Permutation p-value: {pvalue:.4f}')

    # ------------------------------------------------------------------
    # 3. PLOTTING & SAVING RESULTS
    # ------------------------------------------------------------------
    os.makedirs("xgBoost", exist_ok=True)
    plt.subplots(1, figsize=(4, 4))
    g = sns.displot(permutation_scores)
    plt.vlines(score, ymin=0, ymax=g.ax.get_ylim()[1], color='red')
    plt.suptitle(f'{predictor_name} predicting {outcome_var}')
    plt.savefig(
        f'xgBoost/baseline_{predictor_name}_{model}_{outcome_var}.png', dpi=300
    )
    plt.close()

    # ------------------------------------------------------------------
    # 4. POST-PROCESSING & FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    if model == 'ridge':
        final_model.fit(X, y)
        ridge_obj = final_model.named_steps['ridgecv']
        print("Best Parameters (Alpha):", ridge_obj.alpha_)

    elif model == 'xgboost':
        final_model.fit(X, y)
        best_xgb = final_model.named_steps['xgb']

        importance = pd.Series(
            best_xgb.feature_importances_, index=df_model.columns[2:]
        ).sort_values(ascending=False)

        print("\nTop 20 Features:\n", importance.head(20))

    elif model == 'svr':
        final_model.fit(X, y)
        print("SVR Model fitted successfully.")

    return


if __name__ == "__main__":
    main()