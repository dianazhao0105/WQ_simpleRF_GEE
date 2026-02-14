# -*- coding: utf-8 -*- 
"""
@Author: DianaZhao 
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import ee
from geemap import ml

ee.Authenticate()
ee.Initialize()


# -------------------------
# 1) Read data
# -------------------------
train_val_path = "train_val.csv"
ind_test_path  = "independent_test.csv"
feature_map_path = "feature_flags.csv"   # columns: feature, flag

df_all  = pd.read_csv(train_val_path)
df_test = pd.read_csv(ind_test_path)
feat_map = pd.read_csv(feature_map_path)

targets = ["CODMn", "TUB", "TN", "TP"]


# -------------------------
# 2) Feature groups from flag
# -------------------------
def get_group(flag_value):
    return feat_map.loc[feat_map["flag"] == flag_value, "feature"].tolist()

single_bands = get_group("single")
combo_bands  = get_group("combo")
env_cols     = get_group("env")
keep_cols    = get_group("keep")

def keep_existing(cols, df):
    return [c for c in cols if c in df.columns]

single_bands = keep_existing(single_bands, df_all)
combo_bands  = keep_existing(combo_bands, df_all)
env_cols     = keep_existing(env_cols, df_all)
keep_cols    = keep_existing(keep_cols, df_all)


# -------------------------
# 3) Feature selection (train only)
# -------------------------
THRESH = 0.3

def select_spectral(df_train, y_col, single_cols, combo_cols, thresh=0.3):
    y = df_train[y_col]

    # single-band first
    if len(single_cols) > 0:
        r_single = df_train[single_cols].corrwith(y).abs().sort_values(ascending=False)
        chosen_single = r_single[r_single > thresh].index.tolist()
        if chosen_single:
            return chosen_single

    # combos next
    if len(combo_cols) > 0:
        r_combo = df_train[combo_cols].corrwith(y).abs().sort_values(ascending=False)
        chosen_combo = r_combo[r_combo > thresh].index.tolist()
        if chosen_combo:
            return chosen_combo

        r_combo = r_combo.dropna()
        if len(r_combo) > 0:
            return [r_combo.index[0]]

    return []

def select_best_env(df_train, y_col, env_cols):
    if len(env_cols) == 0:
        return []
    y = df_train[y_col]
    r_env = df_train[env_cols].corrwith(y).abs().sort_values(ascending=False).dropna()
    if len(r_env) == 0:
        return []
    return [r_env.index[0]]


# -------------------------
# 4) Manual grid search
# -------------------------
param_grid = {
    "n_estimators": range(20, 101, 10),
    "max_depth": range(7, 16, 1),
    "min_samples_split": range(2, 6, 1),
    "min_samples_leaf": range(1, 6, 1),
}

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


# split + constraint
RANDOM_STATE = 42
VAL_SIZE = 0.30
OVERFIT_GAP_MAX = 0.11

results = []

for tgt in targets:
    # ############ (A) Clean minimal columns first (avoid losing samples due to many env vars)
    essential_cols = list(set(single_bands + combo_bands + keep_cols + [tgt]))
    df_clean = df_all[essential_cols].dropna()

    # Train/Val split
    train_df, val_df = train_test_split(
        df_clean, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # ############ (B) Feature selection on TRAIN only
    selected_spectral = select_spectral(train_df, tgt, single_bands, combo_bands, thresh=THRESH)

    if tgt == "TUB":
        selected_env = []
    else:
        train_env_view = df_all.loc[train_df.index, env_cols + [tgt]].dropna()
        selected_env = select_best_env(train_env_view, tgt, env_cols)

    feature_cols = selected_spectral + selected_env + keep_cols

    # ############ (C) Drop NA
    train_sub = df_all.loc[train_df.index, feature_cols + [tgt]].dropna()
    val_sub   = df_all.loc[val_df.index,   feature_cols + [tgt]].dropna()

    X_train, y_train = train_sub[feature_cols], train_sub[tgt]
    X_val,   y_val   = val_sub[feature_cols],   val_sub[tgt]

    # ############ (D) Manual grid search: fit on TRAIN, pick by VAL with overfit-gap constraint
    best_model = None
    best_params = None
    best_val_r2 = -np.inf
    best_train_r2 = None
    best_gap = None
    used_constraint = True

    candidates = []

    for params in ParameterGrid(param_grid):
        rf = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        rf.fit(X_train, y_train)

        r2_tr = r2_score(y_train, rf.predict(X_train))
        r2_va = r2_score(y_val,   rf.predict(X_val))
        gap = r2_tr - r2_va

        candidates.append((params, r2_tr, r2_va, gap))

        if gap <= OVERFIT_GAP_MAX and r2_va > best_val_r2:
            best_val_r2 = r2_va
            best_train_r2 = r2_tr
            best_gap = gap
            best_params = params
            best_model = rf

    # If nothing satisfies the gap constraint, fall back to best VAL R2
    if best_model is None:
        used_constraint = False
        best_params, best_train_r2, best_val_r2, best_gap = max(candidates, key=lambda x: x[2])
        best_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params).fit(X_train, y_train)

    # ========================== Validation metrics ==========================
    val_pred = best_model.predict(X_val)
    val_m = metrics(y_val.values, val_pred)

    # ========================== Independent test evaluation ==========================
    test_sub = df_test[feature_cols + [tgt]].dropna()
    X_test, y_test = test_sub[feature_cols], test_sub[tgt]
    test_pred = best_model.predict(X_test)
    test_m = metrics(y_test.values, test_pred)

    print(f"\nTarget: {tgt}")
    print(f"Selected spectral ({len(selected_spectral)}): {selected_spectral}")
    print(f"Selected env (0/1): {selected_env}")
    print(f"Keep vars: {len(keep_cols)}")
    print(f"Best params: {best_params} | gap constraint used: {used_constraint}")
    print(f"Train R2: {best_train_r2:.3f} | Val R2: {best_val_r2:.3f} | Gap: {best_gap:.3f}")
    print(f"VAL :  R2={val_m['R2']:.3f}, RMSE={val_m['RMSE']:.3f}, MAE={val_m['MAE']:.3f}")
    print(f"TEST: R2={test_m['R2']:.3f}, RMSE={test_m['RMSE']:.3f}, MAE={test_m['MAE']:.3f}")

    results.append({
        "target": tgt,
        "n_train": len(train_sub),
        "n_val": len(val_sub),
        "n_test": len(test_sub),
        "n_features": len(feature_cols),
        "n_spectral": len(selected_spectral),
        "selected_env": selected_env[0] if len(selected_env) == 1 else "",
        "n_keep": len(keep_cols),
        "best_params": best_params,
        "gap_constraint_used": used_constraint,
        "train_R2": float(best_train_r2),
        "val_R2": float(best_val_r2),
        "train_val_gap": float(best_gap),
        "val_RMSE": val_m["RMSE"],
        "val_MAE": val_m["MAE"],
        "test_R2": test_m["R2"],
        "test_RMSE": test_m["RMSE"],
        "test_MAE": test_m["MAE"],
    })

    # ############ (E) Upload RF model to GEE
    trees = ml.rf_to_strings(best_model, feature_cols, output_mode="REGRESSION")

    cloud_model_name = f"users/your_username/RF_{tgt}"

    task = ml.export_trees_to_fc(trees, cloud_model_name)
    task.start()
    print(f"Started GEE export task for {tgt}: {cloud_model_name}")

pd.DataFrame(results).to_csv("rf_manual_gridsearch_val_then_independent_test_summary.csv", index=False)
print("\nSaved: rf_manual_gridsearch_val_then_independent_test_summary.csv")
