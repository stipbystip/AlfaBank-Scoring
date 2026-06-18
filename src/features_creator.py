from copy import deepcopy

import numpy as np
import pandas as pd

from config import AGG_DICT, CAT_COLS


def make_last_n_features(df, cols, n=3):
    df = df.sort_values(["id", "rn"]).copy()
    parts = []

    for i in range(1, n + 1):
        tmp = df.groupby("id").nth(-i).reset_index()
        tmp = tmp[["id"] + cols].copy()
        tmp.columns = ["id"] + [f"last{i}_{col}" for col in cols]
        parts.append(tmp)

    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on="id", how="left")

    return result


def make_global_agg_features(df, agg_dict=AGG_DICT):
    agg_dict_local = {col: funcs for col, funcs in deepcopy(agg_dict).items() if col in df.columns}

    agg = df.groupby("id", sort=False).agg(agg_dict_local)
    agg.columns = [f"global_{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    history_len = df.groupby("id", sort=False).size().rename("history_len").reset_index()

    return agg.merge(history_len, on="id", how="left").copy()


def make_n_last_cat_features(df, cat_cols=CAT_COLS, n=3):
    cat_cols = [col for col in cat_cols if col in df.columns]
    parts = []

    for i in range(1, n + 1):
        tmp = df.groupby("id").nth(-i).reset_index()
        tmp = tmp[["id"] + cat_cols].copy()
        tmp.columns = ["id"] + [f"last{i}_{col}" for col in cat_cols]
        parts.append(tmp)

    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on="id", how="left")

    return result


def make_count_unique_cat_cols(df, cat_cols=CAT_COLS):
    cat_cols = [col for col in cat_cols if col in df.columns]
    if not cat_cols:
        return df[["id"]].drop_duplicates().copy()

    return (
        df.groupby("id")[cat_cols]
        .nunique()
        .rename(columns={col: f"{col}_nunique" for col in cat_cols})
        .reset_index()
    )


def make_last_k_agg_features(df, cols, k_values=(3, 5), agg_funcs=("mean", "max", "sum")):
    df = df.sort_values(["id", "rn"]).copy()
    cols = [col for col in cols if col in df.columns]
    parts = []

    for k in k_values:
        tail_df = df.groupby("id", group_keys=False).tail(k)
        agg = tail_df.groupby("id", sort=False)[cols].agg(list(agg_funcs))
        agg.columns = [f"last{k}_{col}_{stat}" for col, stat in agg.columns]
        parts.append(agg.reset_index())

    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on="id", how="left")

    return result


def add_diff_features(features, cols):
    new_features = {}

    for col in cols:
        last_col = f"last1_{col}"
        global_mean_col = f"global_{col}_mean"
        global_max_col = f"global_{col}_max"

        if last_col in features.columns and global_mean_col in features.columns:
            new_features[f"diff_last1_vs_global_mean_{col}"] = (
                features[last_col] - features[global_mean_col]
            )

        if last_col in features.columns and global_max_col in features.columns:
            new_features[f"diff_last1_vs_global_max_{col}"] = (
                features[last_col] - features[global_max_col]
            )

    if not new_features:
        return features.copy()

    return pd.concat(
        [features, pd.DataFrame(new_features, index=features.index)],
        axis=1,
    ).copy()


def add_last_diff_features(features, cols):
    new_cols = {}

    for col in cols:
        c1 = f"last1_{col}"
        c2 = f"last2_{col}"

        if c1 in features.columns and c2 in features.columns:
            new_cols[f"diff_last1_last2_{col}"] = features[c1] - features[c2]

    if not new_cols:
        return features.copy()

    return pd.concat(
        [features, pd.DataFrame(new_cols, index=features.index)],
        axis=1,
    ).copy()


def extract_client_features(df, version=1):
    if version not in (1, 2):
        raise ValueError("version must be 1 or 2")

    df = df.sort_values(["id", "rn"]).copy()
    feature_cols = [c for c in df.columns if c not in ["id", "rn"]]
    missing_cat_cols = [col for col in CAT_COLS if col not in df.columns]
    if missing_cat_cols:
        print("Missing cat_cols:", missing_cat_cols)

    global_features = make_global_agg_features(df)
    last_n_features = make_last_n_features(df, feature_cols, n=3)
    last_n_cat_features = make_n_last_cat_features(df, CAT_COLS, n=3)

    features = global_features.merge(last_n_features, on="id", how="left")
    features = features.merge(last_n_cat_features, on="id", how="left")
    version_1_cols = features.columns.tolist()
    version_2_cols = []

    if version == 2:
        cat_count_unique_features = make_count_unique_cat_cols(df, CAT_COLS)
        last_k_features = make_last_k_agg_features(df, feature_cols, k_values=(3, 5))
        features = features.merge(cat_count_unique_features, on="id", how="left")
        features = features.merge(last_k_features, on="id", how="left")
        features = add_diff_features(features, feature_cols)
        features = add_last_diff_features(features, feature_cols)
        version_2_cols = features.columns.tolist()

    return features, version_1_cols, version_2_cols


def extract_client_features_v1(df):
    return extract_client_features(df, version=1)[0]


def numeric_feature_columns(df):
    return [
        col
        for col in df.columns
        if col not in ("id", "flag") and np.issubdtype(df[col].dtype, np.number)
    ]
