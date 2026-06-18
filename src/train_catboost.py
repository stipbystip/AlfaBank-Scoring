import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import CAT_COLS, TrainingConfig


def get_catboost_feature_indices(X):
    cat_features = [
        col
        for col in X.columns
        if col.startswith("last") and any(cat_col in col for cat_col in CAT_COLS)
    ]
    cat_features_idx = [X.columns.get_loc(col) for col in cat_features]
    return cat_features, cat_features_idx


def cast_catboost_categoricals(X_train, X_test, cat_features):
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in cat_features:
        X_train[col] = X_train[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)

    return X_train, X_test


def train_catboost_cv(
    X,
    y,
    X_test,
    models_dir,
    train_dirs_dir,
    config=TrainingConfig(),
    task_type="GPU",
    model_name_prefix="catboost",
    cat_features=None,
):
    from catboost import CatBoostClassifier, Pool

    models_dir = Path(models_dir)
    train_dirs_dir = Path(train_dirs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    train_dirs_dir.mkdir(parents=True, exist_ok=True)

    if cat_features is None:
        cat_features, cat_features_idx = get_catboost_feature_indices(X)
    else:
        cat_features_idx = [X.columns.get_loc(col) for col in cat_features]

    X, X_test = cast_catboost_categoricals(X, X_test, cat_features)

    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    oof_predictions = np.zeros(len(X), dtype=np.float32)
    test_predictions = np.zeros(len(X_test), dtype=np.float32)
    cv_scores = []
    test_pool = Pool(X_test, cat_features=cat_features_idx)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_train_dir = train_dirs_dir / f"fold_{fold}"
        fold_train_dir.mkdir(parents=True, exist_ok=True)

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
        val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)

        pos_rate = y_train.mean()
        scale_pos_weight = ((1 - pos_rate) / pos_rate) ** 0.5

        model = CatBoostClassifier(
            iterations=config.catboost_iterations,
            depth=config.catboost_depth,
            learning_rate=config.catboost_learning_rate,
            eval_metric="AUC",
            random_seed=config.random_state,
            scale_pos_weight=scale_pos_weight,
            verbose=100,
            task_type=task_type,
            train_dir=str(fold_train_dir),
        )
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=150,
            use_best_model=True,
        )
        model.save_model(str(models_dir / f"{model_name_prefix}_fold_{fold}.cbm"))

        val_pred = model.predict_proba(val_pool)[:, 1]
        test_pred = model.predict_proba(test_pool)[:, 1]
        score = roc_auc_score(y_val, val_pred)

        cv_scores.append(score)
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / config.n_splits

        del train_pool, val_pool, model
        gc.collect()

    return {
        "oof": oof_predictions,
        "test": test_predictions,
        "scores": cv_scores,
        "mean_auc": float(np.mean(cv_scores)),
        "std_auc": float(np.std(cv_scores)),
    }


def fit_catboost_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    model_path=None,
    train_dir=None,
    config=TrainingConfig(),
    task_type="GPU",
    cat_features=None,
    model_factory=None,
):
    from catboost import CatBoostClassifier, Pool

    cat_features = cat_features or []
    cat_features_idx = [X_train.columns.get_loc(col) for col in cat_features]
    X_train, X_val = cast_catboost_categoricals(X_train, X_val, cat_features)
    _, X_test = cast_catboost_categoricals(X_train, X_test, cat_features)

    train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)
    test_pool = Pool(X_test, cat_features=cat_features_idx)

    if model_factory is None:
        pos_rate = y_train.mean()
        scale_pos_weight = ((1 - pos_rate) / pos_rate) ** 0.5
        model = CatBoostClassifier(
            iterations=config.catboost_iterations,
            depth=config.catboost_depth,
            learning_rate=config.catboost_learning_rate,
            eval_metric="AUC",
            random_seed=config.random_state,
            scale_pos_weight=scale_pos_weight,
            verbose=100,
            task_type=task_type,
            train_dir=str(train_dir) if train_dir else None,
        )
    else:
        model = model_factory(
            seed=config.random_state,
            train_dir=train_dir,
            scale_pos_weight=((1 - y_train.mean()) / y_train.mean()) ** 0.5,
        )

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=150,
        use_best_model=True,
    )

    if model_path is not None:
        model.save_model(str(model_path))

    val_pred = model.predict_proba(val_pool)[:, 1]
    test_pred = model.predict_proba(test_pool)[:, 1]
    score = roc_auc_score(y_val, val_pred)

    return model, val_pred, test_pred, score


def train_meta_catboost(
    train_target,
    first_train_pred,
    first_test_pred,
    second_train_pred,
    second_test_pred,
    test_ids,
    first_col="pred_mlp_transformer",
    second_col="pred_tr_cat",
    target_col="flag",
    random_state=42,
):
    from catboost import CatBoostClassifier

    meta_train = (
        train_target[["id", target_col]]
        .merge(first_train_pred[["id", first_col]], on="id", how="left")
        .merge(second_train_pred[["id", second_col]], on="id", how="left")
    )
    meta_train = add_meta_prediction_features(meta_train, first_col, second_col)

    X_meta = meta_train[[first_col, second_col, "mean_pred", "diff", "abs_diff"]]
    y_meta = meta_train[target_col].values

    meta_test = first_test_pred[["id", first_col]].merge(
        second_test_pred[["id", second_col]],
        on="id",
        how="inner",
    )
    meta_test = add_meta_prediction_features(meta_test, first_col, second_col)
    X_test_meta = meta_test[[first_col, second_col, "mean_pred", "diff", "abs_diff"]]

    pos_rate = y_meta.mean()
    scale_pos_weight = ((1 - pos_rate) / pos_rate) ** 0.4

    final_meta_model = CatBoostClassifier(
        iterations=1000,
        depth=4,
        learning_rate=0.02,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=100,
        scale_pos_weight=scale_pos_weight,
    )
    final_meta_model.fit(X_meta, y_meta)
    test_predictions_meta = final_meta_model.predict_proba(X_test_meta)[:, 1]

    submission_df = pd.DataFrame({"id": test_ids, "flag": test_predictions_meta})
    return final_meta_model, submission_df, meta_train, meta_test


def add_meta_prediction_features(df, first_col, second_col):
    df = df.copy()
    df["mean_pred"] = (df[first_col] + df[second_col]) / 2
    df["diff"] = df[first_col] - df[second_col]
    df["abs_diff"] = df["diff"].abs()
    return df
