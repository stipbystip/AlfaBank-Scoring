from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import TrainingConfig
from features_creator import extract_client_features
from models import CreditTransformer
from train_catboost import train_catboost_cv, train_meta_catboost
from train_mlp import predict_mlp, train_mlp_on_embeddings
from train_sequence_models import (
    build_embedding_frames,
    build_sequences,
    extract_sequence_embeddings,
    make_sequence_fold_data,
    resolve_device,
    train_sequence_model,
    train_transformer,
    train_xlstm,
)
from utils import make_train_with_pseudo, optimize_data_types


def prepare_tabular_features(train_data, test_data, train_target, version=1, pseudo_target=None):
    train_features, f1_train, f2_train = extract_client_features(train_data, version=version)
    train_features = optimize_data_types(train_features)

    test_features, _, _ = extract_client_features(test_data, version=version)
    test_features = optimize_data_types(test_features)

    df_train = train_features.merge(train_target, on="id", how="inner")
    if pseudo_target is not None:
        df_train, _ = make_train_with_pseudo(df_train, test_features, pseudo_target)

    df_train = df_train.sort_values("id").reset_index(drop=True)
    X_base = df_train.drop(columns=["id", "flag"]).copy()
    y = df_train["flag"].astype(int).reset_index(drop=True)
    ids = df_train["id"].astype(int).values

    X_test_base = test_features.drop(columns=["id"]).copy()
    test_ids = test_features["id"].astype(int).values

    return {
        "X": X_base,
        "y": y,
        "ids": ids,
        "X_test": X_test_base,
        "test_ids": test_ids,
        "feature_cols_v1": f1_train,
        "feature_cols_v2": f2_train,
    }


def train_final_mlp_on_sequence_embeddings_cv(
    X_base,
    y,
    ids,
    test_ids,
    train_sequences,
    test_sequences,
    models_dir,
    config=TrainingConfig(),
    model_factory=CreditTransformer,
    embedding_prefix="trans_emb",
):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    oof_predictions = np.zeros(len(X_base), dtype=np.float32)
    oof_mask = np.zeros(len(X_base), dtype=bool)
    test_predictions = np.zeros(len(test_ids), dtype=np.float32)
    cv_scores = []

    ids_int = pd.Series(ids).astype(int).values
    test_ids_int = pd.Series(test_ids).astype(int).values
    id_to_pos = pd.Series(np.arange(len(ids_int)), index=ids_int)
    device = resolve_device(config.device)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_base, y), start=1):
        train_ids = ids_int[train_idx]
        val_ids = ids_int[val_idx]
        y_train_raw = y.iloc[train_idx].values
        y_val_raw = y.iloc[val_idx].values

        (
            train_fold_sequences,
            train_fold_targets,
            val_fold_sequences,
            val_fold_targets,
        ) = make_sequence_fold_data(
            train_ids,
            val_ids,
            y_train_raw,
            y_val_raw,
            train_sequences,
        )

        y_train_fold_clean = np.array(list(train_fold_targets.values()), dtype=np.float32)
        num_pos = y_train_fold_clean.sum()
        num_neg = len(y_train_fold_clean) - num_pos
        scale_pos_weight = min(np.sqrt(num_neg / max(num_pos, 1)), 6.0)

        sequence_model = train_sequence_model(
            train_fold_sequences,
            train_fold_targets,
            val_fold_sequences,
            val_fold_targets,
            model_factory=model_factory,
            epochs=config.transformer_epochs,
            scale_pos_weight=scale_pos_weight,
            batch_size=config.batch_size_sequences,
            device=device,
        )

        train_embeddings_df = extract_sequence_embeddings(
            sequence_model,
            train_fold_sequences,
            prefix=embedding_prefix,
            device=device,
        )
        val_embeddings_df = extract_sequence_embeddings(
            sequence_model,
            val_fold_sequences,
            prefix=embedding_prefix,
            device=device,
        )
        test_embeddings_df = extract_sequence_embeddings(
            sequence_model,
            test_sequences,
            prefix=embedding_prefix,
            device=device,
        )

        train_df, val_df, test_df = build_embedding_frames(
            train_embeddings_df,
            val_embeddings_df,
            test_embeddings_df,
            train_ids,
            val_ids,
            test_ids_int,
            y_train_raw,
            y_val_raw,
        )

        emb_cols = [col for col in train_df.columns if col.startswith(f"{embedding_prefix}_")]
        X_train = train_df[emb_cols].copy()
        y_train = train_df["target"].values
        X_val = val_df[emb_cols].copy()
        y_val = val_df["target"].values
        X_test = test_df[emb_cols].copy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        pos_rate = y_train.mean()
        mlp_scale_pos_weight = min(((1 - pos_rate) / max(pos_rate, 1e-8)) ** 0.5, 6.0)

        model_mlp, best_mlp_auc = train_mlp_on_embeddings(
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            input_dim=X_train_scaled.shape[1],
            scale_pos_weight=mlp_scale_pos_weight,
            epochs=config.mlp_epochs,
            batch_size=config.batch_size_embeddings,
            device=device,
        )

        torch.save(
            {
                "model_state_dict": model_mlp.state_dict(),
                "input_dim": X_train_scaled.shape[1],
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "best_auc": best_mlp_auc,
            },
            models_dir / f"mlp_{embedding_prefix}_fold_{fold}.pt",
        )

        mlp_val_pred = predict_mlp(
            model=model_mlp,
            X=X_val_scaled,
            batch_size=config.batch_size_embeddings,
            device=device,
        )
        mlp_test_pred = predict_mlp(
            model=model_mlp,
            X=X_test_scaled,
            batch_size=config.batch_size_embeddings,
            device=device,
        )

        score = roc_auc_score(y_val, mlp_val_pred)
        cv_scores.append(score)

        val_pred_df = pd.DataFrame({"id": val_df["id"].astype(int).values, "pred": mlp_val_pred})
        positions = id_to_pos.loc[val_pred_df["id"].values].values
        oof_predictions[positions] = val_pred_df["pred"].values
        oof_mask[positions] = True
        test_predictions += mlp_test_pred / config.n_splits

        del sequence_model, model_mlp
        torch.cuda.empty_cache()

    return {
        "oof": oof_predictions,
        "oof_mask": oof_mask,
        "test": test_predictions,
        "scores": cv_scores,
        "mean_auc": float(np.mean(cv_scores)),
        "std_auc": float(np.std(cv_scores)),
        "oof_auc": float(roc_auc_score(y.values[oof_mask], oof_predictions[oof_mask])),
    }


# Backward-compatible names from the first refactoring pass.
train_mlp_on_sequence_embeddings_cv = train_final_mlp_on_sequence_embeddings_cv

__all__ = [
    "build_sequences",
    "extract_sequence_embeddings",
    "prepare_tabular_features",
    "train_catboost_cv",
    "train_final_mlp_on_sequence_embeddings_cv",
    "train_meta_catboost",
    "train_mlp_on_embeddings",
    "train_mlp_on_sequence_embeddings_cv",
    "train_sequence_model",
    "train_transformer",
    "train_xlstm",
]
