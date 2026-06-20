import inspect
import json
from pathlib import Path
from pprint import pformat

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import CAT_COLS, TrainingConfig
from features_creator import extract_client_features
from models import CreditBiLSTM, CreditTransformer, CreditXLSTM, CrossAttentionFusionMLP
from train_catboost import fit_catboost_fold
from train_sequence_models import (
    build_sequences,
    extract_sequence_embeddings,
    make_sequence_fold_data,
    predict_cross_attention_fusion_mlp,
    resolve_device,
    train_cross_attention_fusion_mlp,
    train_sequence_model,
)
from utils import optimize_data_types


def create_transformer_model():
    return CreditTransformer()


def create_xlstm_model():
    return CreditXLSTM()


def create_bilstm_model():
    return CreditBiLSTM()


def create_cross_attention_fusion_model(d_tr, d_lstm):
    return CrossAttentionFusionMLP(d_tr=d_tr, d_lstm=d_lstm)


def create_catboost_model(seed=42, train_dir=None, scale_pos_weight=1.0):
    return CatBoostClassifier(
        iterations=5000,
        depth=8,
        learning_rate=0.02,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        scale_pos_weight=scale_pos_weight,
        verbose=100,
        task_type="GPU",
        train_dir=str(train_dir) if train_dir else None,
    )


def create_meta_catboost_model(seed=42, train_dir=None, scale_pos_weight=1.0):
    return CatBoostClassifier(
        iterations=1000,
        depth=4,
        learning_rate=0.02,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        scale_pos_weight=scale_pos_weight,
        verbose=100,
        train_dir=str(train_dir) if train_dir else None,
    )


def build_static_features_v1(train_data, test_data, train_target):
    train_features = extract_client_features(train_data, version=1)[0]
    test_features = extract_client_features(test_data, version=1)[0]

    train_features = optimize_data_types(train_features)
    test_features = optimize_data_types(test_features)

    train_df = (
        train_features.merge(train_target[["id", "flag"]], on="id", how="inner")
        .sort_values("id")
        .reset_index(drop=True)
    )
    test_df = test_features.sort_values("id").reset_index(drop=True)
    return train_df, test_df


def load_data(config):
    data_cfg = config["data"]
    return {
        "train_data": pd.read_parquet(data_cfg["train_data_path"]),
        "test_data": pd.read_parquet(data_cfg["test_data_path"]),
        "target": pd.read_csv(data_cfg["target_path"]),
    }


def main_train(config):
    output_dir = Path(config["data"]["output_dir"])
    predictions_dir = output_dir / "predictions"
    models_dir = output_dir / "models"
    train_dirs_dir = output_dir / "catboost_train_dirs"

    predictions_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    train_dirs_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(config)
    train_target_df = data["target"][["id", "flag"]].sort_values("id").reset_index(drop=True)
    test_id_df = (
        data["test_data"][["id"]].drop_duplicates().sort_values("id").reset_index(drop=True)
    )

    static_train_df = None
    static_test_df = None
    if config.get("static_features", {}).get("enabled", True):
        builder = config["static_features"].get("builder", build_static_features_v1)
        static_train_df, static_test_df = builder(
            data["train_data"],
            data["test_data"],
            data["target"],
        )

    all_oof = []
    all_test = []
    runner_results = []

    total_models = len(config["models"])
    for model_idx, model_cfg in enumerate(config["models"], start=1):
        print("=" * 80)
        print(
            f"Starting model {model_idx}/{total_models}: {model_cfg['name']} "
            f"({total_models - model_idx} remaining)"
        )
        print("Model config:")
        print(pformat(model_cfg, sort_dicts=False))
        print("=" * 80)

        runner = ModelRunner(
            model_cfg=model_cfg,
            output_dir=output_dir,
            static_train_df=static_train_df,
            static_test_df=static_test_df,
            train_target_df=train_target_df,
            test_id_df=test_id_df,
            train_data_path=config["data"]["train_data_path"],
            test_data_path=config["data"]["test_data_path"],
        )
        result = runner.run()
        runner_results.append(result)
        all_oof.append(result.oof_predictions)
        all_test.append(result.test_predictions)

        result.oof_predictions.to_csv(predictions_dir / f"{result.name}_oof.csv", index=False)
        result.test_predictions.to_csv(predictions_dir / f"{result.name}_test.csv", index=False)

    oof_predictions = pd.concat(all_oof, ignore_index=True)
    test_predictions = pd.concat(all_test, ignore_index=True)
    oof_predictions.to_csv(predictions_dir / "oof_predictions.csv", index=False)
    test_predictions.to_csv(predictions_dir / "test_predictions.csv", index=False)

    meta_result = None
    meta_cfg = config.get("meta_model", {})
    if meta_cfg.get("enabled", True) and len(runner_results) >= meta_cfg.get("min_models", 2):
        meta_result = train_meta_model(
            runner_results=runner_results,
            train_target_df=train_target_df,
            test_ids=test_id_df["id"].astype(int).values,
            config=config,
            output_dir=output_dir,
        )

    return {
        "model_results": runner_results,
        "oof_predictions": oof_predictions,
        "test_predictions": test_predictions,
        "meta_result": meta_result,
    }


class RunnerResult:
    def __init__(self, name, oof_predictions, test_predictions, scores):
        self.name = name
        self.oof_predictions = oof_predictions
        self.test_predictions = test_predictions
        self.scores = scores
        self.mean_auc = float(np.mean(scores))
        self.std_auc = float(np.std(scores))


class ModelRunner:
    def __init__(
        self,
        model_cfg,
        output_dir,
        static_train_df,
        static_test_df,
        train_target_df,
        test_id_df,
        train_data_path,
        test_data_path,
    ):
        self.model_cfg = model_cfg
        self.name = model_cfg["name"]
        self.output_dir = Path(output_dir)
        self.static_train_df = static_train_df
        self.static_test_df = static_test_df
        self.train_target_df = train_target_df
        self.test_id_df = test_id_df
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def run(self):
        if self.model_cfg["type"] == "static_catboost":
            return self._run_static_catboost()
        if self.model_cfg["type"] == "neural_embeddings_catboost":
            return self._run_neural_embeddings_catboost()
        if self.model_cfg["type"] == "orthogonalized_embeddings_catboost":
            return self._run_orthogonalized_embeddings_catboost()
        if self.model_cfg["type"] == "cross_attention_fusion_mlp":
            return self._run_cross_attention_fusion_mlp()
        raise ValueError(f"Unknown model type: {self.model_cfg['type']}")

    def _training_config(self):
        return TrainingConfig(
            n_splits=self.model_cfg.get("n_splits", 5),
            random_state=self.model_cfg.get("seed", 42),
            catboost_iterations=self.model_cfg.get("catboost_iterations", 5000),
            catboost_depth=self.model_cfg.get("catboost_depth", 8),
            catboost_learning_rate=self.model_cfg.get("catboost_learning_rate", 0.02),
            transformer_epochs=self.model_cfg.get("sequence_epochs", 15),
            mlp_epochs=self.model_cfg.get("mlp_epochs", 40),
            batch_size_sequences=self.model_cfg.get("batch_size_sequences", 256),
            batch_size_embeddings=self.model_cfg.get("batch_size_embeddings", 4096),
            device=self.model_cfg.get("device"),
        )

    def _run_static_catboost(self):
        self._require_static_features()
        cfg = self._training_config()
        train_df = self.static_train_df.copy()
        test_df = self.static_test_df.copy()

        X = train_df.drop(columns=["id", "flag"])
        y = train_df["flag"].astype(int).reset_index(drop=True)
        X_test = test_df.drop(columns=["id"])

        print(
            f"{self.name}: starting OOF training | "
            f"train_rows={len(X)}, test_rows={len(X_test)}, "
            f"features={X.shape[1]}, folds={cfg.n_splits}"
        )

        return _run_catboost_on_feature_matrix(
            name=self.name,
            X=X,
            y=y,
            X_test=X_test,
            train_ids=train_df["id"].astype(int).values,
            test_ids=test_df["id"].astype(int).values,
            output_dir=self.output_dir,
            config=cfg,
            model_factory=self.model_cfg.get("final_model_factory", create_catboost_model),
        )

    def _run_neural_embeddings_catboost(self):
        cfg = self._training_config()
        device = resolve_device(cfg.device)

        train_sequences, test_sequences = build_sequences(
            self.train_data_path,
            self.test_data_path,
            chunk_size=self.model_cfg.get("sequence_chunk_size", 500_000),
        )

        train_df = self._base_train_frame()
        test_df = self._base_test_frame()
        y = train_df["flag"].astype(int).reset_index(drop=True)
        ids = train_df["id"].astype(int).values
        test_ids = test_df["id"].astype(int).values

        skf = StratifiedKFold(
            n_splits=cfg.n_splits,
            shuffle=True,
            random_state=cfg.random_state,
        )

        oof = np.zeros(len(train_df), dtype=np.float32)
        test_pred = np.zeros(len(test_df), dtype=np.float32)
        scores = []
        id_to_pos = pd.Series(np.arange(len(ids)), index=ids)

        print(
            f"{self.name}: starting OOF training | "
            f"train_clients={len(train_df)}, test_clients={len(test_df)}, "
            f"train_sequences={len(train_sequences)}, test_sequences={len(test_sequences)}, "
            f"folds={cfg.n_splits}"
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y), start=1):
            fold_dir = self.output_dir / "models" / self.name / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_ids = ids[train_idx]
            val_ids = ids[val_idx]
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

            seq_scale_pos_weight = _sequence_scale_pos_weight(train_fold_targets)
            print(f"seq_scale_pos_weight: {seq_scale_pos_weight}")
            sequence_model = self._train_sequence_model(
                train_fold_sequences=train_fold_sequences,
                train_fold_targets=train_fold_targets,
                val_fold_sequences=val_fold_sequences,
                val_fold_targets=val_fold_targets,
                scale_pos_weight=seq_scale_pos_weight,
                config=cfg,
                device=device,
            )

            embedding_prefix = self.model_cfg.get("embedding_prefix", self.name)
            train_emb = self._extract_embeddings(
                sequence_model,
                train_fold_sequences,
                prefix=embedding_prefix,
                device=device,
            )
            val_emb = self._extract_embeddings(
                sequence_model,
                val_fold_sequences,
                prefix=embedding_prefix,
                device=device,
            )
            test_emb = self._extract_embeddings(
                sequence_model,
                test_sequences,
                prefix=embedding_prefix,
                device=device,
            )

            X_train, X_val, X_test, y_train, y_val, val_ids_for_pred = self._build_final_features(
                train_emb=train_emb,
                val_emb=val_emb,
                test_emb=test_emb,
                train_ids=train_ids,
                val_ids=val_ids,
                y_train=y_train_raw,
                y_val=y_val_raw,
            )

            _, val_pred, fold_test_pred, score = fit_catboost_fold(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                model_path=fold_dir / "catboost.cbm",
                train_dir=fold_dir / "catboost_train_dir",
                config=cfg,
                task_type=self.model_cfg.get("task_type", "GPU"),
                cat_features=_cat_features_for(X_train),
                model_factory=self.model_cfg.get("final_model_factory", create_catboost_model),
            )

            positions = id_to_pos.loc[val_ids_for_pred].values
            oof[positions] = val_pred
            test_pred += fold_test_pred / cfg.n_splits
            scores.append(score)

            del sequence_model

        return _runner_result_from_arrays(
            name=self.name,
            ids=ids,
            y=y.values,
            test_ids=test_ids,
            oof=oof,
            test_pred=test_pred,
            scores=scores,
        )

    def _run_cross_attention_fusion_mlp(self):
        cfg = self._training_config()
        device = resolve_device(cfg.device)

        train_sequences, test_sequences = build_sequences(
            self.train_data_path,
            self.test_data_path,
            chunk_size=self.model_cfg.get("sequence_chunk_size", 500_000),
        )

        train_df = self._base_train_frame()
        test_df = self._base_test_frame()
        y = train_df["flag"].astype(int).reset_index(drop=True)
        ids = train_df["id"].astype(int).values
        test_ids = test_df["id"].astype(int).values

        skf = StratifiedKFold(
            n_splits=cfg.n_splits,
            shuffle=True,
            random_state=cfg.random_state,
        )

        oof = np.zeros(len(train_df), dtype=np.float32)
        test_pred = np.zeros(len(test_df), dtype=np.float32)
        scores = []
        id_to_pos = pd.Series(np.arange(len(ids)), index=ids)
        test_id_to_pos = pd.Series(np.arange(len(test_ids)), index=test_ids)

        print(
            f"{self.name}: starting OOF training | "
            f"train_clients={len(train_df)}, test_clients={len(test_df)}, "
            f"train_sequences={len(train_sequences)}, test_sequences={len(test_sequences)}, "
            f"folds={cfg.n_splits}"
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y), start=1):
            fold_dir = self.output_dir / "models" / self.name / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_ids = ids[train_idx]
            val_ids = ids[val_idx]
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

            scale_pos_weight = _sequence_scale_pos_weight(train_fold_targets)
            print(f"fusion sequence scale_pos_weight: {scale_pos_weight}")

            transformer_model = train_sequence_model(
                train_fold_sequences,
                train_fold_targets,
                val_fold_sequences,
                val_fold_targets,
                model_factory=self.model_cfg.get(
                    "transformer_model_factory",
                    create_transformer_model,
                ),
                epochs=cfg.transformer_epochs,
                scale_pos_weight=scale_pos_weight,
                batch_size=cfg.batch_size_sequences,
                device=device,
            )
            bilstm_model = train_sequence_model(
                train_fold_sequences,
                train_fold_targets,
                val_fold_sequences,
                val_fold_targets,
                model_factory=self.model_cfg.get("bilstm_model_factory", create_bilstm_model),
                epochs=cfg.transformer_epochs,
                scale_pos_weight=scale_pos_weight,
                batch_size=cfg.batch_size_sequences,
                device=device,
            )

            fusion_model, best_auc = train_cross_attention_fusion_mlp(
                transformer_model=transformer_model,
                bilstm_model=bilstm_model,
                train_sequences=train_fold_sequences,
                train_targets=train_fold_targets,
                val_sequences=val_fold_sequences,
                val_targets=val_fold_targets,
                fusion_model_factory=self.model_cfg.get(
                    "fusion_model_factory",
                    create_cross_attention_fusion_model,
                ),
                epochs=self.model_cfg.get("fusion_epochs", cfg.mlp_epochs),
                scale_pos_weight=scale_pos_weight,
                batch_size=cfg.batch_size_sequences,
                device=device,
            )

            val_pred, y_val_for_score = predict_cross_attention_fusion_mlp(
                transformer_model,
                bilstm_model,
                fusion_model,
                val_fold_sequences,
                targets=val_fold_targets,
                batch_size=cfg.batch_size_sequences,
                device=device,
            )
            fold_test_pred = predict_cross_attention_fusion_mlp(
                transformer_model,
                bilstm_model,
                fusion_model,
                test_sequences,
                targets=None,
                batch_size=cfg.batch_size_sequences,
                device=device,
            )

            val_ids_for_pred = np.array(list(val_fold_sequences.keys()), dtype=int)
            positions = id_to_pos.loc[val_ids_for_pred].values
            oof[positions] = val_pred

            fold_test_ids = np.array(list(test_sequences.keys()), dtype=int)
            test_positions = test_id_to_pos.loc[fold_test_ids].values
            test_pred[test_positions] += fold_test_pred / cfg.n_splits

            score = roc_auc_score(y_val_for_score, val_pred)
            scores.append(score)
            print(f"{self.name} fold {fold} fusion best_auc={best_auc:.6f}, auc={score:.6f}")

            torch.save(
                {
                    "transformer_state_dict": transformer_model.state_dict(),
                    "bilstm_state_dict": bilstm_model.state_dict(),
                    "fusion_state_dict": fusion_model.state_dict(),
                    "best_auc": best_auc,
                },
                fold_dir / "cross_attention_fusion_mlp.pt",
            )

            del transformer_model, bilstm_model, fusion_model
            torch.cuda.empty_cache()

        return _runner_result_from_arrays(
            name=self.name,
            ids=ids,
            y=y.values,
            test_ids=test_ids,
            oof=oof,
            test_pred=test_pred,
            scores=scores,
        )

    def _run_orthogonalized_embeddings_catboost(self):
        cfg = self._training_config()
        device = resolve_device(cfg.device)
        train_sequences, test_sequences = build_sequences(
            self.train_data_path, self.test_data_path,
            chunk_size=self.model_cfg.get("sequence_chunk_size", 500_000),
        )
        train_df, test_df = self._base_train_frame(), self._base_test_frame()
        y = train_df["flag"].astype(int).reset_index(drop=True)
        ids = train_df["id"].astype(int).values
        test_ids = test_df["id"].astype(int).values
        skf = StratifiedKFold(cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        oof = np.zeros(len(train_df), dtype=np.float32)
        test_pred = np.zeros(len(test_df), dtype=np.float32)
        scores = []
        id_to_pos = pd.Series(np.arange(len(ids)), index=ids)
        alpha = float(self.model_cfg.get("ridge_alpha", 100.0))
        gamma = float(self.model_cfg.get("residual_gamma", 1.0))
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"residual_gamma must be in [0, 1], got {gamma}")
        print(f"{self.name}: ridge OOF | folds={cfg.n_splits}, alpha={alpha}, gamma={gamma}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y), 1):
            fold_dir = self.output_dir / "models" / self.name / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_ids, val_ids = ids[train_idx], ids[val_idx]
            y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
            train_seq, train_targets, val_seq, val_targets = make_sequence_fold_data(
                train_ids, val_ids, y_train, y_val, train_sequences
            )
            scale_pos_weight = _sequence_scale_pos_weight(train_targets)
            train_kwargs = dict(
                train_sequences=train_seq, train_targets=train_targets,
                val_sequences=val_seq, val_targets=val_targets,
                epochs=cfg.transformer_epochs, scale_pos_weight=scale_pos_weight,
                batch_size=cfg.batch_size_sequences, device=device,
            )
            transformer = train_sequence_model(
                **train_kwargs,
                model_factory=self.model_cfg.get(
                    "transformer_model_factory", create_transformer_model
                ),
            )
            bilstm = train_sequence_model(
                **train_kwargs,
                model_factory=self.model_cfg.get("bilstm_model_factory", create_bilstm_model),
            )
            batch_size = self.model_cfg.get("embedding_batch_size", 2048)
            embeddings = {}
            for name, sequences, expected_ids in (
                ("train", train_seq, train_ids), ("val", val_seq, val_ids),
                ("test", test_sequences, test_ids),
            ):
                tr = extract_sequence_embeddings(
                    transformer, sequences, prefix="transformer_emb",
                    batch_size=batch_size, device=device,
                )
                lstm = extract_sequence_embeddings(
                    bilstm, sequences, prefix="bilstm_emb",
                    batch_size=batch_size, device=device,
                )
                embeddings[name] = _align_paired_embeddings(tr, lstm, expected_ids, name)

            tr_scaler, lstm_scaler = StandardScaler(), StandardScaler()
            tr_train = tr_scaler.fit_transform(embeddings["train"][0])
            lstm_train = lstm_scaler.fit_transform(embeddings["train"][1])
            tr_val = tr_scaler.transform(embeddings["val"][0])
            lstm_val = lstm_scaler.transform(embeddings["val"][1])
            tr_test = tr_scaler.transform(embeddings["test"][0])
            lstm_test = lstm_scaler.transform(embeddings["test"][1])
            ridge = Ridge(alpha=alpha).fit(tr_train, lstm_train)
            pred_train, pred_val, pred_test = (
                ridge.predict(tr_train), ridge.predict(tr_val), ridge.predict(tr_test)
            )
            res_train = lstm_train - gamma * pred_train
            res_val = lstm_val - gamma * pred_val
            res_test = lstm_test - gamma * pred_test
            X_train = _make_orthogonalized_feature_frame(tr_train, res_train)
            X_val = _make_orthogonalized_feature_frame(tr_val, res_val)
            X_test = _make_orthogonalized_feature_frame(tr_test, res_test)
            _, val_pred, fold_test_pred, score = fit_catboost_fold(
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                X_test=X_test, model_path=fold_dir / "catboost.cbm",
                train_dir=fold_dir / "catboost_train_dir", config=cfg,
                task_type=self.model_cfg.get("task_type", "GPU"), cat_features=[],
                model_factory=self.model_cfg.get("final_model_factory", create_catboost_model),
            )
            oof[id_to_pos.loc[embeddings["val"][2]].values] = val_pred
            test_pred += fold_test_pred / cfg.n_splits
            scores.append(score)
            torch.save({
                "transformer_state_dict": _state_dict_to_cpu(transformer),
                "bilstm_state_dict": _state_dict_to_cpu(bilstm),
            }, fold_dir / "sequence_models.pt")
            joblib.dump(tr_scaler, fold_dir / "transformer_scaler.joblib")
            joblib.dump(lstm_scaler, fold_dir / "bilstm_scaler.joblib")
            joblib.dump(ridge, fold_dir / "ridge.joblib")
            metrics = {
                "fold": fold, "alpha": alpha, "gamma": gamma, "fold_auc": float(score),
                "explained_variance": {
                    "train": float(explained_variance_score(
                        lstm_train, pred_train, multioutput="variance_weighted")),
                    "val": float(explained_variance_score(
                        lstm_val, pred_val, multioutput="variance_weighted")),
                },
                "mean_l2_norm": {
                    "train": _embedding_norm_metrics(
                        embeddings["train"], tr_train, lstm_train, res_train
                    ),
                    "val": _embedding_norm_metrics(embeddings["val"], tr_val, lstm_val, res_val),
                    "test": _embedding_norm_metrics(
                        embeddings["test"], tr_test, lstm_test, res_test
                    ),
                },
                "n_train": len(X_train), "n_val": len(X_val),
                "n_test": len(X_test), "n_features": X_train.shape[1],
            }
            with (fold_dir / "metrics.json").open("w", encoding="utf-8") as file:
                json.dump(metrics, file, ensure_ascii=False, indent=2)
            print(f"{self.name} fold {fold}: auc={score:.6f}, "
                  f"EV train={metrics['explained_variance']['train']:.6f}, "
                  f"EV val={metrics['explained_variance']['val']:.6f}")
            del transformer, bilstm
            torch.cuda.empty_cache()

        return _runner_result_from_arrays(
            self.name, ids, y.values, test_ids, oof, test_pred, scores
        )

    def _train_sequence_model(
        self,
        train_fold_sequences,
        train_fold_targets,
        val_fold_sequences,
        val_fold_targets,
        scale_pos_weight,
        config,
        device,
    ):
        train_fn = self.model_cfg.get("train_sequence_model_fn")
        model_factory = self.model_cfg.get("sequence_model_factory", create_transformer_model)

        if train_fn is None:
            return train_sequence_model(
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

        kwargs = {
            "epochs": config.transformer_epochs,
            "scale_pos_weight": scale_pos_weight,
            "batch_size": config.batch_size_sequences,
            "device": device,
            "model_factory": model_factory,
        }
        return _call_with_supported_kwargs(
            train_fn,
            train_fold_sequences,
            train_fold_targets,
            val_fold_sequences,
            val_fold_targets,
            **kwargs,
        )

    def _extract_embeddings(self, model, sequences, prefix, device):
        extract_fn = self.model_cfg.get("extract_features_fn", extract_sequence_embeddings)
        return _call_with_supported_kwargs(
            extract_fn,
            model,
            sequences,
            prefix=prefix,
            batch_size=self.model_cfg.get("embedding_batch_size", 1024),
            device=device,
        )

    def _build_final_features(
        self, train_emb, val_emb, test_emb, train_ids, val_ids, y_train, y_val
    ):
        train_target_df = pd.DataFrame({"id": train_ids.astype(int), "target": y_train})
        val_target_df = pd.DataFrame({"id": val_ids.astype(int), "target": y_val})

        train_df = train_emb.merge(train_target_df, on="id", how="inner")
        val_df = val_emb.merge(val_target_df, on="id", how="inner")
        test_df = self._base_test_frame()[["id"]].merge(test_emb, on="id", how="left")

        if self.model_cfg.get("use_static_features", False):
            self._require_static_features()
            static_cols = [c for c in self.static_train_df.columns if c not in ("id", "flag")]
            train_df = train_df.merge(
                self.static_train_df[["id"] + static_cols],
                on="id",
                how="left",
            )
            val_df = val_df.merge(
                self.static_train_df[["id"] + static_cols],
                on="id",
                how="left",
            )
            test_df = test_df.merge(
                self.static_test_df[["id"] + static_cols],
                on="id",
                how="left",
            )

        feature_cols = [c for c in train_df.columns if c not in ("id", "target")]
        return (
            train_df[feature_cols].copy(),
            val_df[feature_cols].copy(),
            test_df[feature_cols].copy(),
            train_df["target"].values,
            val_df["target"].values,
            val_df["id"].astype(int).values,
        )

    def _require_static_features(self):
        if self.static_train_df is None or self.static_test_df is None:
            raise ValueError(f"{self.name} requires static features")

    def _base_train_frame(self):
        return self.train_target_df[["id", "flag"]].copy()

    def _base_test_frame(self):
        return self.test_id_df[["id"]].copy()


def _align_paired_embeddings(transformer_df, bilstm_df, expected_ids, split_name):
    expected = pd.DataFrame({"id": np.asarray(expected_ids, dtype=int)})
    if expected["id"].duplicated().any():
        raise ValueError(f"Duplicate expected ids in {split_name}")
    if transformer_df["id"].duplicated().any() or bilstm_df["id"].duplicated().any():
        raise ValueError(f"Duplicate embedding ids in {split_name}")
    tr_cols = [col for col in transformer_df if col != "id"]
    lstm_cols = [col for col in bilstm_df if col != "id"]
    paired = expected.merge(transformer_df, on="id", how="left", validate="one_to_one")
    paired = paired.merge(bilstm_df, on="id", how="left", validate="one_to_one")
    if paired[tr_cols + lstm_cols].isna().to_numpy().any():
        missing = paired.loc[paired[tr_cols + lstm_cols].isna().any(axis=1), "id"]
        raise ValueError(f"Missing {split_name} embeddings for ids: {missing.head(10).tolist()}")
    return (paired[tr_cols].to_numpy(np.float32), paired[lstm_cols].to_numpy(np.float32),
            paired["id"].to_numpy(int))


def _make_orthogonalized_feature_frame(transformer, residual):
    columns = ([f"transformer_scaled_{i}" for i in range(transformer.shape[1])] +
               [f"bilstm_residual_{i}" for i in range(residual.shape[1])])
    return pd.DataFrame(np.concatenate([transformer, residual], axis=1).astype(np.float32),
                        columns=columns)


def _mean_l2_norm(values):
    return float(np.linalg.norm(values, axis=1).mean())


def _embedding_norm_metrics(raw_pair, transformer, bilstm, residual):
    return {"transformer_raw": _mean_l2_norm(raw_pair[0]),
            "bilstm_raw": _mean_l2_norm(raw_pair[1]),
            "transformer_scaled": _mean_l2_norm(transformer),
            "bilstm_scaled": _mean_l2_norm(bilstm),
            "bilstm_residual": _mean_l2_norm(residual)}


def _state_dict_to_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_meta_model(runner_results, train_target_df, test_ids, config, output_dir):
    meta_cfg = config.get("meta_model", {})
    output_dir = Path(output_dir)

    train_wide = train_target_df[["id", "flag"]].copy()
    test_wide = pd.DataFrame({"id": test_ids})

    for result in runner_results:
        oof_part = result.oof_predictions[["id", "prediction"]].rename(
            columns={"prediction": result.name}
        )
        test_part = result.test_predictions[["id", "prediction"]].rename(
            columns={"prediction": result.name}
        )
        train_wide = train_wide.merge(oof_part, on="id", how="left")
        test_wide = test_wide.merge(test_part, on="id", how="left")

    model_cols = [result.name for result in runner_results]
    X = train_wide[model_cols]
    y = train_wide["flag"].astype(int)
    X_test = test_wide[model_cols]

    pos_rate = y.mean()
    scale_pos_weight = ((1 - pos_rate) / pos_rate) ** 0.4
    model_factory = meta_cfg.get("model_factory", create_meta_catboost_model)
    model = model_factory(
        seed=meta_cfg.get("seed", config["models"][0].get("seed", 42)),
        train_dir=output_dir / "meta_model_train_dir",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X, y)

    test_pred = model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({"id": test_ids, "flag": test_pred})

    meta_dir = output_dir / "meta_model"
    meta_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(meta_dir / "meta_catboost.cbm"))
    train_wide.to_csv(meta_dir / "meta_train_features.csv", index=False)
    test_wide.to_csv(meta_dir / "meta_test_features.csv", index=False)
    submission.to_csv(meta_dir / "submission_meta.csv", index=False)

    return {
        "model": model,
        "train_features": train_wide,
        "test_features": test_wide,
        "submission": submission,
    }


def _run_catboost_on_feature_matrix(
    name,
    X,
    y,
    X_test,
    train_ids,
    test_ids,
    output_dir,
    config,
    model_factory,
):
    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    oof = np.zeros(len(X), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_dir = Path(output_dir) / "models" / name / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        _, val_pred, fold_test_pred, score = fit_catboost_fold(
            X_train=X.iloc[train_idx].copy(),
            y_train=y.iloc[train_idx],
            X_val=X.iloc[val_idx].copy(),
            y_val=y.iloc[val_idx],
            X_test=X_test.copy(),
            model_path=fold_dir / "catboost.cbm",
            train_dir=fold_dir / "catboost_train_dir",
            config=config,
            cat_features=_cat_features_for(X),
            model_factory=model_factory,
        )

        oof[val_idx] = val_pred
        test_pred += fold_test_pred / config.n_splits
        scores.append(score)

    return _runner_result_from_arrays(
        name=name,
        ids=train_ids,
        y=y.values,
        test_ids=test_ids,
        oof=oof,
        test_pred=test_pred,
        scores=scores,
    )


def _runner_result_from_arrays(name, ids, y, test_ids, oof, test_pred, scores):
    oof_df = pd.DataFrame(
        {
            "id": ids,
            "target": y,
            "model": name,
            "prediction": oof,
        }
    )
    test_df = pd.DataFrame(
        {
            "id": test_ids,
            "model": name,
            "prediction": test_pred,
        }
    )

    print(f"{name} fold AUC mean={np.mean(scores):.6f}, std={np.std(scores):.6f}")
    print(f"{name} OOF AUC={roc_auc_score(y, oof):.6f}")

    return RunnerResult(
        name=name,
        oof_predictions=oof_df,
        test_predictions=test_df,
        scores=scores,
    )


def _sequence_scale_pos_weight(train_fold_targets):
    y = np.array(list(train_fold_targets.values()), dtype=np.float32)
    num_pos = y.sum()
    num_neg = len(y) - num_pos
    return min(np.sqrt(num_neg / max(num_pos, 1)), 6.0)


def _cat_features_for(X):
    return [
        col
        for col in X.columns
        if col.startswith("last") and any(cat_col in col for cat_col in CAT_COLS)
    ]


def _call_with_supported_kwargs(fn, *args, **kwargs):
    signature = inspect.signature(fn)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return fn(*args, **kwargs)

    supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(*args, **supported_kwargs)
