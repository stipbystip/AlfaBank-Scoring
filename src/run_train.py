from main_train import (
    build_static_features_v1,
    create_bilstm_model,
    create_catboost_model,
    create_cross_attention_fusion_model,
    create_meta_catboost_model,
    create_transformer_model,
    create_xlstm_model,
    extract_sequence_embeddings,
    main_train,
)

CONFIG = {
    "data": {
        "train_data_path": "stepan/data/train_data_pseudo.parquet",
        "test_data_path": "stepan/data/test_data.parquet",
        "target_path": "stepan/data/train_target_pseudo.csv",
        "output_dir": "stepan/data/exp_002",
    },
    "static_features": {
        "enabled": False,
        "builder": build_static_features_v1,
    },
    "models": [
        {
            "name": "transformer_bilstm_cross_attention_mlp_seed42",
            "type": "cross_attention_fusion_mlp",
            "seed": 42,
            "n_splits": 5,
            "sequence_epochs": 10,
            "fusion_epochs": 15,
            "transformer_model_factory": create_transformer_model,
            "bilstm_model_factory": create_bilstm_model,
            "fusion_model_factory": create_cross_attention_fusion_model,
        },
        {
            "name": "bilstm_catboost_seed43",
            "type": "neural_embeddings_catboost",
            "seed": 43,
            "n_splits": 5,
            "sequence_epochs": 10,
            "sequence_model_factory": create_bilstm_model,
            "train_sequence_model_fn": None,
            "extract_features_fn": extract_sequence_embeddings,
            "final_model_factory": create_catboost_model,
            "use_static_features": False,
            "embedding_prefix": "bilstm_emb",
        },
        {
            "name": "transformer_catboost_seed44",
            "type": "neural_embeddings_catboost",
            "seed": 44,
            "n_splits": 5,
            "sequence_epochs": 10,
            "sequence_model_factory": create_transformer_model,
            "train_sequence_model_fn": None,
            "extract_features_fn": extract_sequence_embeddings,
            "final_model_factory": create_catboost_model,
            "use_static_features": False,
            "embedding_prefix": "transformer_emb",
        },
        {
            "name": "xlstm_catboost_seed42",
            "type": "neural_embeddings_catboost",
            "seed": 42,
            "n_splits": 10,
            "sequence_epochs": 2,
            "sequence_model_factory": create_xlstm_model,
            "train_sequence_model_fn": None,
            "extract_features_fn": extract_sequence_embeddings,
            "final_model_factory": create_catboost_model,
            "use_static_features": False,
            "embedding_prefix": "xlstm_emb",
        },
    ],
    "meta_model": {
        "enabled": True,
        "min_models": 2,
        "model_factory": create_meta_catboost_model,
    },
}


if __name__ == "__main__":
    main_train(CONFIG)
