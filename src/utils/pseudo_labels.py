from pathlib import Path

import pandas as pd


def make_train_with_pseudo(df_train, df_test, pseudo_df, id_col="id"):
    pseudo_col = "pseudo_target" if "pseudo_target" in pseudo_df.columns else "flag"
    pseudo_part = df_test.merge(
        pseudo_df[[id_col, pseudo_col]],
        on=id_col,
        how="inner",
    )
    pseudo_part["flag"] = pseudo_part[pseudo_col].astype(df_train["flag"].dtype)
    if pseudo_col != "flag":
        pseudo_part = pseudo_part.drop(columns=[pseudo_col])
    pseudo_part = pseudo_part[df_train.columns]

    df_train_new = pd.concat([df_train, pseudo_part], ignore_index=True)
    return df_train_new, pseudo_part


def create_pseudo_train_parquet(
    train_data_path: str,
    test_data_path: str,
    train_target_path: str,
    pseudo_target_path: str,
    output_train_data_path: str,
    output_train_target_path: str,
    id_col: str = "id",
    target_col: str = "flag",
) -> tuple[Path, Path]:
    output_train_data_path = Path(output_train_data_path)
    output_train_target_path = Path(output_train_target_path)
    output_train_data_path.parent.mkdir(parents=True, exist_ok=True)
    output_train_target_path.parent.mkdir(parents=True, exist_ok=True)

    train_data = pd.read_parquet(train_data_path)
    test_data = pd.read_parquet(test_data_path)
    train_target = pd.read_csv(train_target_path)
    pseudo_target = pd.read_csv(pseudo_target_path)

    required_target_cols = {id_col, target_col}
    if not required_target_cols.issubset(train_target.columns):
        raise ValueError(
            f"train_target must contain {required_target_cols}; got {train_target.columns.tolist()}"
        )
    if not required_target_cols.issubset(pseudo_target.columns):
        raise ValueError(
            f"pseudo_target must contain {required_target_cols}; got {pseudo_target.columns.tolist()}"
        )
    if id_col not in train_data.columns:
        raise ValueError(f"train_data does not contain column {id_col}")
    if id_col not in test_data.columns:
        raise ValueError(f"test_data does not contain column {id_col}")

    pseudo_ids = set(pseudo_target[id_col].unique())
    pseudo_test_data = test_data[test_data[id_col].isin(pseudo_ids)].copy()
    missing_ids = pseudo_ids - set(pseudo_test_data[id_col].unique())
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} pseudo ids were not found in test_data")

    pseudo_train_data = pd.concat(
        [train_data, pseudo_test_data],
        axis=0,
        ignore_index=True,
    )

    sort_cols = [id_col]
    if "rn" in pseudo_train_data.columns:
        sort_cols.append("rn")
    pseudo_train_data = pseudo_train_data.sort_values(sort_cols).reset_index(drop=True)

    pseudo_train_target = pd.concat(
        [
            train_target[[id_col, target_col]],
            pseudo_target[[id_col, target_col]],
        ],
        axis=0,
        ignore_index=True,
    )
    pseudo_train_target = pseudo_train_target.drop_duplicates(
        subset=[id_col],
        keep="first",
    ).reset_index(drop=True)

    pseudo_train_data.to_parquet(output_train_data_path, index=False)
    pseudo_train_target.to_csv(output_train_target_path, index=False)

    return output_train_data_path, output_train_target_path
