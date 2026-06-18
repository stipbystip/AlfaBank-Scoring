import numpy as np
import pandas as pd


def _uint_type_for(value):
    if value < np.iinfo(np.uint8).max:
        return np.uint8
    if value < np.iinfo(np.uint16).max:
        return np.uint16
    if value < np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _int_type_for(value):
    if value < np.iinfo(np.int8).max:
        return np.int8
    if value < np.iinfo(np.int16).max:
        return np.int16
    if value < np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def optimize_data_types(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not pd.api.types.is_categorical_dtype(col_type):
            val_min = df[col].min()
            val_max = df[col].max()

            if str(col_type).startswith("int"):
                max_abs = max(abs(val_min), abs(val_max))
                if val_min >= 0:
                    df[col] = df[col].astype(_uint_type_for(max_abs))
                else:
                    df[col] = df[col].astype(_int_type_for(max_abs))

        if "float" in str(col_type):
            df[col] = df[col].astype(np.float32)

    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB")

    return df
