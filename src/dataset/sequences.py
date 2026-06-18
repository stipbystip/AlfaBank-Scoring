import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import FEATURE_COLS, SEQUENCE_MAX_LENGTH


def pq_chunk_reader(file_path, cols, chunk_size=1_000_000):
    parquet_file = pq.ParquetFile(file_path)
    leftover_df = pd.DataFrame()
    total_rows = parquet_file.metadata.num_rows
    total_batches = math.ceil(total_rows / chunk_size)

    for batch in tqdm(
        parquet_file.iter_batches(batch_size=chunk_size, columns=cols),
        total=total_batches,
        desc="Reading parquet batches",
    ):
        batch_df = batch.to_pandas()
        if batch_df.empty:
            continue

        if not leftover_df.empty:
            batch_df = pd.concat([leftover_df, batch_df], ignore_index=True)
            leftover_df = pd.DataFrame()

        last_id = batch_df["id"].iloc[-1]
        mask = batch_df["id"] == last_id
        first_same_id_idx = mask.idxmax()
        if first_same_id_idx != batch_df.index[0]:
            leftover_df = batch_df.loc[first_same_id_idx:].copy()
            batch_df = batch_df.iloc[:first_same_id_idx]

        if not batch_df.empty:
            yield batch_df

    if not leftover_df.empty:
        yield leftover_df


def create_sequence(file_path, chunk_size=1_000_000):
    columns = ["id", "rn"] + FEATURE_COLS
    sequences = {}

    for chunk_df in pq_chunk_reader(file_path, columns, chunk_size):
        if chunk_df.empty:
            continue

        ids = chunk_df["id"].to_numpy()
        features_matrix = chunk_df[FEATURE_COLS].to_numpy(dtype=np.int64) + 1
        split_indices = np.where(ids[:-1] != ids[1:])[0] + 1
        split_features = np.split(features_matrix, split_indices)
        unique_ids = ids[np.concatenate(([0], split_indices))]

        for client_id, seq in zip(unique_ids, split_features, strict=False):
            if len(seq) > SEQUENCE_MAX_LENGTH:
                seq = seq[-SEQUENCE_MAX_LENGTH:]
            sequences[int(client_id)] = seq

    return sequences


class CreditDataset(Dataset):
    def __init__(self, sequences, targets=None, is_train=True):
        self.sequences = sequences
        self.targets = targets
        self.is_train = is_train
        self.client_ids = list(sequences.keys())

    def __getitem__(self, idx):
        client_id = self.client_ids[idx]
        sequence = self.sequences[client_id]

        if len(sequence) > SEQUENCE_MAX_LENGTH:
            sequence = sequence[-SEQUENCE_MAX_LENGTH:]

        sequence_length = len(sequence)
        if sequence_length == 0:
            raise ValueError(f"Empty sequence for client_id={client_id}")

        padded_sequence = np.zeros(
            (SEQUENCE_MAX_LENGTH, len(FEATURE_COLS)),
            dtype=np.int64,
        )
        padded_sequence[:sequence_length, :] = sequence

        padding_mask = np.ones(SEQUENCE_MAX_LENGTH, dtype=np.bool_)
        padding_mask[:sequence_length] = False

        item = {
            "client_id": client_id,
            "sequence": torch.from_numpy(padded_sequence),
            "padding_mask": torch.from_numpy(padding_mask),
        }

        if self.is_train and self.targets is not None:
            item["target"] = torch.tensor(
                self.targets[client_id],
                dtype=torch.float32,
            )

        return item

    def __len__(self):
        return len(self.client_ids)
