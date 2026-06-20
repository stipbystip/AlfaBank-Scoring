import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CreditDataset, create_sequence
from models import CreditTransformer, CreditXLSTM, CrossAttentionFusionMLP


def resolve_device(device=None):
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_sequence_model(
    train_sequences,
    train_targets,
    val_sequences,
    val_targets,
    model_factory=CreditTransformer,
    epochs=5,
    scale_pos_weight=1.0,
    patience=2,
    min_delta=1e-5,
    batch_size=512,
    device=None,
):
    device = resolve_device(device)
    train_dataset = CreditDataset(train_sequences, targets=train_targets, is_train=True)
    val_dataset = CreditDataset(val_sequences, targets=val_targets, is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = model_factory().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scale_pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    best_auc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc="Training sequence model"):
            sequences = batch["sequence"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True).view(-1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(sequences, padding_mask)["logits"].view(-1)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise ValueError("NaN/Inf in train logits")

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        val_auc = evaluate_sequence_model(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={np.mean(train_losses):.5f} | val_auc={val_auc:.5f}"
        )

        if val_auc > best_auc + min_delta:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best val_auc={best_auc:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_transformer(*args, **kwargs):
    return train_sequence_model(*args, model_factory=CreditTransformer, **kwargs)


def train_xlstm(*args, **kwargs):
    return train_sequence_model(*args, model_factory=CreditXLSTM, **kwargs)


def train_cross_attention_fusion_mlp(
    transformer_model,
    bilstm_model,
    train_sequences,
    train_targets,
    val_sequences,
    val_targets,
    fusion_model_factory=CrossAttentionFusionMLP,
    epochs=20,
    scale_pos_weight=1.0,
    patience=5,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-3,
    device=None,
):
    device = resolve_device(device)
    transformer_model = transformer_model.to(device).eval()
    bilstm_model = bilstm_model.to(device).eval()

    d_tr = transformer_model.embedding_dim
    d_lstm = bilstm_model.embedding_dim
    fusion_model = fusion_model_factory(d_tr=d_tr, d_lstm=d_lstm).to(device)

    train_loader = DataLoader(
        CreditDataset(train_sequences, targets=train_targets, is_train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        CreditDataset(val_sequences, targets=val_targets, is_train=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scale_pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_auc = -np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        fusion_model.train()
        losses = []

        for batch in tqdm(train_loader, desc="Training cross-attention fusion MLP"):
            sequences = batch["sequence"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True).view(-1)

            with torch.no_grad():
                tr_states = transformer_model.get_hidden_states(sequences, padding_mask)
                lstm_states = bilstm_model.get_hidden_states(sequences, padding_mask)

            optimizer.zero_grad(set_to_none=True)
            logits = fusion_model(tr_states, lstm_states, padding_mask)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        val_pred, val_target = predict_cross_attention_fusion_mlp(
            transformer_model,
            bilstm_model,
            fusion_model,
            val_sequences,
            targets=val_targets,
            batch_size=batch_size,
            device=device,
        )
        val_auc = roc_auc_score(val_target, val_pred)
        scheduler.step(val_auc)
        print(f"Fusion epoch {epoch:02d} | loss={np.mean(losses):.5f} | val_auc={val_auc:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(fusion_model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Fusion early stopping. Best AUC: {best_auc:.6f}")
                break

    if best_state is not None:
        fusion_model.load_state_dict(best_state)

    return fusion_model, best_auc


@torch.no_grad()
def predict_cross_attention_fusion_mlp(
    transformer_model,
    bilstm_model,
    fusion_model,
    sequences,
    targets=None,
    batch_size=512,
    device=None,
):
    device = resolve_device(device)
    transformer_model = transformer_model.to(device).eval()
    bilstm_model = bilstm_model.to(device).eval()
    fusion_model = fusion_model.to(device).eval()

    dataloader = DataLoader(
        CreditDataset(sequences, targets=targets, is_train=targets is not None),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    preds = []
    target_values = []
    for batch in dataloader:
        sequences_batch = batch["sequence"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)

        tr_states = transformer_model.get_hidden_states(sequences_batch, padding_mask)
        lstm_states = bilstm_model.get_hidden_states(sequences_batch, padding_mask)
        logits = fusion_model(tr_states, lstm_states, padding_mask)
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())

        if targets is not None:
            target_values.append(batch["target"].detach().cpu().numpy())

    preds = np.concatenate(preds)
    if targets is None:
        return preds
    return preds, np.concatenate(target_values)


def evaluate_sequence_model(model, dataloader, device):
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            sequences = batch["sequence"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            targets = batch["target"].cpu().numpy()

            logits = model(sequences, padding_mask)["logits"]
            if torch.isnan(logits).any():
                print("NaN logits batch:", batch_idx)
                raise ValueError("NaN in logits")

            all_targets.append(targets)
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

    return roc_auc_score(np.concatenate(all_targets), np.concatenate(all_preds))


def extract_sequence_embeddings(model, sequences, prefix="trans_emb", batch_size=1024, device=None):
    device = resolve_device(device)
    model = model.to(device)
    model.eval()

    dataset = CreditDataset(sequences, targets=None, is_train=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    client_ids_list = []
    embeddings_list = []
    with torch.no_grad():
        for batch in dataloader:
            seq = batch["sequence"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            outputs = model(seq, padding_mask)
            client_ids_list.extend(batch["client_id"])
            embeddings_list.append(outputs["embeddings"].detach().cpu().numpy())

    embeddings_arr = np.concatenate(embeddings_list, axis=0)
    emb_cols = [f"{prefix}_{i}" for i in range(embeddings_arr.shape[1])]
    embeddings_df = pd.DataFrame(embeddings_arr, columns=emb_cols)
    embeddings_df["id"] = [int(cid) for cid in client_ids_list]
    return embeddings_df


def build_sequences(train_data_path, test_data_path, chunk_size=1_000_000):
    return (
        create_sequence(train_data_path, chunk_size=chunk_size),
        create_sequence(test_data_path, chunk_size=chunk_size),
    )


def make_sequence_fold_data(train_ids, val_ids, y_train, y_val, train_sequences):
    train_fold_sequences = {
        int(cid): train_sequences[int(cid)] for cid in train_ids if int(cid) in train_sequences
    }
    val_fold_sequences = {
        int(cid): train_sequences[int(cid)] for cid in val_ids if int(cid) in train_sequences
    }
    train_fold_targets = {
        int(cid): float(target)
        for cid, target in zip(train_ids, y_train, strict=False)
        if int(cid) in train_fold_sequences
    }
    val_fold_targets = {
        int(cid): float(target)
        for cid, target in zip(val_ids, y_val, strict=False)
        if int(cid) in val_fold_sequences
    }

    return train_fold_sequences, train_fold_targets, val_fold_sequences, val_fold_targets


def build_embedding_frames(
    train_embeddings_df,
    val_embeddings_df,
    test_embeddings_df,
    train_ids,
    val_ids,
    test_ids,
    y_train,
    y_val,
):
    for df in (train_embeddings_df, val_embeddings_df, test_embeddings_df):
        df["id"] = df["id"].astype(int)

    train_target_df = pd.DataFrame({"id": train_ids.astype(int), "target": y_train})
    val_target_df = pd.DataFrame({"id": val_ids.astype(int), "target": y_val})
    test_id_df = pd.DataFrame({"id": test_ids})

    train_df = train_embeddings_df.merge(train_target_df, on="id", how="inner")
    val_df = val_embeddings_df.merge(val_target_df, on="id", how="inner")
    test_df = test_id_df.merge(test_embeddings_df, on="id", how="left")
    return train_df, val_df, test_df


extract_transformer_embeddings = extract_sequence_embeddings
