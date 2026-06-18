import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from models import GatedResMLPClassifier


class EmbeddingDataset(Dataset):
    def __init__(self, X, y=None):
        if hasattr(X, "values"):
            X = X.values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


@torch.no_grad()
def predict_mlp(model, X, batch_size=4096, device="cuda"):
    dataset = EmbeddingDataset(X)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    preds = []
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        probs = torch.sigmoid(model(xb))
        preds.append(probs.detach().cpu().numpy())

    return np.concatenate(preds)


def train_mlp_on_embeddings(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    scale_pos_weight=5.0,
    epochs=40,
    batch_size=4096,
    lr=1e-3,
    weight_decay=1e-3,
    device="cuda",
):
    train_dataset = EmbeddingDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = GatedResMLPClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scale_pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_auc = -np.inf
    best_state = None
    patience = 6
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        val_pred = predict_mlp(model, X_val, batch_size=batch_size, device=device)
        val_auc = roc_auc_score(y_val, val_pred)
        scheduler.step(val_auc)
        print(f"Epoch {epoch:02d} | loss: {np.mean(losses):.5f} | val_auc: {val_auc:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping. Best AUC: {best_auc:.6f}")
            break

    model.load_state_dict(best_state)
    return model, best_auc
