import torch
import torch.nn as nn

from config import FEATURE_CARDINALITIES, FEATURE_COLS, SEQUENCE_MAX_LENGTH


class CreditTransformer(torch.nn.Module):
    def __init__(
        self,
        feature_embeddings_dim=32,
        embedding_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.2,
    ):
        super().__init__()
        self.feature_embeddings_dim = feature_embeddings_dim
        self.embedding_dim = embedding_dim
        self.concat_dim = len(FEATURE_COLS) * self.feature_embeddings_dim

        self.feature_embeddings = torch.nn.ModuleDict(
            {
                col: torch.nn.Embedding(
                    num_embeddings=cardinality + 1,
                    embedding_dim=self.feature_embeddings_dim,
                    padding_idx=0,
                )
                for col, cardinality in FEATURE_CARDINALITIES.items()
            }
        )

        self.init_proj = torch.nn.Sequential(
            torch.nn.Linear(self.concat_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, self.embedding_dim),
            torch.nn.LayerNorm(self.embedding_dim),
        )

        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.pos_encoder = nn.Embedding(SEQUENCE_MAX_LENGTH, self.embedding_dim)
        self.cls_head = torch.nn.Linear(self.embedding_dim, 1)

    def _embed_sequence(self, x, padding_mask):
        _validate_sequence_batch(x)
        bs, seq_len, _ = x.shape

        embedded_features = []
        for i, col in enumerate(FEATURE_COLS):
            feature_tensor = x[:, :, i].long().masked_fill(padding_mask, 0)
            embedded_features.append(self.feature_embeddings[col](feature_tensor))

        x = torch.cat(embedded_features, dim=-1)
        x = self.init_proj(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bs, seq_len)
        return x + self.pos_encoder(positions)

    def get_hidden_states(self, x, padding_mask):
        x = self._embed_sequence(x, padding_mask)
        return self.transformer_encoder(x, src_key_padding_mask=padding_mask)

    def forward(self, x, padding_mask):
        ts_out = self.get_hidden_states(x, padding_mask)
        valid_mask = (~padding_mask).float().unsqueeze(-1)
        embeddings = (ts_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        logits = self.cls_head(embeddings).squeeze(-1)
        return {"logits": logits, "embeddings": embeddings}


class CreditXLSTM(torch.nn.Module):
    def __init__(
        self,
        num_blocks=4,
        embedding_dim=256,
        feature_embeddings_dim=32,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()
        try:
            from xlstm import (
                FeedForwardConfig,
                mLSTMBlockConfig,
                mLSTMLayerConfig,
                sLSTMBlockConfig,
                sLSTMLayerConfig,
                xLSTMBlockStack,
                xLSTMBlockStackConfig,
            )
        except ImportError as exc:
            raise ImportError("Install xlstm to use CreditXLSTM") from exc

        self.feature_embeddings_dim = feature_embeddings_dim
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.concat_dim = len(FEATURE_COLS) * self.feature_embeddings_dim

        self.feature_embeddings = torch.nn.ModuleDict(
            {
                col: torch.nn.Embedding(
                    num_embeddings=cardinality + 1,
                    embedding_dim=self.feature_embeddings_dim,
                    padding_idx=0,
                )
                for col, cardinality in FEATURE_CARDINALITIES.items()
            }
        )

        self.init_proj = torch.nn.Sequential(
            torch.nn.Linear(self.concat_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, self.embedding_dim),
            torch.nn.LayerNorm(self.embedding_dim),
        )

        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=SEQUENCE_MAX_LENGTH,
            num_blocks=num_blocks,
            embedding_dim=self.embedding_dim,
            slstm_at=[1],
        )

        self.xlstm = xLSTMBlockStack(xlstm_cfg)
        self.cls_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embedding_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.embedding_dim, 1),
        )

    def forward(self, x, padding_mask):
        _validate_sequence_batch(x)
        bs = x.shape[0]

        embedded_features = []
        for i, col in enumerate(FEATURE_COLS):
            feature_tensor = x[:, :, i].long().masked_fill(padding_mask, 0)
            embedded_features.append(self.feature_embeddings[col](feature_tensor))

        x = torch.cat(embedded_features, dim=-1)
        x = self.init_proj(x)
        ts_out = self.xlstm(x)
        valid_mask = (~padding_mask).float().unsqueeze(-1)

        if self.pooling == "mean":
            embeddings = (ts_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        elif self.pooling == "last":
            lengths = (~padding_mask).sum(dim=1).clamp(min=1)
            last_idx = lengths - 1
            batch_idx = torch.arange(bs, device=x.device)
            embeddings = ts_out[batch_idx, last_idx]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.cls_head(embeddings).squeeze(-1)
        return {"logits": logits, "embeddings": embeddings}


class CreditBiLSTM(nn.Module):
    def __init__(
        self,
        num_layers=2,
        embedding_dim=256,
        feature_embeddings_dim=32,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()

        self.feature_embeddings_dim = feature_embeddings_dim
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.concat_dim = len(FEATURE_COLS) * self.feature_embeddings_dim
        self.hidden_dim = embedding_dim // 2

        self.feature_embeddings = torch.nn.ModuleDict(
            {
                col: torch.nn.Embedding(
                    num_embeddings=cardinality + 1,
                    embedding_dim=self.feature_embeddings_dim,
                    padding_idx=0,
                )
                for col, cardinality in FEATURE_CARDINALITIES.items()
            }
        )

        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for bidirectional LSTM")

        self.init_proj = nn.Sequential(
            nn.Linear(self.concat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 1),
        )

    def _embed_sequence(self, x, padding_mask):
        _validate_sequence_batch(x)

        embedded_features = []
        for i, col in enumerate(FEATURE_COLS):
            feature_tensor = x[:, :, i].long().masked_fill(padding_mask, 0)
            embedded_features.append(self.feature_embeddings[col](feature_tensor))

        x = torch.cat(embedded_features, dim=-1)
        return self.init_proj(x)

    def get_hidden_states(self, x, padding_mask):
        x = self._embed_sequence(x, padding_mask)
        lengths = (~padding_mask).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.shape[1],
        )
        return lstm_out

    def forward(self, x, padding_mask):
        bs = x.shape[0]
        lstm_out = self.get_hidden_states(x, padding_mask)
        lengths = (~padding_mask).sum(dim=1).clamp(min=1)

        valid_mask = (~padding_mask).float().unsqueeze(-1)
        if self.pooling == "mean":
            embeddings = (lstm_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        elif self.pooling == "last":
            last_idx = lengths.to(lstm_out.device) - 1
            batch_idx = torch.arange(bs, device=lstm_out.device)
            embeddings = lstm_out[batch_idx, last_idx]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.cls_head(embeddings).squeeze(-1)
        return {"logits": logits, "embeddings": embeddings}


class CrossAttentionFusionMLP(nn.Module):
    def __init__(
        self,
        d_tr,
        d_lstm,
        d_model=256,
        n_heads=4,
        hidden=256,
        dropout=0.2,
    ):
        super().__init__()

        self.tr_proj = nn.Linear(d_tr, d_model)
        self.lstm_proj = nn.Linear(d_lstm, d_model)

        self.tr_to_lstm = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.lstm_to_tr = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_tr = nn.LayerNorm(d_model)
        self.norm_lstm = nn.LayerNorm(d_model)

        self.fusion_dim = d_model * 4

        self.mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def masked_mean_pooling(self, h, padding_mask=None):
        if padding_mask is None:
            return h.mean(dim=1)

        mask = (~padding_mask).unsqueeze(-1).float()
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def extract_fusion_embedding(self, tr_states, lstm_states, padding_mask=None):
        tr = self.tr_proj(tr_states)
        lstm = self.lstm_proj(lstm_states)

        tr_attn, _ = self.tr_to_lstm(
            query=tr,
            key=lstm,
            value=lstm,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        lstm_attn, _ = self.lstm_to_tr(
            query=lstm,
            key=tr,
            value=tr,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        tr_fused = self.norm_tr(tr + tr_attn)
        lstm_fused = self.norm_lstm(lstm + lstm_attn)

        tr_pool = self.masked_mean_pooling(tr_fused, padding_mask)
        lstm_pool = self.masked_mean_pooling(lstm_fused, padding_mask)

        return torch.cat(
            [
                tr_pool,
                lstm_pool,
                tr_pool * lstm_pool,
                torch.abs(tr_pool - lstm_pool),
            ],
            dim=1,
        )

    def forward(self, tr_states, lstm_states, padding_mask=None):
        fusion_emb = self.extract_fusion_embedding(
            tr_states,
            lstm_states,
            padding_mask,
        )
        logits = self.mlp(fusion_emb)
        return logits.squeeze(1)


def _validate_sequence_batch(x):
    if x.numel() == 0:
        raise ValueError(f"x has zero numel: x.shape={x.shape}")
    if x.ndim != 3:
        raise ValueError(f"x should be 3D, got shape={x.shape}")

    _, seq_len, num_features = x.shape
    if seq_len == 0:
        raise ValueError(f"zero seq_len: x.shape={x.shape}")
    if num_features != len(FEATURE_COLS):
        raise ValueError(f"Expected {len(FEATURE_COLS)} features, got {num_features}")
    if seq_len > SEQUENCE_MAX_LENGTH:
        raise ValueError(f"seq_len={seq_len} > SEQUENCE_MAX_LENGTH={SEQUENCE_MAX_LENGTH}")
