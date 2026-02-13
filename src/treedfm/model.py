
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeEditTransformer(nn.Module):
    """
    Transformer encoder over a set of nodes with tree-structure features.

    Node features:
      x[..., 0] = depth
      x[..., 1] = rank (0..K-1)
      x[..., 2] = type token (0=empty/inactive)
      x[..., 3] = parent index (-1 for root)

    Outputs per node and per slot:
      - rates:      [B, N, K, 3] (ins, del, sub), nonnegative
      - ins_logits: [B, N, K, C]
      - sub_logits: [B, N, K, C]
    """

    def __init__(
        self,
        *,
        num_types: int,
        k: int,
        max_depth: int,
        max_nodes: int,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_types = int(num_types)
        self.k = int(k)
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.d_model = int(d_model)

        # Token / structural embeddings
        self.type_emb = nn.Embedding(self.num_types + 1, d_model, padding_idx=0)
        self.depth_emb = nn.Embedding(self.max_depth + 2, d_model)
        self.rank_emb = nn.Embedding(self.k + 1, d_model)

        # Index embeddings to help link parent-child via attention
        self.pos_emb = nn.Embedding(self.max_nodes + 1, d_model, padding_idx=0)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Heads
        self.rate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.k * 3),
            nn.Softplus(),
        )
        self.ins_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.k * self.num_types),
        )
        self.sub_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.k * self.num_types),
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor, t: torch.Tensor):
        """
        x: [B,N,4]
        src_key_padding_mask: [B,N] True to mask out
        t: [B] in [0,1]
        """
        if x.dim() != 3 or x.size(-1) != 4:
            raise ValueError(f"TreeEditTransformer: expected x [B,N,4], got {tuple(x.shape)}")
        B, N, _ = x.shape
        device = x.device

        depth = x[:, :, 0].clamp(min=0, max=self.max_depth + 1)
        rank = x[:, :, 1].clamp(min=0, max=self.k)
        typ = x[:, :, 2].clamp(min=0, max=self.num_types)

        # positional ids based on sequence index (1..N), 0 reserved
        pos_ids = torch.arange(N, device=device).view(1, N).expand(B, N) + 1
        pos_ids = pos_ids.clamp(max=self.max_nodes)

        parent_idx = x[:, :, 3].clone()
        parent_ids = (parent_idx + 1).clamp(min=0, max=self.max_nodes)  # -1 -> 0

        h = (
            self.type_emb(typ)
            + self.depth_emb(depth)
            + self.rank_emb(rank)
            + self.pos_emb(pos_ids)
            + self.pos_emb(parent_ids)
            + self.time_mlp(t.view(B, 1)).unsqueeze(1)
        )

        out = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        rates = self.rate_head(out).view(B, N, self.k, 3)
        ins_logits = self.ins_head(out).view(B, N, self.k, self.num_types)
        sub_logits = self.sub_head(out).view(B, N, self.k, self.num_types)
        return rates, ins_logits, sub_logits
