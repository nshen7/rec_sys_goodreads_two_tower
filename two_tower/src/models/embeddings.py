"""
Two-Tower Embedding Modules

Reusable building blocks consumed by both towers:
- IDEmbedding       : user_id / item_id lookup tables
- CategoricalEmbedding : single categorical feature (author, language, format)
- ShelfEmbedding    : shared table for all 3 shelf slots, mean-pooled
- HistoryPooling    : weighted mean-pool of item embeddings over history window
- build_mlp         : factory for Linear → LayerNorm → ReLU → Dropout stacks
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ── ID Embedding ───────────────────────────────────────────────────────────────

class IDEmbedding(nn.Module):
    """
    Embedding table for user_id or item_id.

    Initialised with N(0, 1/sqrt(d)) for reasonable gradient scale.
    padding_idx=0 is used for item_id (PAD item) so the PAD embedding stays
    zero through training; user_id has no padding so padding_idx=None.

    Parameters
    ----------
    num_entities : int
        Highest possible ID. Table size = num_entities + 1 (to include index 0).
    d : int
        Embedding dimension.
    padding_idx : int or None
        If 0, the zeroth row is frozen at zero (item tower). None for user tower.
    """

    def __init__(self, num_entities: int, d: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_entities + 1, d, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d))
        if padding_idx is not None:
            # Re-zero the pad row after init (normal_ may have set it)
            with torch.no_grad():
                self.embedding.weight[padding_idx].zero_()

    def forward(self, ids: Tensor) -> Tensor:
        """ids: LongTensor [...] → embeddings [..., d]"""
        return self.embedding(ids)


# ── Categorical Embedding ──────────────────────────────────────────────────────

class CategoricalEmbedding(nn.Module):
    """
    Embedding table for a single categorical feature (author, language, format).

    Index 0 = unknown/OOV, kept at zero by padding_idx so the model never
    "learns" a representation for the unknown token — it is simply absent.

    Parameters
    ----------
    num_categories : int
        Number of known categories (excluding index 0). Table size = num_categories + 1.
    d : int
        Embedding dimension.
    """

    def __init__(self, num_categories: int, d: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_categories + 1, d, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d))
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, indices: Tensor) -> Tensor:
        """indices: LongTensor [...] → embeddings [..., d]"""
        return self.embedding(indices)


# ── Shelf Embedding ────────────────────────────────────────────────────────────

class ShelfEmbedding(nn.Module):
    """
    A single shared embedding table for all three shelf slots.

    Each item has up to 3 shelf labels (shelf_0, shelf_1, shelf_2). Rather than
    three independent tables, we use one table and mean-pool the non-pad slots.
    This enforces that "fantasy" in slot 0 means the same thing as "fantasy" in
    slot 1, and uses 1/3 the parameters.

    Index 0 = "" (pad / empty slot) → frozen at zero, excluded from the mean.

    Parameters
    ----------
    num_shelves : int
        Number of known shelf tokens (excluding index 0). Table size = num_shelves + 1.
    d : int
        Embedding dimension.
    """

    def __init__(self, num_shelves: int, d: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_shelves + 1, d, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d))
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, shelf_indices: Tensor) -> Tensor:
        """
        Parameters
        ----------
        shelf_indices : LongTensor [..., 3]
            Indices for shelf slots 0, 1, 2. 0 = pad (empty slot).

        Returns
        -------
        Tensor [..., d]
            Weighted mean of non-pad slot embeddings.
            If all slots are pad (item has no shelves), returns zeros.
        """
        embs = self.embedding(shelf_indices)   # [..., 3, d]
        mask = (shelf_indices != 0).float()    # [..., 3]  1.0 for non-pad slots

        # Sum non-pad embeddings, divide by count of non-pad slots
        numer = (embs * mask.unsqueeze(-1)).sum(dim=-2)   # [..., d]
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [..., 1]
        return numer / denom


# ── History Pooling ────────────────────────────────────────────────────────────

class HistoryPooling(nn.Module):
    """
    Weighted mean-pool of item_id embeddings over the user's history window.

    The item_id embedding used here is the shared table from the item tower
    (passed in at construction time). This ties history representations to the
    item embedding space without adding parameters.

    Pad slots (history_item_ids == 0, history_item_weights == 0) contribute
    zero to the weighted sum because:
      - item_id_embedding has padding_idx=0, so pad embs are zero vectors.
      - history_item_weights are 0.0 for pad slots.

    Parameters
    ----------
    item_id_embedding : IDEmbedding
        Shared item embedding table from the item tower (padding_idx=0).
    """

    def __init__(self, item_id_embedding: IDEmbedding) -> None:
        super().__init__()
        self.item_id_embedding = item_id_embedding

    def forward(self, history_item_ids: Tensor, history_item_weights: Tensor) -> Tensor:
        """
        Parameters
        ----------
        history_item_ids : LongTensor [B, H]
            Item IDs in the history window. 0 = pad.
        history_item_weights : FloatTensor [B, H]
            Interaction strength for each history item. 0.0 for pad slots.

        Returns
        -------
        FloatTensor [B, d_id]
            Weighted mean-pooled history representation.
            All-pad rows return zero vectors.
        """
        embs = self.item_id_embedding(history_item_ids)          # [B, H, d_id]
        weights = history_item_weights.unsqueeze(-1)              # [B, H, 1]

        numer = (embs * weights).sum(dim=1)                       # [B, d_id]
        denom = weights.sum(dim=1).clamp(min=1e-8)                # [B, 1]
        return numer / denom


# ── MLP factory ───────────────────────────────────────────────────────────────

def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """
    Build a fully-connected MLP:
        [Linear → LayerNorm → ReLU → Dropout] × len(hidden_dims) → Linear

    LayerNorm (rather than BatchNorm) is used for robustness to small or
    varying batch sizes at evaluation time.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    hidden_dims : list[int]
        Sizes of hidden layers.
    output_dim : int
        Final output dimensionality (d_out).
    dropout : float
        Dropout probability applied after each ReLU.

    Returns
    -------
    nn.Sequential
    """
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.extend([
            nn.Linear(in_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)
