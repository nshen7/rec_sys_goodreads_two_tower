"""
Two-Tower Model

ItemTower : encodes items from id + categorical + numeric features
UserTower : encodes users from user_id + weighted history of item_ids
TwoTowerModel : owns all embedding tables and both towers; exposes forward(),
                encode_items(), and encode_users() for training and inference.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from .embeddings import (
    CategoricalEmbedding,
    HistoryPooling,
    IDEmbedding,
    ShelfEmbedding,
    build_mlp,
)


# ── Item Tower ────────────────────────────────────────────────────────────────

class ItemTower(nn.Module):
    """
    Encodes a batch of M items into item_vecs [M, d_out].

    Inputs (pre-looked-up by TwoTowerModel before calling forward):
        item_id_emb  [M, d_id]    — from IDEmbedding
        author_emb   [M, d_cat]   — from CategoricalEmbedding
        language_emb [M, d_cat]   — from CategoricalEmbedding
        format_emb   [M, d_cat]   — from CategoricalEmbedding
        shelf_emb    [M, d_cat]   — from ShelfEmbedding (mean of 3 slots)
        numeric      [M, 5]       — pre-normalized floats

    Total input dim: d_id + 4*d_cat + 5 = 128 + 128 + 5 = 261

    Processing: concat → MLP (item_mlp_hidden) → (optional L2-norm) → item_vec
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d_id = cfg.model.d_id
        d_cat = cfg.model.d_cat
        d_out = cfg.model.d_out
        input_dim = d_id + 4 * d_cat + 5
        self.mlp = build_mlp(
            input_dim=input_dim,
            hidden_dims=list(cfg.model.item_mlp_hidden),
            output_dim=d_out,
            dropout=cfg.model.dropout,
        )
        self.normalize = cfg.model.normalize

    def forward(
        self,
        item_id_emb: Tensor,
        author_emb: Tensor,
        language_emb: Tensor,
        format_emb: Tensor,
        shelf_emb: Tensor,
        numeric: Tensor,
    ) -> Tensor:
        """
        Returns
        -------
        item_vec : FloatTensor [M, d_out]
        """
        x = torch.cat([item_id_emb, author_emb, language_emb, format_emb, shelf_emb, numeric], dim=-1)
        out = self.mlp(x)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


# ── User Tower ────────────────────────────────────────────────────────────────

class UserTower(nn.Module):
    """
    Encodes a batch of B users into user_vecs [B, d_out].

    Inputs:
        user_id_emb   [B, d_id]    — from IDEmbedding
        pooled_history [B, d_id]   — from HistoryPooling

    Total input dim: d_id + d_id = 2 * d_id = 256

    Processing: concat → MLP (user_mlp_hidden) → (optional L2-norm) → user_vec

    The HistoryPooling module is held here but uses the item_id_embedding
    from the item tower (passed in at TwoTowerModel construction), sharing
    the item embedding space with the item tower.
    """

    def __init__(self, cfg: DictConfig, item_id_embedding: IDEmbedding) -> None:
        super().__init__()
        d_id = cfg.model.d_id
        d_out = cfg.model.d_out
        self.history_pooling = HistoryPooling(item_id_embedding)
        input_dim = 2 * d_id
        self.mlp = build_mlp(
            input_dim=input_dim,
            hidden_dims=list(cfg.model.user_mlp_hidden),
            output_dim=d_out,
            dropout=cfg.model.dropout,
        )
        self.normalize = cfg.model.normalize

    def forward(
        self,
        user_id_emb: Tensor,
        history_item_ids: Tensor,
        history_item_weights: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        user_id_emb         : FloatTensor [B, d_id]
        history_item_ids    : LongTensor  [B, H]
        history_item_weights: FloatTensor [B, H]

        Returns
        -------
        user_vec : FloatTensor [B, d_out]
        """
        pooled = self.history_pooling(history_item_ids, history_item_weights)  # [B, d_id]
        x = torch.cat([user_id_emb, pooled], dim=-1)                           # [B, 2*d_id]
        out = self.mlp(x)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


# ── Two-Tower Model ───────────────────────────────────────────────────────────

class TwoTowerModel(nn.Module):
    """
    Owns all embedding tables and both towers.

    Training: forward(batch) → (user_vecs [B, d], item_vecs [B+N_neg, d])
    The caller computes scores = user_vecs @ item_vecs.T and passes to the loss.

    Inference:
        encode_items(item_ids, item_cat_feats, item_num_feats) → item_vecs
        encode_users(user_id, history_item_ids, history_item_weights) → user_vecs

    Parameters
    ----------
    cfg : DictConfig
        Full config (baseline.yaml).
    artifacts : dict
        Output of build_artifacts() — provides vocab sizes for embedding tables.
    """

    def __init__(self, cfg: DictConfig, artifacts: dict[str, Any]) -> None:
        super().__init__()

        d_id = cfg.model.d_id
        d_cat = cfg.model.d_cat

        num_users = artifacts["num_users"]
        num_items = artifacts["num_items"]
        num_authors = len(artifacts["author_vocab"]) - 1     # subtract pad index
        num_languages = len(artifacts["language_vocab"]) - 1
        num_formats = len(artifacts["format_vocab"]) - 1
        num_shelves = len(artifacts["shelf_vocab"]) - 1

        # ── Embedding tables ──────────────────────────────────────────────────

        # User IDs: no padding (every user is valid)
        self.user_id_embedding = IDEmbedding(num_users, d_id, padding_idx=None)

        # Item IDs: padding_idx=0 so PAD item stays at zero
        # Shared between ItemTower and HistoryPooling in UserTower
        self.item_id_embedding = IDEmbedding(num_items, d_id, padding_idx=0)

        self.author_embedding = CategoricalEmbedding(num_authors, d_cat)
        self.language_embedding = CategoricalEmbedding(num_languages, d_cat)
        self.format_embedding = CategoricalEmbedding(num_formats, d_cat)
        self.shelf_embedding = ShelfEmbedding(num_shelves, d_cat)

        # ── Towers ────────────────────────────────────────────────────────────

        self.item_tower = ItemTower(cfg)
        # UserTower receives the shared item_id_embedding for HistoryPooling
        self.user_tower = UserTower(cfg, self.item_id_embedding)

    def encode_items(
        self,
        item_ids: Tensor,
        item_cat_feats: Tensor,
        item_num_feats: Tensor,
    ) -> Tensor:
        """
        Encode a batch of M items.

        Parameters
        ----------
        item_ids       : LongTensor  [M]
        item_cat_feats : LongTensor  [M, 6]
            Columns: author_idx, language_idx, format_idx, shelf_0_idx, shelf_1_idx, shelf_2_idx
        item_num_feats : FloatTensor [M, 5]
            Columns: avg_rating, ratings_count, num_pages, pub_year, is_ebook (all normalized)

        Returns
        -------
        item_vecs : FloatTensor [M, d_out]
        """
        item_id_emb = self.item_id_embedding(item_ids)                         # [M, d_id]
        author_emb = self.author_embedding(item_cat_feats[:, 0])               # [M, d_cat]
        language_emb = self.language_embedding(item_cat_feats[:, 1])           # [M, d_cat]
        format_emb = self.format_embedding(item_cat_feats[:, 2])               # [M, d_cat]
        shelf_emb = self.shelf_embedding(item_cat_feats[:, 3:6])               # [M, d_cat]

        return self.item_tower(
            item_id_emb, author_emb, language_emb, format_emb, shelf_emb, item_num_feats
        )

    def encode_users(
        self,
        user_id: Tensor,
        history_item_ids: Tensor,
        history_item_weights: Tensor,
    ) -> Tensor:
        """
        Encode a batch of B users.

        Parameters
        ----------
        user_id              : LongTensor  [B]
        history_item_ids     : LongTensor  [B, H]
        history_item_weights : FloatTensor [B, H]

        Returns
        -------
        user_vecs : FloatTensor [B, d_out]
        """
        user_id_emb = self.user_id_embedding(user_id)   # [B, d_id]
        return self.user_tower(user_id_emb, history_item_ids, history_item_weights)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """
        Full forward pass for training.

        Parameters
        ----------
        batch : dict
            Output of TwoTowerCollator.__call__(). Expected keys:
            - user_id              [B]
            - history_item_ids     [B, H]
            - history_item_weights [B, H]
            - item_ids             [B + N_neg]
            - item_cat_feats       [B + N_neg, 6]
            - item_num_feats       [B + N_neg, 5]

        Returns
        -------
        user_vecs : FloatTensor [B, d_out]
        item_vecs : FloatTensor [B + N_neg, d_out]
        """
        user_vecs = self.encode_users(
            batch["user_id"],
            batch["history_item_ids"],
            batch["history_item_weights"],
        )
        item_vecs = self.encode_items(
            batch["item_ids"],
            batch["item_cat_feats"],
            batch["item_num_feats"],
        )
        return user_vecs, item_vecs
