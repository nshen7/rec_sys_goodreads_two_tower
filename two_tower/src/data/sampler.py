"""
Two-Tower Sampler

Responsibilities:
- Build a per-user confirmed-negatives index from the confirmed_negatives Parquet table
- Provide a weighted random sampler for the DataLoader (training only)
- Provide TwoTowerCollator: a custom collate_fn that appends confirmed-negative items
  to the batch's item matrix so the loss can use them as hard negatives
"""

from __future__ import annotations

import random
from typing import Any

import gc
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Sampler, WeightedRandomSampler


# ── Confirmed-negatives index ──────────────────────────────────────────────────

def build_confirmed_neg_index(confirmed_neg_path: str) -> dict[int, list[int]]:
    """
    Load the confirmed_negatives Parquet table and build a per-user index.

    Parameters
    ----------
    confirmed_neg_path : str
        GCS or local path to the confirmed_negatives/ Parquet directory.
        Schema: user_id (int), item_id (int), rating (int).

    Returns
    -------
    dict[user_id -> list[item_id]]
        For each user, the list of item_ids they explicitly disliked
        (is_read=1, rating∈{1,2}).
    """
    df = pd.read_parquet(confirmed_neg_path, columns=["user_id", "item_id"])
    index: dict[int, list[int]] = {}
    for user_id, group in df.groupby("user_id", sort=False):
        index[int(user_id)] = group["item_id"].tolist()
    del df
    gc.collect()
    return index


# ── Weighted sampler ───────────────────────────────────────────────────────────

## Use WeightedRandomSampler to sample training interactions with probability proportional to their interaction strength. 
# This lets the model see more of the "important" interactions each epoch, which can speed up convergence and improve final 
# performance, and the negative samples are more high-quality.

class NumpyWeightedSampler(Sampler):
    """
    WeightedRandomSampler equivalent backed by numpy.random.choice.

    torch.multinomial is limited to 2^24 categories, which is too small for
    datasets with tens of millions of rows.  numpy.random.choice has no such
    restriction and is used here instead.
    """

    def __init__(self, weights: np.ndarray, num_samples: int) -> None:
        self._probs = weights / weights.sum()
        self._num_samples = num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):
        indices = np.random.choice(
            len(self._probs),
            size=self._num_samples,
            replace=True,
            p=self._probs,
        )
        return iter(indices)  # numpy iterator avoids copying to a Python list


_MULTINOMIAL_LIMIT = 2 ** 24  # torch.multinomial hard limit


def make_weighted_sampler(
    sample_weights: np.ndarray,
    transform: str = "log1p",
    num_samples: int | None = None,
) -> Sampler:
    """
    Build a weighted sampler so higher-confidence interactions are
    sampled more often.

    Parameters
    ----------
    sample_weights : np.ndarray, shape [N]
        Raw sample weights from the training dataframe (interaction strength).
    transform : {"raw", "log1p", "clip"}
        How to map raw weights to sampling probabilities:
        - "raw"   : use weights as-is
        - "log1p" : log1p(weights), dampens the influence of very high weights
        - "clip"  : clip to [1.0, 5.0], then use as-is
    num_samples : int, optional
        Number of samples to draw per epoch. Defaults to len(sample_weights)
        (i.e. same epoch length as unweighted sampling).

    Returns
    -------
    Sampler
        Pass to DataLoader as sampler=.
        Uses NumpyWeightedSampler when N > 2^24 (torch.multinomial limit),
        otherwise uses the standard WeightedRandomSampler.
    """
    w = sample_weights.astype(np.float64)

    if transform == "log1p":
        w = np.log1p(w)
    elif transform == "clip":
        w = np.clip(w, 1.0, 5.0)
    elif transform == "raw":
        pass
    else:
        raise ValueError(f"Unknown sample weight transform: {transform!r}. "
                         f"Choose from 'raw', 'log1p', 'clip'.")

    # Guard against any zero or negative weights
    w = np.maximum(w, 1e-8)

    if num_samples is None:
        num_samples = len(w)

    if len(w) > _MULTINOMIAL_LIMIT:
        return NumpyWeightedSampler(weights=w, num_samples=num_samples)

    return WeightedRandomSampler(
        weights=torch.from_numpy(w).float(),
        num_samples=num_samples,
        replacement=True,
    )


# ── Custom collate ─────────────────────────────────────────────────────────────

class TwoTowerCollator:
    """
    Custom collate_fn for the DataLoader.

    Beyond standard collation, this class appends confirmed-negative items to
    the item matrix each batch. The resulting layout is:

        item_ids         [B + N_neg]   — first B are positives (targets), then conf. negs
        item_cat_feats   [B + N_neg, 6]
        item_num_feats   [B + N_neg, 5]
        item_is_positive [B + N_neg]   — True for first B, False for the rest

    This single item matrix lets the loss compute the full score grid
    user_vecs [B, d] @ item_vecs [B+N_neg, d].T in one matmul.

    Parameters
    ----------
    conf_neg_index : dict[int, list[int]]
        Output of build_confirmed_neg_index().
    item_cat_feats : LongTensor [num_items+1, 6]
        Pre-built categorical feature tensor (shared memory).
    item_num_feats : FloatTensor [num_items+1, 5]
        Pre-built numeric feature tensor (shared memory).
    cfg : DictConfig
        Full config (uses cfg.training.max_confirmed_neg_per_batch).
    """

    def __init__(
        self,
        conf_neg_index: dict[int, list[int]],
        item_cat_feats: Tensor,
        item_num_feats: Tensor,
        cfg: Any,
    ) -> None:
        self.conf_neg_index = conf_neg_index
        self.item_cat_feats = item_cat_feats
        self.item_num_feats = item_num_feats
        self.max_neg = cfg.training.max_confirmed_neg_per_batch

    def __call__(self, samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """
        Collate a list of __getitem__ dicts into a batch dict.

        Output keys
        -----------
        user_id              LongTensor  [B]
        target_item_id       LongTensor  [B]
        history_item_ids     LongTensor  [B, history_length]
        history_item_weights FloatTensor [B, history_length]
        sample_weight        FloatTensor [B]
        item_ids             LongTensor  [B + N_neg]
        item_cat_feats       LongTensor  [B + N_neg, 6]
        item_num_feats       FloatTensor [B + N_neg, 5]
        item_is_positive     BoolTensor  [B + N_neg]
        """
        # Standard stack for per-user fields
        user_ids = torch.stack([s["user_id"] for s in samples])                    # [B]
        target_item_ids = torch.stack([s["target_item_id"] for s in samples])      # [B]
        history_item_ids = torch.stack([s["history_item_ids"] for s in samples])   # [B, H]
        history_item_weights = torch.stack(
            [s["history_item_weights"] for s in samples]
        )                                                                           # [B, H]
        sample_weight = torch.stack([s["sample_weight"] for s in samples])         # [B]

        B = len(samples)
        positive_item_ids = target_item_ids  # [B]

        # Sample confirmed negatives for this batch
        neg_item_ids = self._sample_confirmed_negatives(
            user_ids=user_ids.tolist(),
            positive_ids_in_batch=set(positive_item_ids.tolist()),
        )  # LongTensor [N_neg] or empty

        # Build item matrix [B + N_neg]
        if len(neg_item_ids) > 0:
            all_item_ids = torch.cat([positive_item_ids, neg_item_ids])
        else:
            all_item_ids = positive_item_ids

        M = len(all_item_ids)
        item_is_positive = torch.zeros(M, dtype=torch.bool)
        item_is_positive[:B] = True

        return {
            "user_id": user_ids,
            "target_item_id": target_item_ids,
            "history_item_ids": history_item_ids,
            "history_item_weights": history_item_weights,
            "sample_weight": sample_weight,
            "item_ids": all_item_ids,
            "item_cat_feats": self.item_cat_feats[all_item_ids],  # [M, 6]
            "item_num_feats": self.item_num_feats[all_item_ids],  # [M, 5]
            "item_is_positive": item_is_positive,
        }

    def _sample_confirmed_negatives(
        self,
        user_ids: list[int],
        positive_ids_in_batch: set[int],
    ) -> Tensor:
        """
        Draw up to self.max_neg confirmed-negative item_ids from the union of
        the batch users' confirmed-neg sets.

        Items already present as positives in the batch are excluded to avoid
        contradictory supervision (an item can't be both a positive and a
        confirmed negative in the same batch row).

        Returns
        -------
        LongTensor [N_neg] where N_neg <= self.max_neg.
        Empty tensor if no confirmed negatives exist for this batch.
        """
        # Union of all confirmed neg items for users in this batch
        candidate_set: set[int] = set()
        for uid in user_ids:
            neg_items = self.conf_neg_index.get(uid, [])
            candidate_set.update(neg_items)

        # Remove items already positive in this batch
        candidates = list(candidate_set - positive_ids_in_batch)

        if not candidates:
            return torch.tensor([], dtype=torch.long)

        # Subsample if over budget
        if len(candidates) > self.max_neg:
            candidates = random.sample(candidates, self.max_neg)

        return torch.tensor(candidates, dtype=torch.long)
