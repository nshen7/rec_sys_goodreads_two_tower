"""
Two-Tower Training Loss

InfoNCE loss over the [B, B + N_neg] score matrix produced by:
    scores = user_vecs @ item_vecs.T

The diagonal of the [B, B] submatrix is the positive (target item for each user).
All off-diagonal entries are in-batch negatives.
The extra N_neg columns are confirmed negatives appended by TwoTowerCollator.

Interaction-strength weighting is handled upstream by WeightedRandomSampler in
sampler.py — no per-sample loss scaling is needed here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def infonce_loss(
    user_vecs: Tensor,
    item_vecs: Tensor,
    item_is_positive: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """
    InfoNCE loss for two-tower training.

    Steps
    -----
    1. scores [B, B+N_neg] = user_vecs @ item_vecs.T / temperature
    2. labels [B]          = torch.arange(B)          (diagonal = positive item)
    3. loss                = cross_entropy(scores, labels).mean()

    The standard cross_entropy with labels=arange(B) is correct here because:
    - Position i in item_vecs is the positive item for user i (diagonal).
    - All other positions (off-diagonal B×B block + N_neg columns) are negatives.
    - Confirmed negatives appear only in off-diagonal columns, so the label never
      points to a confirmed-negative slot — no contradictory supervision.

    Parameters
    ----------
    user_vecs : FloatTensor [B, d]
        L2-normalised (or raw) user representations.
    item_vecs : FloatTensor [B + N_neg, d]
        First B rows: positive target items (one per user).
        Remaining N_neg rows: confirmed negatives appended by TwoTowerCollator.
    item_is_positive : BoolTensor [B + N_neg]
        True for the first B entries (positives), False for confirmed negatives.
        Not used in the core loss computation but available for diagnostic logging
        (e.g. tracking the average score on confirmed-negative columns separately).
    temperature : float
        Softmax temperature. Lower values sharpen the distribution and push the
        model to be more discriminative. Start at 1.0; can anneal or make a
        learnable parameter in a later version.

    Returns
    -------
    Tensor (scalar)
        Mean cross-entropy loss.
    """
    B = user_vecs.size(0)

    # [B, B + N_neg]
    scores = user_vecs @ item_vecs.T / temperature

    # Positive for user i is item at position i in item_vecs
    labels = torch.arange(B, device=user_vecs.device)  # [B]

    return F.cross_entropy(scores, labels)
