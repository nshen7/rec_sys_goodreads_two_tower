"""Standalone training script — equivalent to notebooks/03_train.ipynb.

Run from the two_tower/ directory:
    nohup python notebooks/03_train.py > logs/<experiment_name>/train.log 2>&1 &
    echo $! > logs/<experiment_name>/train.pid   # save PID to kill/check later
"""

import datetime
import random
import sys
import time
from pathlib import Path

## resolves to the two_tower/ directory to enable imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.loaders import prepare_data
from src.data.sampler import (
    TwoTowerCollator,
    build_confirmed_neg_index,
    make_weighted_sampler,
)
from src.models.two_tower import TwoTowerModel
from src.train.losses import infonce_loss

# ── Config ──────────────────────────────────────────────────────────────────

cfg = OmegaConf.load("configs/baseline.yaml")

SEED = cfg.training.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("Using device:", DEVICE)
print(OmegaConf.to_yaml(cfg))

# ── Data ─────────────────────────────────────────────────────────────────────

train_dataset, val_dataset, item_cat_feats, item_num_feats, artifacts = prepare_data(cfg)

print("Loading confirmed negatives index...")
conf_neg_index = build_confirmed_neg_index(cfg.data.confirmed_negatives_path)
print(f"  {len(conf_neg_index):,} users with confirmed negatives")

collator = TwoTowerCollator(
    conf_neg_index=conf_neg_index,
    item_cat_feats=item_cat_feats,
    item_num_feats=item_num_feats,
    cfg=cfg,
)

train_sampler = make_weighted_sampler(
    sample_weights=train_dataset.sample_weights,
    transform=cfg.training.sample_weight_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.training.batch_size,
    sampler=train_sampler,
    collate_fn=collator,
    num_workers=cfg.data.num_workers,
    prefetch_factor=cfg.data.prefetch_factor,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.training.batch_size,
    shuffle=False,
    collate_fn=collator,
    num_workers=cfg.data.num_workers,
    prefetch_factor=cfg.data.prefetch_factor,
    pin_memory=True,
)

print(f"Train batches per epoch: {len(train_loader):,}")
print(f"Val   batches per epoch: {len(val_loader):,}")

# ── Model + optimizer ────────────────────────────────────────────────────────

model = TwoTowerModel(cfg, artifacts).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.training.lr,
    weight_decay=cfg.training.weight_decay,
)

# ── MLflow ───────────────────────────────────────────────────────────────────

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = f"two_tower_baseline_{now}"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}")
print(f"Experiment          : {EXPERIMENT_NAME}")

# ── Helpers ──────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path(cfg.training.checkpoint_dir) / EXPERIMENT_NAME
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, val_loss, run_id):
    path = CHECKPOINT_DIR / f"epoch_{epoch:02d}_val{val_loss:.4f}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "mlflow_run_id": run_id,
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")
    return path


def train_one_epoch(model, loader, optimizer, cfg, device, global_step, epoch):
    model.train()
    total_loss = 0.0
    log_loss = 0.0
    log_every = cfg.training.log_every_n_steps
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch} train", leave=True, file=sys.stdout)
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        optimizer.zero_grad()
        user_vecs, item_vecs = model(batch)
        loss = infonce_loss(
            user_vecs,
            item_vecs,
            batch["item_is_positive"],
            temperature=cfg.training.temperature,
        )
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        log_loss += loss_val
        global_step += 1

        if global_step % log_every == 0:
            avg = log_loss / log_every
            mlflow.log_metric("train/loss_step", avg, step=global_step)

            with torch.no_grad():
                scores_raw = user_vecs @ item_vecs.T  # raw cosine, no temperature
                B = user_vecs.size(0)
                pos_scores = scores_raw[torch.arange(B, device=device), torch.arange(B, device=device)]
                neg_mask = ~torch.eye(B, dtype=torch.bool, device=device)
                neg_scores = scores_raw[:, :B][neg_mask]
                gap = pos_scores.mean().item() - neg_scores.mean().item()
            mlflow.log_metric("train/pos_neg_gap_step", gap, step=global_step)

            pbar.set_postfix({"loss": f"{avg:.4f}", "gap": f"{gap:.4f}", "step": global_step})
            print(
                f"  [epoch {epoch} | step {global_step} | batch {step+1}/{n_batches}]"
                f"  loss={avg:.4f}  gap={gap:.4f}",
                flush=True,
            )
            log_loss = 0.0

            # ── Gradient / activation diagnostics (epoch 1, first log step only) ──
            if epoch == 1 and global_step == log_every:
                with torch.no_grad():
                    # 1. Embedding gradient norms — only rows seen in this batch
                    #    (table-wide mean is ~0 because 99%+ of rows have zero grad)
                    uid_grad = model.user_id_embedding.embedding.weight.grad
                    iid_grad = model.item_id_embedding.embedding.weight.grad
                    if uid_grad is not None:
                        uid_rows = uid_grad[batch["user_id"]].abs().mean()
                        print(f"  [diag] user_id emb grad  mean|abs| (active rows): {uid_rows:.6f}", flush=True)
                    else:
                        print("  [diag] user_id emb grad: None", flush=True)
                    if iid_grad is not None:
                        iid_rows = iid_grad[batch["item_ids"]].abs().mean()
                        print(f"  [diag] item_id emb grad  mean|abs| (active rows): {iid_rows:.6f}", flush=True)
                    else:
                        print("  [diag] item_id emb grad: None", flush=True)

                    # 2. Score matrix stats — is there any variance to learn from?
                    scores = user_vecs @ item_vecs.T / cfg.training.temperature
                    print(f"  [diag] scores  mean={scores.mean():.4f}  std={scores.std():.4f}  min={scores.min():.4f}  max={scores.max():.4f}", flush=True)

                    # 3. Output vector norms — are both towers actually normalizing?
                    print(f"  [diag] user_vecs norm  mean={user_vecs.norm(dim=-1).mean():.4f}", flush=True)
                    print(f"  [diag] item_vecs norm  mean={item_vecs.norm(dim=-1).mean():.4f}", flush=True)

                    # 4. History sparsity — how many real (non-pad) history items per user?
                    real_hist_len = (batch["history_item_weights"] > 0).sum(dim=1).float()
                    print(f"  [diag] real history len  mean={real_hist_len.mean():.2f}  min={real_hist_len.min():.0f}  max={real_hist_len.max():.0f}", flush=True)

                    # 5. Positive score vs mean negative score — is the model discriminating at all?
                    B = user_vecs.size(0)
                    pos_scores = scores[torch.arange(B, device=device), torch.arange(B, device=device)]
                    neg_mask = ~torch.eye(B, dtype=torch.bool, device=device)
                    neg_scores = scores[:, :B][neg_mask]
                    print(f"  [diag] pos score  mean={pos_scores.mean():.4f}  |  neg score  mean={neg_scores.mean():.4f}", flush=True)

    return total_loss / len(loader), global_step


def validate(model, loader, cfg, device):
    model.eval()
    total_loss = 0.0
    total_pos_score = 0.0
    total_neg_score = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False, file=sys.stdout):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            user_vecs, item_vecs = model(batch)
            loss = infonce_loss(
                user_vecs,
                item_vecs,
                batch["item_is_positive"],
                temperature=cfg.training.temperature,
            )
            total_loss += loss.item()

            B = user_vecs.size(0)
            scores = user_vecs @ item_vecs.T  # no temperature — raw cosine similarity
            pos_scores = scores[torch.arange(B, device=device), torch.arange(B, device=device)]
            neg_mask = ~torch.eye(B, dtype=torch.bool, device=device)
            neg_scores = scores[:, :B][neg_mask]
            total_pos_score += pos_scores.mean().item()
            total_neg_score += neg_scores.mean().item()

    n = len(loader)
    avg_pos = total_pos_score / n
    avg_neg = total_neg_score / n
    return total_loss / n, avg_pos, avg_neg


# ── Training loop ─────────────────────────────────────────────────────────────

def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key + "."))
        else:
            out[key] = v
    return out


with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow run ID: {run_id}")

    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_params(_flatten(flat_cfg))
    mlflow.log_param("num_users", artifacts["num_users"])
    mlflow.log_param("num_items", artifacts["num_items"])
    mlflow.log_param("total_params", total_params)

    global_step = 0
    best_val_loss = float("inf")
    train_start = time.time()

    for epoch in range(1, cfg.training.epochs + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"Epoch {epoch}/{cfg.training.epochs}  (global step so far: {global_step:,})", flush=True)
        print(f"{'='*60}", flush=True)

        epoch_start = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, cfg, DEVICE, global_step, epoch
        )
        train_elapsed = time.time() - epoch_start
        print(f"\n  Train loss: {train_loss:.4f}  ({train_elapsed:.0f}s)", flush=True)
        mlflow.log_metric("train/loss_epoch", train_loss, step=epoch)

        val_start = time.time()
        val_loss, val_pos, val_neg = validate(model, val_loader, cfg, DEVICE)
        val_elapsed = time.time() - val_start
        val_gap = val_pos - val_neg
        print(f"  Val   loss: {val_loss:.4f}  gap={val_gap:.4f} (pos={val_pos:.4f}, neg={val_neg:.4f})  ({val_elapsed:.0f}s)", flush=True)
        mlflow.log_metric("val/loss_epoch", val_loss, step=epoch)
        mlflow.log_metric("val/pos_neg_gap", val_gap, step=epoch)
        mlflow.log_metric("val/pos_score", val_pos, step=epoch)
        mlflow.log_metric("val/neg_score", val_neg, step=epoch)

        total_elapsed = time.time() - train_start
        print(f"  Total elapsed: {total_elapsed/60:.1f} min", flush=True)

        ckpt_path = save_checkpoint(model, optimizer, epoch, val_loss, run_id)
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            mlflow.log_metric("val/best_loss", best_val_loss, step=epoch)
            print(f"  ** New best val loss: {best_val_loss:.4f}", flush=True)

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
