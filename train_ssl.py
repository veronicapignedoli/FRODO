#!/usr/bin/env python3
"""Stage 1: self-supervised pretraining of the two-stream QSM/FLAIR backbone.

No labels are used. FLAIR and QSM patches (consistently augmented) are passed
through their respective encoder streams; a lightweight MLP head phi is
trained to regress the QSM bottleneck embedding from the FLAIR bottleneck
embedding: L_SSL = || phi(z_F) - z_Q ||^2. After pretraining, phi is
discarded and only the two-stream backbone weights are saved, to be used as
the initialization for Stage 2 (train.py).
"""
import argparse
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

from frodo.model import MultiModalClassifier, ModelEMA
from frodo.losses import ProjectionMLP, ssl_regression_loss
from frodo.dataset import LesionPatchDataset, load_labels
from frodo.augmentations import RandomFlipRotate3D


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: SSL pretraining of the FRODO backbone")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--base_filters", type=int, default=16)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--use_ema", action="store_true",
                         help="Optional EMA of backbone weights; not used in paper experiments")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def split_participants(labels_df, val_ratio, seed):
    participant_ids = sorted(labels_df["participant_id"].unique().tolist())
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(participant_ids, groups=participant_ids))
    return [participant_ids[i] for i in train_idx], [participant_ids[i] for i in val_idx]


def forward_loss(model, phi, qsm, flair):
    z_qsm, z_flair = model.encode(qsm, flair)
    pred_qsm = phi(z_flair)
    return ssl_regression_loss(pred_qsm, z_qsm)


def train_one_epoch(model, phi, loader, optimizer, device, ema):
    model.train()
    phi.train()
    losses = []
    use_amp = device.type == "cuda"
    for qsm, flair, _label, _lesion_id in loader:
        qsm, flair = qsm.to(device), flair.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = forward_loss(model, phi, qsm, flair)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model, phi, loader, device):
    model.eval()
    phi.eval()
    losses = []
    use_amp = device.type == "cuda"
    for qsm, flair, _label, _lesion_id in loader:
        qsm, flair = qsm.to(device), flair.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = forward_loss(model, phi, qsm, flair)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def backbone_state_dict(model):
    """Only the two-stream encoder weights (qsm_stem, flair_stem) — the parts
    actually exercised during SSL pretraining. Everything downstream of the
    FiLM fusion (film, fuse_conv, block3/4, classifier) is untouched by
    Stage 1 and deliberately excluded, so Stage 2 loads it with
    strict=False and keeps its own fresh initialization there."""
    state = {}
    state.update({f"qsm_stem.{k}": v for k, v in model.qsm_stem.state_dict().items()})
    state.update({f"flair_stem.{k}": v for k, v in model.flair_stem.state_dict().items()})
    return state


def main():
    args = parse_args()
    set_seed(args.seed)

    labels_df = load_labels(args.data_dir)
    train_ids, val_ids = split_participants(labels_df, args.val_ratio, args.seed)

    train_dataset = LesionPatchDataset(args.data_dir, train_ids, augment=RandomFlipRotate3D())
    val_dataset = LesionPatchDataset(args.data_dir, val_ids, augment=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalClassifier(base_filters=args.base_filters).to(device)

    with torch.no_grad():
        probe_qsm, probe_flair, _, _ = next(iter(train_loader))
        embed_dim = model.encode(probe_qsm.to(device), probe_flair.to(device))[0].shape[1]
    phi = ProjectionMLP(embed_dim).to(device)

    ema = ModelEMA(model) if args.use_ema else None
    optimizer = AdamW(list(model.parameters()) + list(phi.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, phi, train_loader, optimizer, device, ema)
        eval_model = ema.ema_model if ema is not None else model
        val_loss = evaluate(eval_model, phi, val_loader, device)
        print(f"epoch {epoch} train_loss {train_loss:.6f} val_loss {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(backbone_state_dict(eval_model), os.path.join(args.output_dir, "best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
