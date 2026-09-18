#!/usr/bin/env python3
"""Stage 2: fine-tune the (optionally SSL-pretrained) backbone end to end for
binary Rim+/Rim- classification, using a composite loss (BCE + a
learned-weight supervised contrastive loss). Early stopping on val PR-AUC.
Run once per outer fold; select the held-out test fold with --fold.
"""
import argparse
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from frodo.model import MultiModalClassifier, ModelEMA
from frodo.losses import CompositeSupervisedLoss
from frodo.dataset import LesionPatchDataset, load_labels, participants_in_fold, split_train_val, get_sampler
from frodo.augmentations import RandomFlipRotate3D


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: supervised fine-tuning for Rim+/Rim- classification")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ssl_checkpoint", type=str, default=None,
                         help="Backbone checkpoint from train_ssl.py; if omitted, trains from scratch")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--base_filters", type=int, default=16)
    parser.add_argument("--inner_val_ratio", type=float, default=0.2)
    parser.add_argument("--use_ema", action="store_true",
                         help="Optional EMA of model weights; not used in paper experiments")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, criterion, loader, optimizer, device, grad_clip, ema):
    model.train()
    losses = []
    use_amp = device.type == "cuda"
    for qsm, flair, label, _lesion_id in loader:
        qsm, flair, label = qsm.to(device), flair.to(device), label.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, embedding = model(qsm, flair, return_embedding=True)
            loss, _components = criterion(logits, embedding, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_split(model, criterion, loader, device):
    model.eval()
    losses = []
    all_labels, all_probs = [], []
    use_amp = device.type == "cuda"
    for qsm, flair, label, _lesion_id in loader:
        qsm, flair, label = qsm.to(device), flair.to(device), label.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, embedding = model(qsm, flair, return_embedding=True)
            loss, _components = criterion(logits, embedding, label)
        losses.append(loss.item())
        all_labels.extend(label.detach().cpu().numpy().tolist())
        all_probs.extend(torch.sigmoid(logits.squeeze(1)).detach().float().cpu().numpy().tolist())

    val_loss = float(np.mean(losses)) if losses else float("nan")
    pr_auc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return val_loss, pr_auc


def main():
    args = parse_args()
    set_seed(args.seed)

    labels_df = load_labels(args.data_dir)
    test_ids = participants_in_fold(labels_df, args.fold)
    dev_ids = sorted(set(labels_df["participant_id"]) - set(test_ids))
    train_ids, val_ids = split_train_val(dev_ids, labels_df, args.inner_val_ratio, args.seed)

    train_dataset = LesionPatchDataset(args.data_dir, train_ids, augment=RandomFlipRotate3D())
    val_dataset = LesionPatchDataset(args.data_dir, val_ids, augment=None)

    sampler = get_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalClassifier(base_filters=args.base_filters, dropout=args.dropout).to(device)

    if args.ssl_checkpoint:
        state = torch.load(args.ssl_checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded SSL backbone from {args.ssl_checkpoint} (missing={len(missing)}, unexpected={len(unexpected)})")

    criterion = CompositeSupervisedLoss(temperature=args.tau).to(device)
    ema = ModelEMA(model) if args.use_ema else None

    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_pr_auc = -1.0
    patience_counter = 0

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device, args.grad_clip, ema)
        eval_model = ema.ema_model if ema is not None else model
        val_loss, val_pr_auc = evaluate_split(eval_model, criterion, val_loader, device)
        print(f"epoch {epoch} train_loss {train_loss:.4f} val_loss {val_loss:.4f} val_prauc {val_pr_auc:.4f}")

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_counter = 0
            torch.save(eval_model.state_dict(), os.path.join(args.output_dir, f"fold_{args.fold}.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best val PR-AUC: {best_pr_auc:.4f}")


if __name__ == "__main__":
    main()
