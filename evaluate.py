#!/usr/bin/env python3
"""Load one trained checkpoint per fold (fold_0.pt ... fold_4.pt), run
inference on each fold's held-out test lesions, select a decision threshold
on that fold's validation lesions (grid search over t in [0,1], step 0.01,
maximizing F1), and aggregate lesion-level and person-level metrics across
all 5 folds.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from frodo.model import MultiModalClassifier
from frodo.dataset import LesionPatchDataset, load_labels, participants_in_fold, split_train_val


def parse_args():
    parser = argparse.ArgumentParser(description="5-fold evaluation: lesion-level and person-level metrics")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True,
                         help="Directory containing fold_0.pt ... fold_4.pt (from train.py)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_filters", type=int, default=16, help="Must match the value used in train.py")
    parser.add_argument("--dropout", type=float, default=0.5, help="Must match the value used in train.py")
    parser.add_argument("--inner_val_ratio", type=float, default=0.2, help="Must match the value used in train.py")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="Must match the value used in train.py")
    return parser.parse_args()


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    labels, probs, lesion_ids = [], [], []
    for qsm, flair, label, lesion_id in loader:
        qsm, flair = qsm.to(device), flair.to(device)
        prob = torch.sigmoid(model(qsm, flair).squeeze(1))
        labels.extend(label.numpy().tolist())
        probs.extend(prob.detach().cpu().numpy().tolist())
        lesion_ids.extend(lesion_id)
    return np.array(labels), np.array(probs), np.array(lesion_ids)


def best_f1_threshold(labels, probs):
    thresholds = np.arange(0.0, 1.001, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs > t).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t


def specificity_score(labels, preds):
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_df = load_labels(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    lesion_level_rows = []
    person_rows = []
    roc_curve_rows = []
    pr_curve_rows = []

    for fold in range(5):
        ckpt_path = os.path.join(args.checkpoints_dir, f"fold_{fold}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        test_ids = participants_in_fold(labels_df, fold)
        dev_ids = sorted(set(labels_df["participant_id"]) - set(test_ids))
        _train_ids, val_ids = split_train_val(dev_ids, labels_df, args.inner_val_ratio, args.seed)

        val_dataset = LesionPatchDataset(args.data_dir, val_ids, augment=None)
        test_dataset = LesionPatchDataset(args.data_dir, test_ids, augment=None)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        model = MultiModalClassifier(base_filters=args.base_filters, dropout=args.dropout).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=True)

        val_labels, val_probs, _ = infer(model, val_loader, device)
        threshold = best_f1_threshold(val_labels, val_probs)

        test_labels, test_probs, test_lesion_ids = infer(model, test_loader, device)
        test_preds = (test_probs > threshold).astype(int)
        participant_ids = labels_df.set_index("lesion_id").loc[test_lesion_ids, "participant_id"].to_numpy()

        has_both_classes = len(set(test_labels)) > 1
        roc_auc = roc_auc_score(test_labels, test_probs) if has_both_classes else float("nan")
        pr_auc = average_precision_score(test_labels, test_probs) if has_both_classes else float("nan")

        lesion_level_rows.append({
            "fold": fold,
            "threshold": threshold,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": accuracy_score(test_labels, test_preds),
            "f1": f1_score(test_labels, test_preds, zero_division=0),
            "sensitivity": recall_score(test_labels, test_preds, zero_division=0),
            "specificity": specificity_score(test_labels, test_preds),
            "ppv": precision_score(test_labels, test_preds, zero_division=0),
        })

        for pid in sorted(set(participant_ids)):
            pmask = participant_ids == pid
            person_rows.append({
                "fold": fold,
                "participant_id": pid,
                "gold_positive": int(np.any(test_labels[pmask] == 1)),
                "pred_positive": int(np.any(test_preds[pmask] == 1)),
                "gold_count": int(np.sum(test_labels[pmask] == 1)),
                "pred_count": int(np.sum(test_preds[pmask] == 1)),
            })

        if has_both_classes:
            fpr, tpr, _ = roc_curve(test_labels, test_probs)
            roc_curve_rows.extend({"fold": fold, "fpr": f, "tpr": t} for f, t in zip(fpr, tpr))
            precision, recall, _ = precision_recall_curve(test_labels, test_probs)
            pr_curve_rows.extend({"fold": fold, "precision": p, "recall": r} for p, r in zip(precision, recall))

    lesion_df = pd.DataFrame(lesion_level_rows)
    person_df = pd.DataFrame(person_rows)

    print("\n=== Lesion-level metrics (mean +/- std across folds) ===")
    for key in ["roc_auc", "pr_auc", "accuracy", "f1", "sensitivity", "specificity", "ppv"]:
        print(f"  {key}: {lesion_df[key].mean():.3f} +/- {lesion_df[key].std():.3f}")

    gold_positive = person_df["gold_positive"].to_numpy()
    pred_positive = person_df["pred_positive"].to_numpy()
    pearson_r, _ = pearsonr(person_df["gold_count"], person_df["pred_count"])

    print("\n=== Person-level metrics ===")
    print(f"  sensitivity: {recall_score(gold_positive, pred_positive, zero_division=0):.3f}")
    print(f"  specificity: {specificity_score(gold_positive, pred_positive):.3f}")
    print(f"  accuracy: {accuracy_score(gold_positive, pred_positive):.3f}")
    print(f"  pearson_r (predicted vs. gold Rim+ count per person): {pearson_r:.3f}")

    lesion_df.to_csv(os.path.join(args.output_dir, "lesion_level_metrics.csv"), index=False)
    person_df.to_csv(os.path.join(args.output_dir, "person_level_predictions.csv"), index=False)
    if roc_curve_rows:
        pd.DataFrame(roc_curve_rows).to_csv(os.path.join(args.output_dir, "roc_curve.csv"), index=False)
    if pr_curve_rows:
        pd.DataFrame(pr_curve_rows).to_csv(os.path.join(args.output_dir, "pr_curve.csv"), index=False)


if __name__ == "__main__":
    main()
