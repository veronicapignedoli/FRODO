"""Data loading, normalization, masking, and sampling. No model imports.

Expects a data_root directory laid out as:
    data_root/
        patches/<lesion_id>_qsm.npy    # float32, (64,64,64)
        patches/<lesion_id>_flair.npy  # float32, (64,64,64)
        patches/<lesion_id>_mask.npy   # float32/bool, (64,64,64)
        labels.csv                     # lesion_id,label,participant_id,fold

See data/README.md for the full format specification.
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, WeightedRandomSampler

MIN_LESION_VOXELS = 110
QSM_CLIP_RANGE = (-300.0, 300.0)
FLAIR_PERCENTILES = (0.5, 99.5)


def load_labels(data_root):
    return pd.read_csv(os.path.join(data_root, "labels.csv"))


def participants_in_fold(labels_df, fold):
    return sorted(labels_df.loc[labels_df["fold"] == fold, "participant_id"].unique().tolist())


def split_train_val(participant_ids, labels_df, val_ratio, seed, max_attempts=50):
    """Participant-level split, retried until the validation set contains at
    least one Rim+ participant (otherwise PR-AUC/threshold search on val is
    undefined)."""
    participant_ids = list(participant_ids)
    positive_participants = set(
        labels_df.loc[labels_df["label"] == 1, "participant_id"].unique().tolist()
    )

    for attempt in range(max_attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + attempt)
        train_idx, val_idx = next(splitter.split(participant_ids, groups=participant_ids))
        train_ids = [participant_ids[i] for i in train_idx]
        val_ids = [participant_ids[i] for i in val_idx]
        if any(pid in positive_participants for pid in val_ids):
            return train_ids, val_ids

    raise RuntimeError("Could not create a validation split containing at least one Rim+ participant.")


class LesionPatchDataset(Dataset):
    """Paired QSM/FLAIR lesion patches for a given set of participants."""

    def __init__(self, data_root, participant_ids, augment=None):
        self.data_root = data_root
        self.patches_dir = os.path.join(data_root, "patches")
        self.augment = augment

        labels_df = load_labels(data_root)
        labels_df = labels_df[labels_df["participant_id"].isin(set(participant_ids))]

        rows = []
        n_excluded = 0
        for row in labels_df.itertuples(index=False):
            mask = np.load(self._path(row.lesion_id, "mask"))
            if mask.sum() < MIN_LESION_VOXELS:
                n_excluded += 1
                continue
            rows.append(row)

        print(f"[LesionPatchDataset] {len(participant_ids)} participants: "
              f"kept {len(rows)} lesions, excluded {n_excluded} (<{MIN_LESION_VOXELS} voxels)")

        self.rows = rows
        self.flair_stats = self._compute_flair_stats()

    def _path(self, lesion_id, kind):
        return os.path.join(self.patches_dir, f"{lesion_id}_{kind}.npy")

    def _compute_flair_stats(self):
        """Robust FLAIR percentile range, computed per participant across all
        of their lesion patches (pooled), for [0.5, 99.5] percentile clipping."""
        lesions_by_participant = {}
        for row in self.rows:
            lesions_by_participant.setdefault(row.participant_id, []).append(row.lesion_id)

        stats = {}
        for participant_id, lesion_ids in lesions_by_participant.items():
            values = np.concatenate([
                np.load(self._path(lesion_id, "flair")).ravel() for lesion_id in lesion_ids
            ])
            lo, hi = np.percentile(values, FLAIR_PERCENTILES)
            stats[participant_id] = (float(lo), float(hi))
        return stats

    def _normalize_flair(self, flair, participant_id):
        lo, hi = self.flair_stats[participant_id]
        flair = np.clip(flair, lo, hi)
        return (flair - lo) / (hi - lo + 1e-8)

    @staticmethod
    def _normalize_qsm(qsm):
        lo, hi = QSM_CLIP_RANGE
        qsm = np.clip(qsm, lo, hi)
        return (qsm - lo) / (hi - lo + 1e-8)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        qsm = np.load(self._path(row.lesion_id, "qsm")).astype(np.float32)
        flair = np.load(self._path(row.lesion_id, "flair")).astype(np.float32)
        mask = np.load(self._path(row.lesion_id, "mask")).astype(np.float32)

        qsm = self._normalize_qsm(qsm)
        flair = self._normalize_flair(flair, row.participant_id)

        qsm = qsm * (mask > 0)
        flair = flair * (mask > 0)

        if self.augment is not None:
            qsm, flair = self.augment(qsm, flair)

        qsm_t = torch.from_numpy(np.ascontiguousarray(qsm)).float().unsqueeze(0)
        flair_t = torch.from_numpy(np.ascontiguousarray(flair)).float().unsqueeze(0)
        label = torch.tensor(float(row.label))
        return qsm_t, flair_t, label, str(row.lesion_id)


def get_sampler(dataset):
    """WeightedRandomSampler built from the dataset's own class frequencies."""
    labels = np.array([row.label for row in dataset.rows], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    weight_per_class = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = weight_per_class[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
