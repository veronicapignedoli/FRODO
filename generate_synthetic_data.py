#!/usr/bin/env python3
"""Generate a synthetic dataset with the same directory layout and file
format as the real FRODO cohort (see data/README.md), so the full pipeline
(train_ssl.py -> train.py -> evaluate.py) can be exercised end to end
without any real clinical data. No model imports."""
import argparse
import os

import numpy as np
import pandas as pd

PATCH_SIZE = (64, 64, 64)
MIN_LESION_VOXELS = 110


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic FRODO patches for smoke-testing the pipeline")
    parser.add_argument("--output_dir", type=str, default="data/synthetic")
    parser.add_argument("--n_lesions", type=int, default=100)
    parser.add_argument("--rim_pos_rate", type=float, default=0.07)
    parser.add_argument("--n_participants", type=int, default=20)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def random_ellipsoid_mask(rng, min_voxels=MIN_LESION_VOXELS):
    """A random ellipsoid blob (>= min_voxels), placed with margin from the
    patch border so it is never truncated."""
    radii = rng.uniform(4.0, 9.0, size=3)
    center = np.array([rng.uniform(r + 2, s - r - 2) for r, s in zip(radii, PATCH_SIZE)])
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in PATCH_SIZE], indexing="ij")
    dist = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    )
    mask = (dist <= 1.0).astype(np.float32)
    if mask.sum() < min_voxels:
        return random_ellipsoid_mask(rng, min_voxels)
    return mask, center


def make_patch_pair(rng, is_rim_positive):
    mask, center = random_ellipsoid_mask(rng)

    flair = rng.normal(loc=600.0, scale=80.0, size=PATCH_SIZE).astype(np.float32)
    flair += mask * rng.uniform(150.0, 300.0)
    flair = np.clip(flair, 0, None)

    qsm = rng.normal(loc=0.0, scale=15.0, size=PATCH_SIZE).astype(np.float32)

    if is_rim_positive:
        # Small high-susceptibility ring at the lesion center: a paramagnetic
        # rim analogue, giving the model a learnable signal for smoke tests.
        zz, yy, xx = np.meshgrid(*[np.arange(s) for s in PATCH_SIZE], indexing="ij")
        dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
        ring = (dist > 3.0) & (dist < 5.0)
        qsm[ring] += rng.uniform(120.0, 200.0)

    return qsm.astype(np.float32), flair.astype(np.float32), mask.astype(np.float32)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    patches_dir = os.path.join(args.output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    participant_ids = [f"P{idx:03d}" for idx in range(args.n_participants)]
    participant_fold = {pid: idx % args.n_folds for idx, pid in enumerate(participant_ids)}

    n_pos = int(np.clip(max(round(args.n_lesions * args.rim_pos_rate), args.n_folds), 0, args.n_lesions))
    labels = np.zeros(args.n_lesions, dtype=int)
    labels[:n_pos] = 1
    rng.shuffle(labels)

    lesion_participants = rng.choice(participant_ids, size=args.n_lesions)

    # Guarantee every fold has at least one Rim+ lesion, so downstream
    # threshold search / ROC-AUC never hits a single-class fold.
    pos_idx = np.flatnonzero(labels == 1)
    for fold_id, idx in zip(range(args.n_folds), pos_idx):
        candidates = [p for p in participant_ids if participant_fold[p] == fold_id]
        lesion_participants[idx] = rng.choice(candidates)

    rows = []
    for lesion_idx in range(args.n_lesions):
        lesion_id = f"L{lesion_idx:04d}"
        participant_id = str(lesion_participants[lesion_idx])
        label = int(labels[lesion_idx])

        qsm, flair, mask = make_patch_pair(rng, is_rim_positive=bool(label))
        np.save(os.path.join(patches_dir, f"{lesion_id}_qsm.npy"), qsm)
        np.save(os.path.join(patches_dir, f"{lesion_id}_flair.npy"), flair)
        np.save(os.path.join(patches_dir, f"{lesion_id}_mask.npy"), mask)

        rows.append({
            "lesion_id": lesion_id,
            "label": label,
            "participant_id": participant_id,
            "fold": participant_fold[participant_id],
        })

    labels_df = pd.DataFrame(rows)
    labels_df.to_csv(os.path.join(args.output_dir, "labels.csv"), index=False)

    print(f"Generated {args.n_lesions} synthetic lesions in {args.output_dir}")
    print(f"  Rim+: {int(labels_df['label'].sum())} | Rim-: {int((labels_df['label'] == 0).sum())}")
    print(f"  Participants: {args.n_participants} | Folds: {args.n_folds}")
    for fold_id in range(args.n_folds):
        fold_df = labels_df[labels_df["fold"] == fold_id]
        print(f"    fold {fold_id}: {len(fold_df)} lesions, {int(fold_df['label'].sum())} Rim+")


if __name__ == "__main__":
    main()
