"""
Standalone self-supervised pretraining script for 3D multimodal MRI patches.

Usage examples:
  python pretrain_selfsup.py --dataset_type HSMn --modalities FLAIR,QSM --fold 0 \
      --epochs 200 --batch_size 10 --learning_rate 5e-5 --ssl_mode contrastive

  python pretrain_selfsup.py --dataset_type HSMn --modalities FLAIR,QSM --fold 0 \
      --epochs 200 --batch_size 10 --learning_rate 5e-5 --ssl_mode multimodal \
      --pred_direction flair_to_qsm --stop_grad_target

Manual downstream usage from main.py:
  model = MultiModalClassifier(in_channels_list=[1, 1], base_filters=16, dropout=0.3)
  state = torch.load(".../backbone_pretrained.pt", map_location="cpu")
  model.load_state_dict(state, strict=False)

Notes:
- This script does not modify or call the supervised training pipeline.
- It uses only TRAIN subjects by default to avoid leakage.
"""

import os
import json
import time
import random
import argparse
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from sklearn.model_selection import GroupShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from architecture import MultiModalClassifier
from data_generator import get_patch_dataloaders


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining for 3D multimodal patch classifier")
    parser.add_argument("--dataset_type", type=str, choices=["HSMn"], default="HSMn")
    parser.add_argument("--modalities", type=str, default="FLAIR,QSM")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cv5", action="store_true", help="Run sequentially on all 5 outer CV folds")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ssl_mode", type=str, choices=["contrastive", "multimodal"], default="contrastive")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--pred_direction", type=str, choices=["flair_to_qsm", "qsm_to_flair"], default="flair_to_qsm")
    parser.add_argument("--stop_grad_target", action="store_true")
    parser.add_argument("--use_all_unlabeled", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--sanity_checks",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable lightweight runtime checks (pass false to disable).",
    )
    parser.add_argument(
        "--sanity_checks_max_batches",
        type=int,
        default=2,
        help="Run sanity checks only on the first N batches per epoch.",
    )
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--wandb_project", type=str, default="Classification-w-QSM")
    parser.add_argument("--wandb_entity", type=str, default="RIM-project")
    parser.add_argument("--experiment", type=str, default="ssl_pretrain")
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataset_info(dataset_type):
    if dataset_type == "HSMn":
        return "/data/cil/veronica/HSMn"
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_labels(patches_path):
    labels_path = os.path.join(patches_path, "labels.json")
    with open(labels_path, "r") as f:
        return json.load(f)


def get_outer_folds():
    return {
        0: ["006", "009", "011", "026", "039", "042", "056", "063", "076", "081", "093", "112", "126", "156"],
        1: ["008", "010", "021", "027", "029", "071", "073", "086", "092", "101", "109", "117", "127", "149", "178"],
        2: ["004", "018", "024", "046", "049", "051", "065", "067", "074", "078", "079", "097", "099", "103", "115", "134"],
        3: ["007", "025", "028", "034", "035", "037", "045", "068", "082", "083", "088", "107", "108", "110", "118", "172"],
        4: ["002", "003", "015", "017", "032", "043", "052", "055", "080", "084", "090", "096", "098", "131", "135", "142"],
    }


def get_subject_phase_map(labels_dict):
    subject_phase_map = {}
    for phase, phase_dict in labels_dict.items():
        for subj in phase_dict:
            subject_phase_map[subj] = phase
    return subject_phase_map


def subject_has_positive(subj, labels_dict, subject_phase_map):
    phase = subject_phase_map.get(subj)
    if phase is None:
        return False
    for label in labels_dict.get(phase, {}).get(subj, {}).values():
        if label == 1:
            return True
    return False


def split_dev_subjects(dev_subjects, labels_dict, subject_phase_map, seed, val_ratio=0.2, max_attempts=50):
    dev_subjects = list(dev_subjects)
    for attempt in range(max_attempts):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + attempt)
        train_idx, val_idx = next(gss.split(dev_subjects, groups=dev_subjects))
        train_subjects = [dev_subjects[i] for i in train_idx]
        val_subjects = [dev_subjects[i] for i in val_idx]
        if any(subject_has_positive(s, labels_dict, subject_phase_map) for s in val_subjects):
            return train_subjects, val_subjects
    raise ValueError("Unable to create a validation split with at least one positive subject.")


def build_subject_splits(labels_dict, fold, seed, use_all_unlabeled):
    folds = get_outer_folds()
    if fold not in folds:
        raise ValueError("fold must be in [0,1,2,3,4]")

    subject_phase_map = get_subject_phase_map(labels_dict)

    test_subjects = folds[fold]
    dev_subjects = []
    for k, fold_subjects in folds.items():
        if k != fold:
            dev_subjects.extend(fold_subjects)

    train_subjects, val_subjects = split_dev_subjects(
        dev_subjects=dev_subjects,
        labels_dict=labels_dict,
        subject_phase_map=subject_phase_map,
        seed=seed,
        val_ratio=0.2,
    )

    if use_all_unlabeled:
        all_subjects = []
        for fold_subjects in folds.values():
            all_subjects.extend(fold_subjects)
        all_subjects = sorted(set(all_subjects))
        print("\n" + "!" * 80)
        print("WARNING: --use_all_unlabeled enabled. TRAINING ON ALL SUBJECTS (possible leakage).")
        print("!" * 80 + "\n")
        return {
            "train": all_subjects,
            "val": [],
            "test": test_subjects,
        }, subject_phase_map

    return {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }, subject_phase_map


def reorder_modalities_for_model(volumes, modalities, return_info=False):
    """
    Reorder incoming modality list to [FLAIR, QSM] expected by MultiModalClassifier in architecture.py.
    If one modality is missing in batch, it is replaced by zeros with a compatible shape.
    """
    mod_to_tensor = {m: t for m, t in zip(modalities, volumes)}
    ref = None
    for t in volumes:
        if t is not None:
            ref = t
            break
    if ref is None:
        raise RuntimeError("No valid modality tensors in batch.")

    flair = mod_to_tensor.get("FLAIR")
    qsm = mod_to_tensor.get("QSM")
    missing_modalities = []

    if flair is None:
        missing_modalities.append("FLAIR")
        print("[WARN] FLAIR modality missing in batch; replacing with zeros for compatibility.")
        flair = torch.zeros_like(ref)
    if qsm is None:
        missing_modalities.append("QSM")
        print("[WARN] QSM modality missing in batch; replacing with zeros for compatibility.")
        qsm = torch.zeros_like(ref)

    ordered = [flair, qsm]
    if return_info:
        return ordered, {"missing_modalities": missing_modalities}
    return ordered


def _assert_modalities_aligned(vol_list, modalities):
    if len(vol_list) != len(modalities):
        raise AssertionError(f"Modality mismatch: tensors={len(vol_list)} vs names={len(modalities)}")
    if len(vol_list) == 0:
        raise AssertionError("Empty modality tensor list.")

    base_shape = vol_list[0].shape
    base_dtype = vol_list[0].dtype
    base_device = vol_list[0].device

    for idx, tensor in enumerate(vol_list):
        if tensor is None:
            raise AssertionError(f"Modality tensor at index {idx} is None.")
        if tensor.ndim != 5:
            raise AssertionError(f"Expected 5D tensor (B,1,D,H,W), got shape {tuple(tensor.shape)}")
        if tensor.shape != base_shape:
            raise AssertionError(
                f"Shape mismatch across modalities: {tuple(tensor.shape)} vs {tuple(base_shape)}"
            )
        if tensor.dtype != base_dtype:
            raise AssertionError(f"Dtype mismatch across modalities: {tensor.dtype} vs {base_dtype}")
        if tensor.device != base_device:
            raise AssertionError(f"Device mismatch across modalities: {tensor.device} vs {base_device}")


def _invert_meta_transform(volumes, meta):
    restored = [v.clone() for v in volumes]
    if meta.get("do_rot", False):
        k = int(meta["rot_k"])
        axes = tuple(meta["rot_axes"])
        restored = [torch.rot90(v, k=(-k) % 4, dims=axes) for v in restored]

    if meta.get("flip_w", False):
        restored = [torch.flip(v, dims=[4]) for v in restored]
    if meta.get("flip_h", False):
        restored = [torch.flip(v, dims=[3]) for v in restored]
    if meta.get("flip_d", False):
        restored = [torch.flip(v, dims=[2]) for v in restored]
    return restored


def _check_within_view_alignment(original, augmented, name, meta):
    restored = _invert_meta_transform(augmented, meta)
    for idx, (orig, rec) in enumerate(zip(original, restored)):
        if orig.shape != rec.shape:
            raise AssertionError(
                f"{name}: restored shape mismatch at modality {idx}: {tuple(rec.shape)} vs {tuple(orig.shape)}"
            )
        if not torch.equal(orig, rec):
            raise AssertionError(
                f"{name}: modality {idx} failed transform-consistency check (possible within-view misalignment)."
            )


def _assert_model_input_order(model_inputs):
    if not isinstance(model_inputs, list) or len(model_inputs) != 2:
        raise AssertionError("Model inputs must be a list [FLAIR, QSM].")
    flair, qsm = model_inputs
    if flair is None or qsm is None:
        raise AssertionError("Model inputs must contain both FLAIR and QSM tensors.")
    if flair.shape != qsm.shape:
        raise AssertionError(f"FLAIR/QSM shape mismatch after reorder: {tuple(flair.shape)} vs {tuple(qsm.shape)}")


def _assert_finite_and_plausible(vol_list, modalities, name):
    for modality_name, tensor in zip(modalities, vol_list):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name}: {modality_name} contains NaN/Inf values.")
        min_v = tensor.amin().item()
        max_v = tensor.amax().item()
        if not (math.isfinite(min_v) and math.isfinite(max_v)):
            raise AssertionError(f"{name}: {modality_name} min/max are not finite.")


def random_3d_view(volumes, return_meta=False):
    """
    Apply lightweight stochastic 3D spatial-only transforms consistently across modalities.
    The same random decisions are shared across all modalities in the view to preserve
    voxel correspondence (critical for multimodal paired inputs).
    Input/Output: list of tensors, each (B,1,D,H,W)
    """
    out = [v.clone() for v in volumes]

    flip_d = random.random() > 0.5
    flip_h = random.random() > 0.5
    flip_w = random.random() > 0.5

    do_rot = random.random() > 0.5
    rot_k = random.randint(0, 3) if do_rot else 0
    rot_axes = random.choice([(2, 3), (2, 4), (3, 4)]) if do_rot else (2, 3)

    meta = {
        "flip_d": flip_d,
        "flip_h": flip_h,
        "flip_w": flip_w,
        "do_rot": do_rot,
        "rot_k": rot_k,
        "rot_axes": rot_axes,
    }

    if flip_d:
        out = [torch.flip(v, dims=[2]) for v in out]
    if flip_h:
        out = [torch.flip(v, dims=[3]) for v in out]
    if flip_w:
        out = [torch.flip(v, dims=[4]) for v in out]

    if do_rot:
        out = [torch.rot90(v, k=rot_k, dims=rot_axes) for v in out]

    return (out, meta) if return_meta else out


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossModalPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class ModalityFeatureTap:
    """
    Non-invasive feature extractor using forward hooks on architecture.py bottlenecks.
    Captures modality-specific features without changing supervised code paths.
    """
    def __init__(self, model):
        self.flair = None
        self.qsm = None
        #self.h1 = model.flair_stage2.register_forward_hook(self._save_flair)
        #self.h2 = model.qsm_stage2.register_forward_hook(self._save_qsm)
        self.h1 = model.flair_stage2b.register_forward_hook(self._save_flair)
        self.h2 = model.qsm_stage2b.register_forward_hook(self._save_qsm)

    def _save_flair(self, module, inp, out):
        self.flair = out

    def _save_qsm(self, module, inp, out):
        self.qsm = out

    def clear(self):
        self.flair = None
        self.qsm = None

    def close(self):
        self.h1.remove()
        self.h2.remove()


def nt_xent_loss(z1, z2, tau=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)

    logits = reps @ reps.t()
    logits = logits / tau

    n = z1.size(0)
    total = 2 * n
    diag_mask = torch.eye(total, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(diag_mask, -1e9)

    targets = torch.arange(total, device=logits.device)
    targets = (targets + n) % total
    return F.cross_entropy(logits, targets)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def safe_wandb_log(payload):
    if wandb.run is not None:
        wandb.log(payload)


def run_ssl_epoch(
    model,
    loader,
    optimizer,
    device,
    ssl_mode,
    modalities,
    tau,
    projection_head=None,
    predictor=None,
    pred_direction="flair_to_qsm",
    stop_grad_target=False,
    train=True,
    dry_run=False,
    sanity_checks=True,
    sanity_checks_max_batches=2,
):
    if train:
        model.train()
        if projection_head is not None:
            projection_head.train()
        if predictor is not None:
            predictor.train()
    else:
        model.eval()
        if projection_head is not None:
            projection_head.eval()
        if predictor is not None:
            predictor.eval()

    losses = []
    emb_norms = []
    collapse_vars = []
    align_cosines = []
    reg_errors = []
    aug_counts = {
        "flip_d": 0,
        "flip_h": 0,
        "flip_w": 0,
        "rot": 0,
    }

    tap = ModalityFeatureTap(model) if ssl_mode == "multimodal" else None

    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        for step, (volumes, _labels, _infos) in enumerate(loader):
            if not isinstance(volumes, list) or len(volumes) == 0:
                continue

            volumes = [v.to(device, non_blocking=True) for v in volumes]
            if len(volumes) != len(modalities):
                print(f"[WARN] Skipping batch with modality mismatch: got {len(volumes)}, expected {len(modalities)}")
                continue

            do_checks = sanity_checks and step < sanity_checks_max_batches
            if do_checks:
                _assert_modalities_aligned(volumes, modalities)
                _assert_finite_and_plausible(volumes, modalities, name="pre-augment")

            if train:
                optimizer.zero_grad(set_to_none=True)

            if ssl_mode == "contrastive":
                v1, meta1 = random_3d_view(volumes, return_meta=True)
                v2, meta2 = random_3d_view(volumes, return_meta=True)
                m1, m1_info = reorder_modalities_for_model(v1, modalities, return_info=True)
                m2, m2_info = reorder_modalities_for_model(v2, modalities, return_info=True)

                if do_checks:
                    if "FLAIR" not in modalities or "QSM" not in modalities:
                        raise AssertionError("Contrastive mode requires modalities containing both FLAIR and QSM.")
                    _assert_modalities_aligned(v1, modalities)
                    _assert_modalities_aligned(v2, modalities)
                    _check_within_view_alignment(volumes, v1, name="view1", meta=meta1)
                    _check_within_view_alignment(volumes, v2, name="view2", meta=meta2)
                    _assert_model_input_order(m1)
                    _assert_model_input_order(m2)
                    _assert_finite_and_plausible(m1, ["FLAIR", "QSM"], name="model-input-view1")
                    _assert_finite_and_plausible(m2, ["FLAIR", "QSM"], name="model-input-view2")

                    flair_idx = modalities.index("FLAIR")
                    qsm_idx = modalities.index("QSM")
                    if not torch.equal(m1[0], v1[flair_idx]) or not torch.equal(m1[1], v1[qsm_idx]):
                        raise AssertionError("Model input order mismatch in view1: expected [FLAIR, QSM].")
                    if not torch.equal(m2[0], v2[flair_idx]) or not torch.equal(m2[1], v2[qsm_idx]):
                        raise AssertionError("Model input order mismatch in view2: expected [FLAIR, QSM].")
                    if len(m1_info["missing_modalities"]) > 0 or len(m2_info["missing_modalities"]) > 0:
                        raise RuntimeError(
                            "Missing modality detected after reorder (zero-filled tensor). "
                            "Contrastive mode requires both FLAIR and QSM."
                        )

                if meta1["flip_d"]:
                    aug_counts["flip_d"] += 1
                if meta1["flip_h"]:
                    aug_counts["flip_h"] += 1
                if meta1["flip_w"]:
                    aug_counts["flip_w"] += 1
                if meta1["do_rot"]:
                    aug_counts["rot"] += 1
                if meta2["flip_d"]:
                    aug_counts["flip_d"] += 1
                if meta2["flip_h"]:
                    aug_counts["flip_h"] += 1
                if meta2["flip_w"]:
                    aug_counts["flip_w"] += 1
                if meta2["do_rot"]:
                    aug_counts["rot"] += 1

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _logits1, emb1 = model(m1, return_features=True)
                    _logits2, emb2 = model(m2, return_features=True)
                    z1 = projection_head(emb1)
                    z2 = projection_head(emb2)
                    loss = nt_xent_loss(z1, z2, tau=tau)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(projection_head.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()

                with torch.no_grad():
                    emb_norm = emb1.float().norm(dim=1).mean().item()
                    var_mean = emb1.float().var(dim=0, unbiased=False).mean().item()

                emb_norms.append(emb_norm)
                collapse_vars.append(var_mean)
                losses.append(loss.item())

                if dry_run:
                    print("[DRY-RUN][contrastive]")
                    print(f"  view1 meta: {meta1}")
                    print(f"  view2 meta: {meta2}")
                    print(f"  view1 modality tensors: {[tuple(x.shape) for x in m1]}")
                    print(f"  emb shape: {tuple(emb1.shape)} | proj shape: {tuple(z1.shape)} | loss: {loss.item():.6f}")
                    break

            elif ssl_mode == "multimodal":
                if "FLAIR" not in modalities or "QSM" not in modalities:
                    raise ValueError("Multimodal SSL requires modalities containing both FLAIR and QSM.")

                vm, meta_m = random_3d_view(volumes, return_meta=True)
                model_inputs = reorder_modalities_for_model(vm, modalities)
                tap.clear()

                if do_checks:
                    _assert_modalities_aligned(vm, modalities)
                    _check_within_view_alignment(volumes, vm, name="multimodal-view", meta=meta_m)
                    _assert_model_input_order(model_inputs)
                    flair_idx = modalities.index("FLAIR")
                    qsm_idx = modalities.index("QSM")
                    if not torch.equal(model_inputs[0], vm[flair_idx]) or not torch.equal(model_inputs[1], vm[qsm_idx]):
                        raise AssertionError("Model input order mismatch in multimodal mode: expected [FLAIR, QSM].")
                    _assert_finite_and_plausible(model_inputs, ["FLAIR", "QSM"], name="model-input-multimodal")

                if meta_m["flip_d"]:
                    aug_counts["flip_d"] += 1
                if meta_m["flip_h"]:
                    aug_counts["flip_h"] += 1
                if meta_m["flip_w"]:
                    aug_counts["flip_w"] += 1
                if meta_m["do_rot"]:
                    aug_counts["rot"] += 1

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(model_inputs)

                    if tap.flair is None or tap.qsm is None:
                        raise RuntimeError("Failed to capture modality bottleneck features via hooks.")

                    flair_emb = F.adaptive_avg_pool3d(tap.flair, (1, 1, 1)).flatten(1)
                    qsm_emb = F.adaptive_avg_pool3d(tap.qsm, (1, 1, 1)).flatten(1)

                    if predictor is None:
                        raise RuntimeError("Predictor head is required in multimodal mode.")

                    if pred_direction == "flair_to_qsm":
                        src = flair_emb
                        tgt = qsm_emb
                    else:
                        src = qsm_emb
                        tgt = flair_emb

                    pred = predictor(src)
                    target = tgt.detach() if stop_grad_target else tgt
                    loss = F.mse_loss(pred, target)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(predictor.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()

                with torch.no_grad():
                    align = F.cosine_similarity(src.float(), tgt.float(), dim=1).mean().item()
                    err = (pred.float() - target.float()).pow(2).mean(dim=1)
                    reg_err = err.mean().item()
                    emb_norm = src.float().norm(dim=1).mean().item()
                    var_mean = src.float().var(dim=0, unbiased=False).mean().item()

                align_cosines.append(align)
                reg_errors.extend(err.detach().cpu().tolist())
                emb_norms.append(emb_norm)
                collapse_vars.append(var_mean)
                losses.append(loss.item())

                if dry_run:
                    print("[DRY-RUN][multimodal]")
                    print(f"  view meta: {meta_m}")
                    print(f"  modality tensors: {[tuple(x.shape) for x in model_inputs]}")
                    print(f"  flair_emb: {tuple(flair_emb.shape)} | qsm_emb: {tuple(qsm_emb.shape)}")
                    print(f"  pred shape: {tuple(pred.shape)} | loss: {loss.item():.6f}")
                    break

            else:
                raise ValueError(f"Unknown ssl_mode: {ssl_mode}")

    if tap is not None:
        tap.close()

    if len(losses) == 0:
        return {
            "loss": np.nan,
            "emb_norm": np.nan,
            "collapse_var": np.nan,
            "align_cos": np.nan,
            "reg_err_mean": np.nan,
            "reg_err_std": np.nan,
            "reg_err_p50": np.nan,
            "reg_err_p90": np.nan,
            "aug_flip_d_count": 0,
            "aug_flip_h_count": 0,
            "aug_flip_w_count": 0,
            "aug_rot_count": 0,
        }

    reg_errors_np = np.array(reg_errors) if len(reg_errors) > 0 else np.array([np.nan])
    return {
        "loss": float(np.mean(losses)),
        "emb_norm": float(np.mean(emb_norms)) if emb_norms else np.nan,
        "collapse_var": float(np.mean(collapse_vars)) if collapse_vars else np.nan,
        "align_cos": float(np.mean(align_cosines)) if align_cosines else np.nan,
        "reg_err_mean": float(np.nanmean(reg_errors_np)),
        "reg_err_std": float(np.nanstd(reg_errors_np)),
        "reg_err_p50": float(np.nanpercentile(reg_errors_np, 50)),
        "reg_err_p90": float(np.nanpercentile(reg_errors_np, 90)),
        "aug_flip_d_count": int(aug_counts["flip_d"]),
        "aug_flip_h_count": int(aug_counts["flip_h"]),
        "aug_flip_w_count": int(aug_counts["flip_w"]),
        "aug_rot_count": int(aug_counts["rot"]),
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val, args, projection_head=None, predictor=None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val,
        "args": vars(args),
    }
    if projection_head is not None:
        payload["projection_head_state_dict"] = projection_head.state_dict()
    if predictor is not None:
        payload["predictor_state_dict"] = predictor.state_dict()
    torch.save(payload, path)


def main(args):
    setup_seed(args.seed)

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if len(modalities) < 2:
        raise ValueError("Provide at least two modalities (expected FLAIR,QSM).")
    if "FLAIR" not in modalities or "QSM" not in modalities:
        raise ValueError("Current backbone expects FLAIR and QSM modalities.")

    patches_path = "/data/cil/veronica/3DRim_Classification_patches"
    _ = get_dataset_info(args.dataset_type)

    if not os.path.exists(os.path.join(patches_path, "labels.json")):
        raise FileNotFoundError(f"labels.json not found in {patches_path}")

    labels_dict = load_labels(patches_path)
    subjects, subject_phase_map = build_subject_splits(
        labels_dict=labels_dict,
        fold=args.fold,
        seed=args.seed,
        use_all_unlabeled=args.use_all_unlabeled,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("pretrain_outputs", f"{timestamp}_{args.ssl_mode}_fold{args.fold}")
    run_name = f"{args.experiment}_{args.ssl_mode}_fold{args.fold}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "split_subjects.json"), "w") as f:
        json.dump(subjects, f, indent=2)

    train_loader, val_loader, _test_loader = get_patch_dataloaders(
        patches_path=patches_path,
        labels_dict=labels_dict,
        subjects=subjects,
        modalities=modalities,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        oversample_rim=False,
        balance_classes=False,
        weighted_sampler=False,
        subject_phase_map=subject_phase_map,
    )

    has_val = len(subjects["val"]) > 0 and len(val_loader) > 0

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script requires CUDA for torch.amp.autocast with bfloat16.")
    print(f"Using device: {device}")
    print(f"SSL mode: {args.ssl_mode}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = MultiModalClassifier(
        in_channels_list=[1, 1],
        base_filters=args.filters,
        num_classes=1,
        dropout=args.dropout,
    ).to(device)

    with torch.no_grad():
        probe = next(iter(train_loader))[0]
        probe = [t.to(device) for t in probe]
        probe = reorder_modalities_for_model(probe, modalities)
        _, emb_probe = model(probe, return_features=True)
        emb_dim = emb_probe.shape[1]

        multimodal_dim = None
        if args.ssl_mode == "multimodal":
            tap_probe = ModalityFeatureTap(model)
            tap_probe.clear()
            _ = model(probe)
            flair_probe = F.adaptive_avg_pool3d(tap_probe.flair, (1, 1, 1)).flatten(1)
            multimodal_dim = flair_probe.shape[1]
            tap_probe.close()

    projection_head = ProjectionHead(emb_dim, args.proj_dim).to(device) if args.ssl_mode == "contrastive" else None
    predictor = CrossModalPredictor(multimodal_dim).to(device) if args.ssl_mode == "multimodal" else None

    params = list(model.parameters())
    if projection_head is not None:
        params += list(projection_head.parameters())
    if predictor is not None:
        params += list(predictor.parameters())

    optimizer = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    wandb_config = {
        "ssl_mode": args.ssl_mode,
        "modalities": modalities,
        "tau": args.tau,
        "projection_dim": args.proj_dim,
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "fold": args.fold,
        "seed": args.seed,
        "pred_direction": args.pred_direction,
        "stop_grad_target": args.stop_grad_target,
        "use_all_unlabeled": args.use_all_unlabeled,
        "augment": "flip+rot90 batch-consistent",
        "step_size": args.step_size,
        "gamma": args.gamma,
        "sanity_checks": args.sanity_checks,
        "sanity_checks_max_batches": args.sanity_checks_max_batches,
    }

    try:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=wandb_config,
            name=run_name,
            job_type="selfsup_pretrain",
            reinit=True,
        )
    except Exception as e:
        print(f"W&B initialization failed: {e}")

    if args.use_all_unlabeled:
        safe_wandb_log({"warning/use_all_unlabeled": 1})

    if args.dry_run:
        _ = run_ssl_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            ssl_mode=args.ssl_mode,
            modalities=modalities,
            tau=args.tau,
            projection_head=projection_head,
            predictor=predictor,
            pred_direction=args.pred_direction,
            stop_grad_target=args.stop_grad_target,
            train=True,
            dry_run=True,
            sanity_checks=args.sanity_checks,
            sanity_checks_max_batches=args.sanity_checks_max_batches,
        )
        print("Dry-run complete. Exiting.")
        if wandb.run is not None:
            wandb.finish()
        return

    best_val_loss = float("inf")
    patience_counter = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_stats = run_ssl_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            ssl_mode=args.ssl_mode,
            modalities=modalities,
            tau=args.tau,
            projection_head=projection_head,
            predictor=predictor,
            pred_direction=args.pred_direction,
            stop_grad_target=args.stop_grad_target,
            train=True,
            sanity_checks=args.sanity_checks,
            sanity_checks_max_batches=args.sanity_checks_max_batches,
        )

        if has_val:
            val_stats = run_ssl_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                ssl_mode=args.ssl_mode,
                modalities=modalities,
                tau=args.tau,
                projection_head=projection_head,
                predictor=predictor,
                pred_direction=args.pred_direction,
                stop_grad_target=args.stop_grad_target,
                train=False,
                sanity_checks=args.sanity_checks,
                sanity_checks_max_batches=args.sanity_checks_max_batches,
            )
            current_val = val_stats["loss"]
        else:
            val_stats = None
            current_val = train_stats["loss"]

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | train_ssl={train_stats['loss']:.6f}"
            + (f" | val_ssl={current_val:.6f}" if has_val else "")
            + f" | lr={get_lr(optimizer):.2e}"
        )

        log_payload = {
            "epoch": epoch,
            "lr": get_lr(optimizer),
            "tau": args.tau,
            "ssl/embedding_norm": train_stats["emb_norm"],
            "ssl/collapse_var_mean": train_stats["collapse_var"],
            "aug/flip_d_count": train_stats["aug_flip_d_count"],
            "aug/flip_h_count": train_stats["aug_flip_h_count"],
            "aug/flip_w_count": train_stats["aug_flip_w_count"],
            "aug/rot_count": train_stats["aug_rot_count"],
        }
        if args.ssl_mode == "contrastive":
            log_payload["ssl/contrastive_loss"] = train_stats["loss"]
        else:
            log_payload.update(
                {
                    "ssl/multimodal_loss": train_stats["loss"],
                    "ssl/alignment_cosine": train_stats["align_cos"],
                    "ssl/reg_error_mean": train_stats["reg_err_mean"],
                    "ssl/reg_error_std": train_stats["reg_err_std"],
                    "ssl/reg_error_p50": train_stats["reg_err_p50"],
                    "ssl/reg_error_p90": train_stats["reg_err_p90"],
                }
            )
        if val_stats is not None:
            if args.ssl_mode == "contrastive":
                log_payload["val/contrastive_loss"] = val_stats["loss"]
            else:
                log_payload["val/multimodal_loss"] = val_stats["loss"]
                log_payload["val/alignment_cosine"] = val_stats["align_cos"]
        safe_wandb_log(log_payload)

        save_checkpoint(
            path=os.path.join(output_dir, "checkpoint_last.pt"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val=best_val_loss,
            args=args,
            projection_head=projection_head,
            predictor=predictor,
        )

        if current_val < best_val_loss:
            best_val_loss = current_val
            patience_counter = 0
            save_checkpoint(
                path=os.path.join(output_dir, "checkpoint_best.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val=best_val_loss,
                args=args,
                projection_head=projection_head,
                predictor=predictor,
            )
            torch.save(model.state_dict(), os.path.join(output_dir, f"{run_name}.pt"))
        else:
            patience_counter += 1
            if has_val and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    elapsed = time.time() - start
    print(f"Pretraining complete in {elapsed/60.0:.2f} minutes")
    print(f"Outputs saved in: {output_dir}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    if args.cv5:
        print("Running 5-fold CV pretraining sequentially (folds 0..4)")
        for fold_id in [0, 1, 2, 3, 4]:
            print("\n" + "=" * 80)
            print(f"Starting fold {fold_id}")
            print("=" * 80)
            fold_args = argparse.Namespace(**vars(args))
            fold_args.fold = fold_id
            main(fold_args)
    else:
        main(args)