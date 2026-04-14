import os
import sys
import re
import json
import glob
import argparse
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)
from scipy.stats import wilcoxon, norm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_generator import get_patch_dataloaders
from training_evaluation_functions import evaluate_classifier
from ema import ModelEMA
from train_resnet18_baseline import ResNet18Baseline

import architecture
import architecture2
import architecture3
import architecture4

try:
    from radiomics import featureextractor
except ImportError:
    featureextractor = None


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def canonical_uid(uid_or_meta=None, *, subject=None, patch_id=None):
    if isinstance(uid_or_meta, dict):
        if "lesion_uid" in uid_or_meta:
            uid = str(uid_or_meta["lesion_uid"]).strip()
            if uid:
                return uid
            raise ValueError("Empty lesion_uid found in metadata dict.")

        subject_key = next((k for k in ["subject", "subject_id", "subj"] if k in uid_or_meta), None)
        patch_key = next((k for k in ["patch_id", "lesion_id"] if k in uid_or_meta), None)
        if subject_key is not None and patch_key is not None:
            subj_val = str(uid_or_meta[subject_key]).strip()
            patch_val = str(uid_or_meta[patch_key]).strip()
            if subj_val and patch_val:
                return f"{subj_val}_{patch_val}"
            raise ValueError(
                f"Invalid subject/patch values in metadata dict: {subject_key}='{uid_or_meta[subject_key]}', {patch_key}='{uid_or_meta[patch_key]}'"
            )

        raise ValueError(
            f"Could not reconstruct canonical UID from dict metadata. Available keys: {sorted(uid_or_meta.keys())}"
        )

    if isinstance(uid_or_meta, str):
        uid = uid_or_meta.strip()
        if uid:
            return uid
        raise ValueError("Received empty string UID.")

    if subject is not None and patch_id is not None:
        subj_val = str(subject).strip()
        patch_val = str(patch_id).strip()
        if subj_val and patch_val:
            return f"{subj_val}_{patch_val}"
        raise ValueError(f"Invalid subject/patch for canonical UID: subject='{subject}', patch_id='{patch_id}'")

    meta_type = type(uid_or_meta).__name__
    raise ValueError(
        f"Could not reconstruct canonical UID. uid_or_meta type={meta_type}, subject={subject}, patch_id={patch_id}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Final 5-fold cross-validation evaluation")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA checkpoints when available")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--num_bootstrap", type=int, default=500, help="Number of subject-level bootstrap replicates")
    parser.add_argument("--num_bootstrap_auc", type=int, default=1000, help="Number of subject-level ROC-AUC bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--export_csv", type=str2bool, nargs="?", const=True, default=True,
                        help="Export canonical CSV summaries (default: True)")
    parser.add_argument("--export_dir", type=str, default="exports",
                        help="Subdirectory for CSV exports under FINAL_AGG")
    return parser.parse_args()


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(message, logfile=None):
    line = f"[{ts()}] {message}"
    print(line)
    if logfile is not None:
        with open(logfile, "a") as f:
            f.write(line + "\n")


def read_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def get_outer_folds():
    return {
        0: ['006','009','011','026','039','042','056','063','076','081','093','112','116','122','126','156','162'],
        1: ['008','010','021','027','029','071','073','086','092','101','109','117','127','136','149','176','178'],
        2: ['004','018','024','046','049','051','065','067','074','078','079','097','099','103','115','119','134','183'],
        3: ['007','025','028','034','035','037','045','068','082','083','088','107','108','110','118','172','180','185'],
        4: ['002','003','015','017','032','043','052','055','080','084','090','096','098','131','135','142','143','182']
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


def split_dev_subjects(dev_subjects, labels_dict, subject_phase_map, seed=42, val_ratio=0.2, max_attempts=50):
    dev_subjects = list(dev_subjects)
    for attempt in range(max_attempts):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + attempt)
        train_idx, val_idx = next(gss.split(dev_subjects, groups=dev_subjects))
        train_subjects = [dev_subjects[i] for i in train_idx]
        val_subjects = [dev_subjects[i] for i in val_idx]
        if any(subject_has_positive(s, labels_dict, subject_phase_map) for s in val_subjects):
            return train_subjects, val_subjects
    raise ValueError("Unable to create a validation split with at least one positive subject.")


def discover_experiment_root(experiment):
    candidates = [
        os.path.join("/data/cil/veronica/Experiments/SM_experiments/3D_Rim_Classification", experiment),
        os.path.join("/data/cil/veronica/Experiments/SM_experiments/QSMrimnet_exp", experiment),
        os.path.join("/nethome/vpignedoli/3DRim_classifier/experiments", experiment),
        os.path.join("/nethome/vpignedoli/QSMRim-Net/experiments", experiment),
        os.path.join(os.getcwd(), "experiments", experiment),
        os.path.join(os.getcwd(), experiment),
    ]

    for path in candidates:
        if os.path.isdir(path):
            return path

    search_roots = [
        "/data/cil/veronica/Experiments/SM_experiments/3D_Rim_Classification",
        "/data/cil/veronica/Experiments/SM_experiments/QSMrimnet_exp",
        "/data/cil/veronica/Experiments/SM_experiments/3D_Rim_Classification/experiments",
        "/nethome/vpignedoli/3DRim_classifier",
        "/nethome/vpignedoli/QSMRim-Net",
        os.getcwd(),
    ]

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for cur_root, dirs, _ in os.walk(root):
            if os.path.basename(cur_root) != experiment:
                continue
            fold_like = [d for d in dirs if re.match(r"(?i)^(fold|cv)", d)]
            if fold_like:
                return cur_root

    raise FileNotFoundError(
        f"Could not locate experiment root for '{experiment}'. Checked: {candidates} and recursive search roots {search_roots}"
    )


def fold_sort_key(path):
    name = os.path.basename(path)
    m = re.search(r"(\d+)", name)
    if m:
        return (0, int(m.group(1)), name.lower())
    return (1, 10**9, name.lower())


def discover_fold_dirs(experiment_root):
    patterns = ["fold*", "Fold*", "cv*", "CV*"]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(os.path.join(experiment_root, pattern)))

    fold_dirs = sorted({os.path.abspath(p) for p in found if os.path.isdir(p)}, key=fold_sort_key)
    if len(fold_dirs) != 5:
        raise ValueError(
            f"Expected exactly 5 fold folders under {experiment_root}, found {len(fold_dirs)}: {fold_dirs}"
        )
    return fold_dirs


def recover_runtime_config(experiment_root, fold_dirs):
    cfg = {
        "patches_path": "/data/cil/veronica/3DRim_Classification_patches",
        "batch_size": 32,
        "num_workers": 8,
        "modalities": ["FLAIR", "QSM"],
        "balance_classes": False,
        "weighted_sampler": False,
        "filters": 16,
        "dropout": 0.3,
        "seed": 42,
        "inner_val_ratio": 0.2,
        "no_radiomics": False,
        "model": "resnet18",
    }

    candidate_cfg_files = []
    for base in [experiment_root] + list(fold_dirs):
        candidate_cfg_files.extend([
            os.path.join(base, "config.json"),
            os.path.join(base, "args.json"),
            os.path.join(base, "training_config.json"),
            os.path.join(base, "train_config.json"),
            os.path.join(base, "run_config.json"),
        ])

    for path in candidate_cfg_files:
        data = read_json_if_exists(path)
        if not isinstance(data, dict):
            continue
        if "patches_path" in data:
            cfg["patches_path"] = data["patches_path"]
        if "batch_size" in data:
            cfg["batch_size"] = int(data["batch_size"])
        if "num_workers" in data:
            cfg["num_workers"] = int(data["num_workers"])
        if "modalities" in data:
            if isinstance(data["modalities"], str):
                cfg["modalities"] = [m.strip() for m in data["modalities"].split(",")]
            else:
                cfg["modalities"] = list(data["modalities"])
        if "balance_classes" in data:
            cfg["balance_classes"] = bool(data["balance_classes"])
        if "weighted_sampler" in data:
            cfg["weighted_sampler"] = bool(data["weighted_sampler"])
        if "filters" in data:
            cfg["filters"] = int(data["filters"])
        if "dropout" in data:
            cfg["dropout"] = float(data["dropout"])
        if "seed" in data:
            cfg["seed"] = int(data["seed"])
        if "inner_val_ratio" in data:
            cfg["inner_val_ratio"] = float(data["inner_val_ratio"])
        if "no_radiomics" in data:
            cfg["no_radiomics"] = bool(data["no_radiomics"])
        if "model" in data:
            cfg["model"] = str(data["model"])

    return cfg


class QSMPatchDataset3D(Dataset):
    def __init__(self, patches_path, labels_dict, subjects, modalities, subject_phase_map):
        self.patches_path = patches_path
        self.labels_dict = labels_dict
        self.subjects = subjects
        self.modalities = modalities
        self.subject_phase_map = subject_phase_map
        self.samples = []

        for subj in subjects:
            subj_phase = self.subject_phase_map.get(subj)
            if subj_phase is None:
                continue
            if subj_phase not in labels_dict or subj not in labels_dict[subj_phase]:
                continue

            for patch_id, label in labels_dict[subj_phase][subj].items():
                valid = True
                for mod in modalities:
                    patch_path = os.path.join(patches_path, subj_phase, subj, mod, f"{patch_id}.npy")
                    if not os.path.exists(patch_path):
                        valid = False
                        break
                if valid:
                    self.samples.append({
                        "subject": subj,
                        "patch_id": patch_id,
                        "label": label,
                        "phase": subj_phase,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        subj = sample["subject"]
        patch_id = sample["patch_id"]
        label = sample["label"]
        phase = sample["phase"]

        patches = []
        for mod in self.modalities:
            patch_path = os.path.join(self.patches_path, phase, subj, mod, f"{patch_id}.npy")
            patches.append(np.load(patch_path))

        patches = np.stack(patches, axis=0).astype(np.float32)
        return {
            "patches": torch.from_numpy(patches),
            "label": torch.tensor([label], dtype=torch.float32),
            "subject": subj,
            "patch_id": patch_id,
        }


def extract_radiomics_features(qsm_patch):
    if featureextractor is None:
        return np.zeros(527, dtype=np.float32)

    try:
        import SimpleITK as sitk

        qsm_img = sitk.GetImageFromArray(qsm_patch)
        mask_patch = (qsm_patch > qsm_patch.mean()).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask_patch)

        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        features = extractor.execute(qsm_img, mask_img)

        feature_values = []
        for key, value in features.items():
            if not key.startswith("diagnostics"):
                try:
                    feature_values.append(float(value))
                except (ValueError, TypeError):
                    pass

        feature_vector = np.array(feature_values, dtype=np.float32)
        if len(feature_vector) < 527:
            feature_vector = np.pad(feature_vector, (0, 527 - len(feature_vector)), "constant")
        else:
            feature_vector = feature_vector[:527]
        return feature_vector
    except Exception:
        return np.zeros(527, dtype=np.float32)


def evaluate_qsmrimnet(model, test_loader, device, output_dir, threshold=0.5, use_radiomics=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_uids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            patches = batch["patches"].to(device)
            labels = batch["label"].to(device)
            batch_size = patches.shape[0]

            if use_radiomics:
                radiomics_features = []
                for i in range(batch_size):
                    qsm_patch = patches[i, 1, ...].cpu().numpy()
                    radiomics_features.append(extract_radiomics_features(qsm_patch))
                radiomics_features = torch.from_numpy(np.stack(radiomics_features)).to(device)
            else:
                radiomics_features = torch.zeros(batch_size, 527, device=device)

            dummy_channel = torch.zeros_like(patches[:, 0:1, ...])
            x_input = torch.cat([dummy_channel, patches], dim=1)
            outputs, _ = model(x_input, radiomics_features, labels)

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_uids.extend([
                canonical_uid(subject=batch['subject'][i], patch_id=batch['patch_id'][i]) for i in range(batch_size)
            ])

    all_preds = np.array(all_preds).flatten().astype(int)
    all_labels = np.array(all_labels).flatten().astype(int)
    all_probs = np.array(all_probs).flatten().astype(float)

    try:
        auc_val = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_val = 0.0

    try:
        avg_prec = average_precision_score(all_labels, all_probs)
    except ValueError:
        avg_prec = 0.0

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": auc_val,
        "average_precision": avg_prec,
    }

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({
        "lesion_uid": all_uids,
        "patch_prob": all_probs,
        "label": all_labels,
    }).to_csv(os.path.join(output_dir, "test_results.csv"), index=False)

    return metrics, {
        "y_true": all_labels.tolist(),
        "y_prob": all_probs.tolist(),
        "lesion_uids": all_uids,
    }


def load_labels_dict(patches_path):
    labels_path = os.path.join(patches_path, "labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    with open(labels_path, "r") as f:
        return json.load(f)


def parse_split_file(fold_dir):
    candidates = [
        "split_subjects.json",
        "subjects.json",
        "splits.json",
        "split.json",
    ]

    for name in candidates:
        path = os.path.join(fold_dir, name)
        data = read_json_if_exists(path)
        if isinstance(data, dict) and "test" in data:
            data.setdefault("train", [])
            data.setdefault("val", [])
            return {
                "train": list(data["train"]),
                "val": list(data["val"]),
                "test": list(data["test"]),
            }, path

    txt_candidates = ["test_subjects.txt", "subjects_test.txt"]
    for name in txt_candidates:
        path = os.path.join(fold_dir, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                test_subjects = [ln.strip() for ln in f if ln.strip()]
            if len(test_subjects) > 0:
                return {"train": [], "val": [], "test": test_subjects}, path

    return None, None


def infer_fold_index(fold_dir, fallback_idx):
    m = re.search(r"(\d+)", os.path.basename(fold_dir))
    if m:
        return int(m.group(1))
    return fallback_idx


def fallback_split_for_fold(fold_index, labels_dict, seed=42, inner_val_ratio=0.2):
    folds = get_outer_folds()
    if fold_index not in folds:
        raise ValueError(f"Fold index {fold_index} not in predefined outer folds.")

    test_subjects = list(folds[fold_index])
    dev_subjects = []
    for k, subs in folds.items():
        if k != fold_index:
            dev_subjects.extend(subs)

    subject_phase_map = get_subject_phase_map(labels_dict)
    train_subjects, val_subjects = split_dev_subjects(
        dev_subjects=dev_subjects,
        labels_dict=labels_dict,
        subject_phase_map=subject_phase_map,
        seed=seed,
        val_ratio=inner_val_ratio,
    )

    return {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }


def load_checkpoint_state(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def build_model_for_state(state_dict, cfg, device):
    modalities = cfg["modalities"]
    in_channels_list = [1] * len(modalities)
    base_filters = int(cfg["filters"])
    dropout = float(cfg["dropout"])

    attempts = [
        ("architecture", architecture.MultiModalClassifier),
        ("architecture4", architecture4.MultiModalClassifier),
        ("architecture2", architecture2.MultiModalClassifier),
        ("architecture3", architecture3.MultiModalClassifier),
    ]

    errors = []
    for module_name, cls in attempts:
        try:
            model = cls(
                in_channels_list=in_channels_list,
                base_filters=base_filters,
                dropout=dropout,
                num_classes=1,
            ).to(device)
            model.load_state_dict(state_dict, strict=True)
            return model, module_name
        except Exception as e:
            errors.append(f"{module_name}: {e}")

    baseline_model_names = [cfg.get("model", "resnet18"), "resnet18", "r3d_18"]
    baseline_model_names = list(dict.fromkeys(baseline_model_names))
    for model_name in baseline_model_names:
        try:
            model = ResNet18Baseline(in_channels=len(modalities), model_name=model_name).to(device)
            model.load_state_dict(state_dict, strict=True)
            return model, f"resnet18_baseline:{model_name}"
        except Exception as e:
            errors.append(f"resnet18_baseline:{model_name}: {e}")

    try:
        qsm_src_dir = os.path.join("/nethome/vpignedoli/QSMRim-Net", "src")
        qsm_model_path = os.path.join(qsm_src_dir, "QSMRim-Net.py")
        path_added = False
        if qsm_src_dir not in sys.path:
            sys.path.insert(0, qsm_src_dir)
            path_added = True

        spec = importlib.util.spec_from_file_location("qsmrimnet", qsm_model_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to create import spec for {qsm_model_path}")
        qsmrimnet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qsmrimnet)
        qsm_model = qsmrimnet.res18L4TwoRadsPlusSMOTENew(in_channels=2).to(device)
        qsm_model.load_state_dict(state_dict, strict=True)
        if path_added and len(sys.path) > 0 and sys.path[0] == qsm_src_dir:
            sys.path.pop(0)
        return qsm_model, "qsmrimnet"
    except Exception as e:
        errors.append(f"qsmrimnet: {e}")

    raise RuntimeError(
        "Could not instantiate/load model state dict with architecture.py/architecture2.py/architecture3.py/architecture4.py, ResNet18Baseline, or QSMRim-Net. Errors: "
        + " | ".join(errors)
    )


def step_interp(x, y, grid):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]
    if len(unique_x) == 1:
        return np.full_like(grid, fill_value=float(unique_y[0]), dtype=float)
    idx = np.searchsorted(unique_x, grid, side="right") - 1
    idx = np.clip(idx, 0, len(unique_x) - 1)
    return unique_y[idx]


def truncate_roc_to_max_fpr(fpr, tpr, max_fpr=0.1):
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    mask = fpr <= max_fpr
    fpr_t = fpr[mask]
    tpr_t = tpr[mask]
    if len(fpr_t) == 0:
        return np.array([0.0, max_fpr]), np.array([0.0, 0.0])
    if fpr_t[-1] < max_fpr:
        tpr_at_max = np.interp(max_fpr, fpr, tpr)
        fpr_t = np.concatenate([fpr_t, [max_fpr]])
        tpr_t = np.concatenate([tpr_t, [tpr_at_max]])
    return fpr_t, tpr_t


def best_f1_threshold(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.unique(y_prob)
    if len(thresholds) == 0:
        return 0.5
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def confusion_and_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "specificity": float(specificity),
        "recall": float(recall),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def safe_auc_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = np.nan

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    fpr_proc, tpr_proc = truncate_roc_to_max_fpr(fpr, tpr, max_fpr=0.1)
    proc_auc = float(auc(fpr_proc, tpr_proc))

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall_curve, precision_curve))

    return {
        "roc_auc": roc_auc,
        "proc_auc": proc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "fpr_proc": fpr_proc,
        "tpr_proc": tpr_proc,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "roc_thresholds": roc_thresholds,
        "pr_thresholds": pr_thresholds,
    }


def parse_branch_name_for_export(branch_name):
    if "/" not in branch_name:
        return None, None
    threshold_name, variant_name = branch_name.split("/", 1)
    if variant_name == "legacy_selected":
        return threshold_name, None

    mapping = {
        "pre_retrain/best": "best",
        "pre_retrain/best_ema": "best_ema",
        "post_retrain": "post_retrain",
        "post_retrain_ema": "post_retrain_ema",
    }
    return threshold_name, mapping.get(variant_name)


def extract_subject_id(lesion_uid):
    uid = str(lesion_uid)
    if "_" in uid:
        return uid.split("_", 1)[0]
    m = re.match(r"([A-Za-z]*\d+)", uid)
    if m:
        return m.group(1)
    return uid


def save_curve_plot_with_folds(x_grid, fold_curves, mean_curve, title, xlabel, ylabel, out_path, diagonal=False, annotation_text=None):
    plt.figure(figsize=(8, 6))
    for i, y in enumerate(fold_curves):
        plt.plot(x_grid, y, linewidth=1, alpha=0.5, label=f"fold_{i}")
    plt.plot(x_grid, mean_curve, linewidth=3, color="black", label="mean")
    if diagonal:
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if annotation_text is not None:
        plt.text(
            0.98,
            0.02,
            annotation_text,
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="gray"),
        )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_confusion_image(cm, out_path):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NoRim", "Rim"],
        yticklabels=["NoRim", "Rim"],
    )
    plt.title("Aggregated Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_subject_scatter(gold_counts, pred_counts, out_path):
    plt.figure(figsize=(7, 6))
    plt.scatter(gold_counts, pred_counts, alpha=0.8)
    max_val = max(float(np.max(gold_counts)) if len(gold_counts) else 0.0,
                  float(np.max(pred_counts)) if len(pred_counts) else 0.0)
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=1)
    plt.xlabel("Gold lesion count")
    plt.ylabel("Predicted lesion count")
    plt.title("Subject-wise lesion counts")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def select_checkpoint(fold_dir, use_ema):
    ema_path = os.path.join(fold_dir, "model_best_ema.pt")
    base_path = os.path.join(fold_dir, "model_best.pt")
    base_path_pth = os.path.join(fold_dir, "model_best.pth")

    if use_ema:
        if os.path.exists(ema_path):
            return ema_path, True, None
        if os.path.exists(base_path):
            return base_path, False, f"EMA requested but missing in {fold_dir}; falling back to model_best.pt"
        if os.path.exists(base_path_pth):
            return base_path_pth, False, f"EMA requested but missing in {fold_dir}; falling back to model_best.pth"
        raise FileNotFoundError(f"Missing checkpoints in {fold_dir}: expected model_best_ema.pt, model_best.pt, or model_best.pth")

    if not os.path.exists(base_path):
        if not os.path.exists(base_path_pth):
            raise FileNotFoundError(f"Missing checkpoint: {base_path} (or {base_path_pth})")
        return base_path_pth, False, None
    return base_path, False, None


def infer_probs(model, loader, device, fold_id=None, model_tag=None):
    model.eval()
    all_labels = []
    all_probs = []
    all_uids = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Infer", leave=False)):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                volumes, labels, lesion_ids = batch
                volumes = [vol.to(device) for vol in volumes]
                logits = model(volumes).squeeze(1)
                probs = torch.sigmoid(logits)

                labels_np = labels.detach().to(torch.int64).cpu().numpy().reshape(-1)
                probs_np = probs.detach().float().cpu().numpy().reshape(-1)
                batch_size = labels_np.shape[0]

                if torch.is_tensor(lesion_ids):
                    lesion_items = lesion_ids.detach().cpu().numpy().reshape(-1).tolist()
                elif isinstance(lesion_ids, (list, tuple, np.ndarray)):
                    lesion_items = list(lesion_ids)
                else:
                    lesion_items = []

                try:
                    uid_list = [canonical_uid(item) for item in lesion_items]
                except Exception as e:
                    raise RuntimeError(
                        f"Canonical UID extraction failed in infer_probs (fold_id={fold_id}, model_tag={model_tag}, batch_idx={batch_idx}). "
                        f"This would break cross-experiment pairing. Details: {e}"
                    ) from e

                if len(uid_list) != batch_size:
                    raise RuntimeError(
                        f"UID extraction mismatch in infer_probs (fold_id={fold_id}, model_tag={model_tag}, batch_idx={batch_idx}): "
                        f"expected {batch_size} UIDs, got {len(uid_list)}. Synthetic UID fallback is disabled to preserve cross-experiment pairing."
                    )

                all_labels.extend(labels_np.tolist())
                all_probs.extend(probs_np.tolist())
                all_uids.extend(uid_list)
            elif isinstance(batch, dict):
                patches = batch["patches"].to(device)
                labels = batch["label"].to(device)
                batch_size = patches.shape[0]

                if featureextractor is not None:
                    radiomics_features = []
                    for i in range(batch_size):
                        qsm_patch = patches[i, 1, ...].detach().cpu().numpy()
                        radiomics_features.append(extract_radiomics_features(qsm_patch))
                    radiomics_features = torch.from_numpy(np.stack(radiomics_features)).to(device)
                else:
                    radiomics_features = torch.zeros(batch_size, 527, device=device)

                dummy_channel = torch.zeros_like(patches[:, 0:1, ...])
                x_input = torch.cat([dummy_channel, patches], dim=1)
                outputs, _ = model(x_input, radiomics_features, labels)
                probs = torch.sigmoid(outputs).view(-1)

                labels_np = labels.detach().to(torch.int64).view(-1).cpu().numpy().reshape(-1)
                probs_np = probs.detach().float().cpu().numpy().reshape(-1)
                batch_size = labels_np.shape[0]
                subjects = list(batch.get("subject", [""] * batch_size))
                patch_ids = list(batch.get("patch_id", [""] * batch_size))
                try:
                    uid_list = [canonical_uid(subject=subjects[i], patch_id=patch_ids[i]) for i in range(batch_size)]
                except Exception as e:
                    raise RuntimeError(
                        f"Canonical UID reconstruction failed in infer_probs dict-batch (fold_id={fold_id}, model_tag={model_tag}, batch_idx={batch_idx}). "
                        f"This would break cross-experiment pairing. Details: {e}"
                    ) from e

                all_labels.extend(labels_np.tolist())
                all_probs.extend(probs_np.tolist())
                all_uids.extend(uid_list)
            else:
                raise TypeError(f"Unsupported batch type for infer_probs: {type(batch)}")

    y_true = np.asarray(all_labels, dtype=int).reshape(-1)
    y_prob = np.asarray(all_probs, dtype=float).reshape(-1)
    y_uid = np.asarray(all_uids, dtype=object).reshape(-1)
    return y_true, y_prob, y_uid


def choose_best_threshold_f1(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5, f1_score(y_true, (y_prob > 0.5).astype(int), zero_division=0)

    thresholds = np.linspace(0.0, 1.0, 1001)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def metrics_from_probs(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    y_pred = (y_prob > threshold).astype(int)
    cm_metrics = confusion_and_metrics(y_true, y_pred)
    curve_metrics = safe_auc_metrics(y_true, y_prob)
    metrics_payload = {
        **cm_metrics,
        "roc_auc": float(curve_metrics["roc_auc"]),
        "proc_auc": float(curve_metrics["proc_auc"]),
        "pr_auc": float(curve_metrics["pr_auc"]),
    }
    return metrics_payload, curve_metrics, y_pred


def subject_level_bootstrap(labels, probs, fold_ids, subject_ids, fold_thresholds, num_bootstrap=500, seed=0):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    fold_ids = np.asarray(fold_ids, dtype=int)
    subject_ids = np.asarray(subject_ids)

    unique_subjects = np.unique(subject_ids)
    rng = np.random.default_rng(seed)

    out = {
        "accuracy": [], "specificity": [], "recall": [], "precision": [], "f1": [],
        "roc_auc": [], "proc_auc": [], "pr_auc": [],
    }

    for _ in range(num_bootstrap):
        sampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        sampled_indices = []
        for sid in sampled_subjects:
            sampled_indices.extend(np.where(subject_ids == sid)[0].tolist())
        sampled_indices = np.array(sampled_indices, dtype=int)

        if sampled_indices.size == 0:
            continue

        y_b = labels[sampled_indices]
        p_b = probs[sampled_indices]
        f_b = fold_ids[sampled_indices]

        pred_b = np.zeros_like(y_b)
        for fold_value, threshold in fold_thresholds.items():
            mask = (f_b == fold_value)
            pred_b[mask] = (p_b[mask] >= threshold).astype(int)

        cm_metrics = confusion_and_metrics(y_b, pred_b)
        auc_metrics = safe_auc_metrics(y_b, p_b)

        out["accuracy"].append(cm_metrics["accuracy"])
        out["specificity"].append(cm_metrics["specificity"])
        out["recall"].append(cm_metrics["recall"])
        out["precision"].append(cm_metrics["precision"])
        out["f1"].append(cm_metrics["f1"])
        out["roc_auc"].append(auc_metrics["roc_auc"])
        out["proc_auc"].append(auc_metrics["proc_auc"])
        out["pr_auc"].append(auc_metrics["pr_auc"])

    for key in out:
        out[key] = np.array(out[key], dtype=float)

    return out


def bootstrap_summary(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


def variant_to_model_tag(variant_name):
    mapping = {
        "pre_retrain/best": "best",
        "pre_retrain/best_ema": "best_ema",
        "post_retrain": "post_retrain",
        "post_retrain_ema": "post_retrain_ema",
    }
    return mapping.get(variant_name)


def compute_midrank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    sorted_x = x[order]
    n = sorted_x.shape[0]
    midranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j

    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    predictions_sorted_transposed = np.asarray(predictions_sorted_transposed, dtype=float)
    m = int(label_1_count)
    n = int(predictions_sorted_transposed.shape[1] - m)
    if m <= 0 or n <= 0:
        raise ValueError("Both positive and negative samples are required for DeLong.")

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true, y_prob_a, y_prob_b):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob_a = np.asarray(y_prob_a, dtype=float).reshape(-1)
    y_prob_b = np.asarray(y_prob_b, dtype=float).reshape(-1)

    if y_true.size == 0:
        return np.nan, np.nan, np.nan, 1.0
    if y_true.size != y_prob_a.size or y_true.size != y_prob_b.size:
        raise ValueError("DeLong inputs must have the same length.")
    if np.unique(y_true).size < 2:
        return np.nan, np.nan, np.nan, 1.0

    order = np.argsort(-y_true)
    predictions = np.vstack([y_prob_a, y_prob_b])[:, order]
    label_1_count = int(np.sum(y_true))

    aucs, delong_cov = fast_delong(predictions, label_1_count)
    auc_a = float(aucs[0])
    auc_b = float(aucs[1])
    delta = float(auc_a - auc_b)

    var_delta = float(delong_cov[0, 0] + delong_cov[1, 1] - 2.0 * delong_cov[0, 1])
    if (not np.isfinite(var_delta)) or var_delta <= 0.0:
        p_value = 1.0
    else:
        z = abs(delta) / np.sqrt(var_delta)
        p_value = float(2.0 * (1.0 - norm.cdf(z)))

    return auc_a, auc_b, delta, p_value


def align_model_pair_predictions(raw_pred_df, tag_a, tag_b, fold_id=None):
    subset = raw_pred_df
    if fold_id is not None:
        subset = subset[subset["fold_id"] == int(fold_id)]

    a_df = subset[subset["model_tag"] == tag_a][["fold_id", "lesion_uid", "y_true", "y_prob"]].copy()
    b_df = subset[subset["model_tag"] == tag_b][["fold_id", "lesion_uid", "y_true", "y_prob"]].copy()

    merged = pd.merge(
        a_df,
        b_df,
        on=["fold_id", "lesion_uid"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        return merged, np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    merged = merged[merged["y_true_a"] == merged["y_true_b"]].copy()
    if merged.empty:
        return merged, np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    y_true = merged["y_true_a"].to_numpy(dtype=int)
    y_prob_a = merged["y_prob_a"].to_numpy(dtype=float)
    y_prob_b = merged["y_prob_b"].to_numpy(dtype=float)
    return merged, y_true, y_prob_a, y_prob_b


def run_wilcoxon_fold_auc(vec_a, vec_b):
    arr_a = np.asarray(vec_a, dtype=float)
    arr_b = np.asarray(vec_b, dtype=float)
    valid_mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a = arr_a[valid_mask]
    arr_b = arr_b[valid_mask]

    out = {
        "statistic": 0.0,
        "p_value": 1.0,
        "n_folds": int(arr_a.size),
        "note": "",
    }

    if arr_a.size < 2:
        out["note"] = "insufficient_folds"
        return out

    diff = arr_a - arr_b
    if np.allclose(diff, 0.0):
        out["note"] = "all_zero_differences"
        return out

    try:
        stat, p_val = wilcoxon(arr_a, arr_b, alternative="two-sided", zero_method="wilcox")
        out["statistic"] = float(stat)
        out["p_value"] = float(p_val)
    except ValueError:
        out["note"] = "wilcoxon_value_error"

    return out


def bootstrap_subject_auc_distribution(y_true, y_prob, subject_ids, n_boot=1000, seed=0):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    subject_ids = np.asarray(subject_ids).reshape(-1)

    if y_true.size == 0 or y_prob.size == 0 or subject_ids.size == 0:
        return np.array([], dtype=float)
    if np.unique(y_true).size < 2:
        return np.array([], dtype=float)

    unique_subjects = np.unique(subject_ids)
    idx_by_subject = {sid: np.where(subject_ids == sid)[0] for sid in unique_subjects}
    rng = np.random.default_rng(seed)

    out = []
    attempts = 0
    max_attempts = max(int(n_boot) * 100, int(n_boot) + 10)

    while len(out) < int(n_boot) and attempts < max_attempts:
        attempts += 1
        # Multiplicity bootstrap: when a subject is sampled multiple times,
        # all its lesion rows are appended multiple times in sampled_indices.
        sampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        sampled_indices = []
        for sid in sampled_subjects:
            sampled_indices.extend(idx_by_subject[sid].tolist())

        if len(sampled_indices) == 0:
            continue

        y_b = y_true[sampled_indices]
        p_b = y_prob[sampled_indices]
        if np.unique(y_b).size < 2:
            continue

        try:
            auc_val = float(roc_auc_score(y_b, p_b))
        except ValueError:
            continue
        if np.isfinite(auc_val):
            out.append(auc_val)

    return np.asarray(out, dtype=float)


def summarize_auc_bootstrap(boot_aucs):
    boot_aucs = np.asarray(boot_aucs, dtype=float)
    boot_aucs = boot_aucs[np.isfinite(boot_aucs)]
    if boot_aucs.size == 0:
        return {
            "ci_low": np.nan,
            "ci_high": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "n_boot_valid": 0,
        }
    return {
        "ci_low": float(np.percentile(boot_aucs, 2.5)),
        "ci_high": float(np.percentile(boot_aucs, 97.5)),
        "median": float(np.median(boot_aucs)),
        "mean": float(np.mean(boot_aucs)),
        "n_boot_valid": int(boot_aucs.size),
    }


def aggregate_branch_entries(entries):
    entries = sorted(entries, key=lambda x: x["fold"])
    if len(entries) == 0:
        return None

    labels_all = np.concatenate([e["y_true"] for e in entries]).astype(int)
    probs_all = np.concatenate([e["y_prob"] for e in entries]).astype(float)
    preds_all = np.concatenate([e["y_pred"] for e in entries]).astype(int)

    overall_cm = confusion_and_metrics(labels_all, preds_all)
    overall_curves = safe_auc_metrics(labels_all, probs_all)

    fpr_grid = np.linspace(0.0, 1.0, 200)
    fpr_proc_grid = np.linspace(0.0, 0.1, 200)
    recall_grid = np.linspace(0.0, 1.0, 200)

    fold_roc_interp = np.vstack([step_interp(e["curves"]["fpr"], e["curves"]["tpr"], fpr_grid) for e in entries])
    fold_proc_interp = np.vstack([step_interp(e["curves"]["fpr_proc"], e["curves"]["tpr_proc"], fpr_proc_grid) for e in entries])
    fold_pr_interp = np.vstack([step_interp(e["curves"]["recall_curve"], e["curves"]["precision_curve"], recall_grid) for e in entries])

    mean_roc = np.mean(fold_roc_interp, axis=0)
    mean_proc = np.mean(fold_proc_interp, axis=0)
    mean_pr = np.mean(fold_pr_interp, axis=0)

    metric_names = ["accuracy", "specificity", "recall", "precision", "f1", "roc_auc", "proc_auc", "pr_auc"]
    cv_mean_std = {}
    for name in metric_names:
        vals = np.array([e["metrics"][name] for e in entries], dtype=float)
        cv_mean_std[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "n_folds": len(entries),
        "thresholds_per_fold": {str(e["fold"]): float(e["threshold"]) for e in entries},
        "validation_f1_per_fold": {
            str(e["fold"]): (None if e["val_best_f1"] is None else float(e["val_best_f1"]))
            for e in entries
        },
        "final": {
            "roc_auc": float(overall_curves["roc_auc"]),
            "proc_auc": float(overall_curves["proc_auc"]),
            "pr_auc": float(overall_curves["pr_auc"]),
            "accuracy": overall_cm["accuracy"],
            "specificity": overall_cm["specificity"],
            "recall": overall_cm["recall"],
            "precision": overall_cm["precision"],
            "f1": overall_cm["f1"],
            "tp": overall_cm["tp"],
            "fp": overall_cm["fp"],
            "fn": overall_cm["fn"],
            "tn": overall_cm["tn"],
        },
        "cv_mean_curve_auc": {
            "roc_auc": float(auc(fpr_grid, mean_roc)),
            "proc_auc": float(auc(fpr_proc_grid, mean_proc)),
            "pr_auc": float(auc(recall_grid, mean_pr)),
        },
        "cv_mean_std": cv_mean_std,
        "folds": [
            {
                "fold": int(e["fold"]),
                "threshold": float(e["threshold"]),
                "val_best_f1": None if e["val_best_f1"] is None else float(e["val_best_f1"]),
                "checkpoint": e["checkpoint"],
                "model_module": e["model_module"],
                **e["metrics"],
            }
            for e in entries
        ],
    }


def pearson_with_fisher_ci(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 4 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan, np.nan, np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(x.size - 3)
    z_low = z - 1.96 * se
    z_high = z + 1.96 * se
    return r, float(np.tanh(z_low)), float(np.tanh(z_high))


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    experiment_root = discover_experiment_root(args.experiment)
    fold_dirs = discover_fold_dirs(experiment_root)

    final_agg_dir = os.path.join(experiment_root, "FINAL_AGG")
    os.makedirs(final_agg_dir, exist_ok=True)
    plots_dir = os.path.join(final_agg_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    export_path = os.path.join(final_agg_dir, args.export_dir)
    if args.export_csv:
        os.makedirs(export_path, exist_ok=True)

    log_path = os.path.join(final_agg_dir, "logs.txt")
    with open(log_path, "w") as f:
        f.write("")

    log_line(f"Experiment root: {experiment_root}", log_path)
    log_line(f"Using device: {device}", log_path)
    log_line(f"Discovered folds: {fold_dirs}", log_path)

    cfg = recover_runtime_config(experiment_root, fold_dirs)
    log_line(
        "Recovered config: "
        f"patches_path={cfg['patches_path']}, batch_size={cfg['batch_size']}, num_workers={cfg['num_workers']}, "
        f"modalities={cfg['modalities']}, balance_classes={cfg['balance_classes']}, weighted_sampler={cfg['weighted_sampler']}, "
        f"filters={cfg['filters']}, dropout={cfg['dropout']}, seed={cfg['seed']}, inner_val_ratio={cfg['inner_val_ratio']}, "
        f"no_radiomics={cfg['no_radiomics']}, model={cfg['model']}",
        log_path,
    )

    labels_dict = load_labels_dict(cfg["patches_path"])
    subject_phase_map = get_subject_phase_map(labels_dict)

    wandb.init(
        project="Final_RIM",
        entity="RIM-project",
        job_type="final_evaluation",
        name=f"{args.experiment}_final_eval",
        config={
            "experiment": args.experiment,
            "use_ema": args.use_ema,
            "threshold": args.threshold,
            "gpu": args.gpu,
            "num_bootstrap": args.num_bootstrap,
            "num_bootstrap_auc": args.num_bootstrap_auc,
            "seed": args.seed,
            "device": str(device),
            **cfg,
        },
    )

    log_line("Canonical UID enforced for exports. Synthetic UID fallback disabled.", log_path)
    if wandb.run is not None:
        wandb.log({
            "uid/canonical_enforced": 1,
            "uid/synthetic_fallback_disabled": 1,
        })

    fixed_threshold = 0.5
    if abs(float(args.threshold) - 0.5) > 1e-12:
        log_line(f"INFO: fixed-threshold branch uses 0.5; ignoring --threshold={args.threshold} for fixed branch.", log_path)

    fold_results = []
    lesion_rows = []
    fold_thresholds = {}

    branch_results = {}
    optimized_thresholds_per_fold = {
        "pre_retrain/best": {},
        "pre_retrain/best_ema": {},
    }

    all_labels_list = []
    all_probs_list = []
    all_fold_ids_list = []
    all_subject_ids_list = []
    all_lesion_uids_list = []
    raw_prediction_rows = []

    def add_branch_result(branch_name, fold_idx, threshold, val_best_f1, split_source, model_module, checkpoint, y_true, y_prob):
        metrics_payload, curve_metrics, y_pred = metrics_from_probs(y_true, y_prob, threshold)
        branch_results.setdefault(branch_name, []).append({
            "fold": int(fold_idx),
            "threshold": float(threshold),
            "val_best_f1": None if val_best_f1 is None else float(val_best_f1),
            "split_source": split_source,
            "model_module": model_module,
            "checkpoint": checkpoint,
            "metrics": metrics_payload,
            "curves": curve_metrics,
            "y_true": np.asarray(y_true, dtype=int).reshape(-1),
            "y_prob": np.asarray(y_prob, dtype=float).reshape(-1),
            "y_pred": np.asarray(y_pred, dtype=int).reshape(-1),
        })
        return metrics_payload, curve_metrics, y_pred

    for idx, fold_dir in enumerate(fold_dirs):
        fold_idx = infer_fold_index(fold_dir, idx)
        log_line(f"Starting fold {fold_idx} at {fold_dir}", log_path)

        split_dict, split_source = parse_split_file(fold_dir)
        if split_dict is None:
            split_dict = fallback_split_for_fold(
                fold_idx,
                labels_dict,
                seed=cfg["seed"],
                inner_val_ratio=cfg["inner_val_ratio"],
            )
            split_source = "fallback:get_outer_folds+split_dev_subjects"

        if "test" not in split_dict or len(split_dict["test"]) == 0:
            raise ValueError(f"No test subjects found for fold folder {fold_dir} (source: {split_source})")

        log_line(f"Fold {fold_idx} split source: {split_source}", log_path)

        ckpt_path, used_ema_file, warning = select_checkpoint(fold_dir, args.use_ema)
        if warning is not None:
            log_line(f"WARNING: {warning}", log_path)
        log_line(f"Fold {fold_idx} checkpoint: {ckpt_path}", log_path)

        pre_best_path, _, _ = select_checkpoint(fold_dir, use_ema=False)
        pre_best_ema_path = os.path.join(fold_dir, "model_best_ema.pt")
        post_retrain_path = os.path.join(fold_dir, "model_post_retrain.pt")
        post_retrain_ema_path = os.path.join(fold_dir, "model_post_retrain_ema.pt")

        variant_ckpts = {
            "pre_retrain/best": pre_best_path,
            "pre_retrain/best_ema": pre_best_ema_path if os.path.exists(pre_best_ema_path) else None,
            "post_retrain": post_retrain_path if os.path.exists(post_retrain_path) else None,
            "post_retrain_ema": post_retrain_ema_path if os.path.exists(post_retrain_ema_path) else None,
        }

        for vname, vpath in variant_ckpts.items():
            if vpath is None:
                log_line(f"INFO: fold {fold_idx} missing optional checkpoint for {vname}; skipping.", log_path)

        test_output_dir = os.path.join(fold_dir, "final_evaluation")
        os.makedirs(test_output_dir, exist_ok=True)

        model_state = load_checkpoint_state(ckpt_path, device)
        model, model_module = build_model_for_state(model_state, cfg, device)
        log_line(f"Fold {fold_idx} model architecture resolved to {model_module}", log_path)

        if model_module == "qsmrimnet":
            qsm_val_dataset = QSMPatchDataset3D(
                patches_path=cfg["patches_path"],
                labels_dict=labels_dict,
                subjects=list(split_dict.get("val", [])),
                modalities=list(cfg["modalities"]),
                subject_phase_map=subject_phase_map,
            )
            qsm_test_dataset = QSMPatchDataset3D(
                patches_path=cfg["patches_path"],
                labels_dict=labels_dict,
                subjects=list(split_dict["test"]),
                modalities=list(cfg["modalities"]),
                subject_phase_map=subject_phase_map,
            )
            val_loader = DataLoader(
                qsm_val_dataset,
                batch_size=int(cfg["batch_size"]),
                shuffle=False,
                num_workers=int(cfg["num_workers"]),
                pin_memory=True,
            )
            test_loader = DataLoader(
                qsm_test_dataset,
                batch_size=int(cfg["batch_size"]),
                shuffle=False,
                num_workers=int(cfg["num_workers"]),
                pin_memory=True,
            )
        else:
            _, val_loader, test_loader = get_patch_dataloaders(
                patches_path=cfg["patches_path"],
                labels_dict=labels_dict,
                subjects={
                    "train": list(split_dict.get("train", [])),
                    "val": list(split_dict.get("val", [])),
                    "test": list(split_dict["test"]),
                },
                modalities=list(cfg["modalities"]),
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
                balance_classes=bool(cfg["balance_classes"]),
                weighted_sampler=bool(cfg["weighted_sampler"]),
                subject_phase_map=subject_phase_map,
            )

        ema_obj = None
        ema_used = False
        if args.use_ema and used_ema_file:
            ema_obj = ModelEMA(model, decay=0.999)
            ema_state = torch.load(ckpt_path, map_location=device)
            if isinstance(ema_state, dict) and "model_state_dict" in ema_state:
                ema_state = ema_state["model_state_dict"]
            ema_obj.ema.load_state_dict(ema_state, strict=True)
            ema_used = True

        if model_module == "qsmrimnet":
            _, details = evaluate_qsmrimnet(
                model=model,
                test_loader=test_loader,
                device=device,
                output_dir=test_output_dir,
                threshold=fixed_threshold,
                use_radiomics=not bool(cfg.get("no_radiomics", False)),
            )
        else:
            _, details = evaluate_classifier(
                model=model,
                test_loader=test_loader,
                device=device,
                output_dir=test_output_dir,
                threshold=fixed_threshold,
                ema=ema_obj,
                return_details=True,
            )

        y_true = np.array(details["y_true"], dtype=int)
        y_prob = np.array(details["y_prob"], dtype=float)
        lesion_uids = np.array(details["lesion_uids"], dtype=object)
        subject_ids = np.array([extract_subject_id(uid) for uid in lesion_uids], dtype=object)

        threshold_fold = fixed_threshold
        fold_thresholds[fold_idx] = threshold_fold
        cm_metrics, curve_metrics, y_pred = metrics_from_probs(y_true, y_prob, threshold_fold)

        metrics_payload = {
            "threshold": float(threshold_fold),
            "ema_used": bool(ema_used),
            "checkpoint": ckpt_path,
            "split_source": split_source,
            "model_module": model_module,
            **cm_metrics,
        }

        with open(os.path.join(test_output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_payload, f, indent=2)

        np.savez(
            os.path.join(test_output_dir, "predictions.npz"),
            all_labels=y_true,
            all_probs=y_prob,
            all_preds=y_pred,
            lesion_uids=lesion_uids,
            subject_ids=subject_ids,
            threshold_used=np.array([threshold_fold] * len(y_true), dtype=float),
        )

        wandb.log({
            f"fold{fold_idx}/threshold": float(threshold_fold),
            f"fold{fold_idx}/roc_auc": float(curve_metrics["roc_auc"]),
            f"fold{fold_idx}/proc_auc": float(curve_metrics["proc_auc"]),
            f"fold{fold_idx}/pr_auc": float(curve_metrics["pr_auc"]),
            f"fold{fold_idx}/accuracy": cm_metrics["accuracy"],
            f"fold{fold_idx}/specificity": cm_metrics["specificity"],
            f"fold{fold_idx}/recall": cm_metrics["recall"],
            f"fold{fold_idx}/precision": cm_metrics["precision"],
            f"fold{fold_idx}/f1": cm_metrics["f1"],
            f"fold{fold_idx}/tp": cm_metrics["tp"],
            f"fold{fold_idx}/fp": cm_metrics["fp"],
            f"fold{fold_idx}/fn": cm_metrics["fn"],
            f"fold{fold_idx}/tn": cm_metrics["tn"],
        })

        log_line(
            f"Fold {fold_idx} fixed_0.5 threshold={threshold_fold:.6f}, "
            f"AUC={curve_metrics['roc_auc']:.4f}, pAUC@0.1={curve_metrics['proc_auc']:.4f}, PR_AUC={curve_metrics['pr_auc']:.4f}, "
            f"ACC={cm_metrics['accuracy']:.4f}, SPEC={cm_metrics['specificity']:.4f}, REC={cm_metrics['recall']:.4f}, PREC={cm_metrics['precision']:.4f}, F1={cm_metrics['f1']:.4f}",
            log_path,
        )

        add_branch_result(
            branch_name="fixed_0.5/legacy_selected",
            fold_idx=fold_idx,
            threshold=threshold_fold,
            val_best_f1=None,
            split_source=split_source,
            model_module=model_module,
            checkpoint=ckpt_path,
            y_true=y_true,
            y_prob=y_prob,
        )

        optimized_thresholds_this_fold = {}
        optimized_val_f1_this_fold = {}

        for source_variant in ["pre_retrain/best", "pre_retrain/best_ema"]:
            source_ckpt = variant_ckpts.get(source_variant)
            if source_ckpt is None:
                continue

            source_state = load_checkpoint_state(source_ckpt, device)
            source_model, source_module = build_model_for_state(source_state, cfg, device)

            if val_loader is None:
                val_y_true = np.array([], dtype=int)
                val_y_prob = np.array([], dtype=float)
            else:
                val_y_true, val_y_prob, _ = infer_probs(
                    source_model,
                    val_loader,
                    device,
                    fold_id=fold_idx,
                    model_tag=f"{source_variant}:val",
                )

            if val_y_true.size == 0:
                log_line(
                    f"WARNING: fold {fold_idx} {source_variant} has empty validation set; using t*=0.5.",
                    log_path,
                )
            elif np.unique(val_y_true).size < 2:
                log_line(
                    f"WARNING: fold {fold_idx} {source_variant} validation is single-class; using t*=0.5.",
                    log_path,
                )

            t_star, best_val_f1 = choose_best_threshold_f1(val_y_true, val_y_prob)
            optimized_thresholds_this_fold[source_variant] = float(t_star)
            optimized_val_f1_this_fold[source_variant] = float(best_val_f1)
            optimized_thresholds_per_fold[source_variant][fold_idx] = float(t_star)

            log_line(
                f"Fold {fold_idx} optimized/{source_variant}: t*={t_star:.6f}, val_f1={best_val_f1:.6f}",
                log_path,
            )

            wandb.log({
                f"fold{fold_idx}/threshold_strategies/{source_variant}/optimized_threshold": float(t_star),
                f"fold{fold_idx}/threshold_strategies/{source_variant}/optimized_val_f1": float(best_val_f1),
            })

        inference_cache = {}
        for variant_name, variant_ckpt in variant_ckpts.items():
            if variant_ckpt is None:
                continue

            variant_state = load_checkpoint_state(variant_ckpt, device)
            variant_model, variant_module = build_model_for_state(variant_state, cfg, device)
            test_y_true, test_y_prob, test_lesion_uids = infer_probs(
                variant_model,
                test_loader,
                device,
                fold_id=fold_idx,
                model_tag=variant_name,
            )
            test_subject_ids = np.array([extract_subject_id(uid) for uid in test_lesion_uids], dtype=object)

            inference_cache[variant_name] = {
                "module": variant_module,
                "checkpoint": variant_ckpt,
                "y_true": test_y_true,
                "y_prob": test_y_prob,
                "lesion_uids": np.asarray(test_lesion_uids, dtype=object),
                "subject_ids": test_subject_ids,
            }

            fixed_metrics, _, _ = add_branch_result(
                branch_name=f"fixed_0.5/{variant_name}",
                fold_idx=fold_idx,
                threshold=fixed_threshold,
                val_best_f1=None,
                split_source=split_source,
                model_module=variant_module,
                checkpoint=variant_ckpt,
                y_true=test_y_true,
                y_prob=test_y_prob,
            )

            wandb.log({
                f"fold{fold_idx}/threshold_strategies/fixed_0.5/{variant_name}/accuracy": fixed_metrics["accuracy"],
                f"fold{fold_idx}/threshold_strategies/fixed_0.5/{variant_name}/specificity": fixed_metrics["specificity"],
                f"fold{fold_idx}/threshold_strategies/fixed_0.5/{variant_name}/recall": fixed_metrics["recall"],
                f"fold{fold_idx}/threshold_strategies/fixed_0.5/{variant_name}/precision": fixed_metrics["precision"],
                f"fold{fold_idx}/threshold_strategies/fixed_0.5/{variant_name}/f1": fixed_metrics["f1"],
                f"fold{fold_idx}/threshold_strategies/fixed_0.5/{variant_name}/roc_auc": fixed_metrics["roc_auc"],
                f"fold{fold_idx}/threshold_strategies/fixed_0.5/{variant_name}/proc_auc": fixed_metrics["proc_auc"],
                f"fold{fold_idx}/threshold_strategies/fixed_0.5/{variant_name}/pr_auc": fixed_metrics["pr_auc"],
            })

            if variant_name.endswith("ema"):
                threshold_source = "pre_retrain/best_ema"
            else:
                threshold_source = "pre_retrain/best"

            model_tag = variant_to_model_tag(variant_name)
            if model_tag is not None:
                threshold_for_export = float(fixed_threshold)
                if threshold_source in optimized_thresholds_this_fold:
                    threshold_for_export = float(optimized_thresholds_this_fold[threshold_source])

                y_true_arr = np.asarray(test_y_true, dtype=int).reshape(-1)
                y_prob_arr = np.asarray(test_y_prob, dtype=float).reshape(-1)
                uid_arr = np.asarray(test_lesion_uids, dtype=object).reshape(-1)
                subj_arr = np.asarray(test_subject_ids, dtype=object).reshape(-1)
                n_min = int(min(y_true_arr.size, y_prob_arr.size, uid_arr.size, subj_arr.size))
                for i in range(n_min):
                    raw_prediction_rows.append({
                        "fold_id": int(fold_idx),
                        "model_tag": str(model_tag),
                        "subject_id": str(subj_arr[i]),
                        "lesion_uid": canonical_uid(uid_arr[i]),
                        "y_true": int(y_true_arr[i]),
                        "y_prob": float(y_prob_arr[i]),
                        "threshold_used": float(threshold_for_export),
                        "checkpoint": str(variant_ckpt),
                    })

        fold_uid_preview = sorted({str(r["lesion_uid"]) for r in raw_prediction_rows if int(r["fold_id"]) == int(fold_idx)})[:3]
        if len(fold_uid_preview) > 0:
            log_line(f"Fold {fold_idx} canonical UID examples: {fold_uid_preview}", log_path)

            if threshold_source not in optimized_thresholds_this_fold:
                log_line(
                    f"INFO: fold {fold_idx} skipping optimized/{variant_name} (missing threshold source {threshold_source}).",
                    log_path,
                )
                continue

            opt_t = optimized_thresholds_this_fold[threshold_source]
            opt_val_f1 = optimized_val_f1_this_fold[threshold_source]
            opt_metrics, _, _ = add_branch_result(
                branch_name=f"optimized/{variant_name}",
                fold_idx=fold_idx,
                threshold=opt_t,
                val_best_f1=opt_val_f1,
                split_source=split_source,
                model_module=variant_module,
                checkpoint=variant_ckpt,
                y_true=test_y_true,
                y_prob=test_y_prob,
            )

            log_line(
                f"Fold {fold_idx} optimized/{variant_name}: threshold={opt_t:.6f}, val_f1={opt_val_f1:.6f}, "
                f"AUC={opt_metrics['roc_auc']:.4f}, pAUC@0.1={opt_metrics['proc_auc']:.4f}, PR_AUC={opt_metrics['pr_auc']:.4f}, "
                f"ACC={opt_metrics['accuracy']:.4f}, SPEC={opt_metrics['specificity']:.4f}, REC={opt_metrics['recall']:.4f}, "
                f"PREC={opt_metrics['precision']:.4f}, F1={opt_metrics['f1']:.4f}",
                log_path,
            )

            wandb.log({
                f"fold{fold_idx}/threshold_strategies/optimized/{variant_name}/threshold": float(opt_t),
                f"fold{fold_idx}/threshold_strategies/optimized/{variant_name}/val_f1": float(opt_val_f1),
                f"fold{fold_idx}/threshold_strategies/optimized/{variant_name}/accuracy": opt_metrics["accuracy"],
                f"fold{fold_idx}/threshold_strategies/optimized/{variant_name}/specificity": opt_metrics["specificity"],
                f"fold{fold_idx}/threshold_strategies/optimized/{variant_name}/recall": opt_metrics["recall"],
                f"fold{fold_idx}/threshold_strategies/optimized/{variant_name}/precision": opt_metrics["precision"],
                f"fold{fold_idx}/threshold_strategies/optimized/{variant_name}/f1": opt_metrics["f1"],
                f"fold{fold_idx}/threshold_strategies/optimized/{variant_name}/roc_auc": opt_metrics["roc_auc"],
                f"fold{fold_idx}/threshold_strategies/optimized/{variant_name}/proc_auc": opt_metrics["proc_auc"],
                f"fold{fold_idx}/threshold_strategies/optimized/{variant_name}/pr_auc": opt_metrics["pr_auc"],
            })

        fold_results.append({
            "fold": fold_idx,
            "fold_dir": fold_dir,
            "threshold": float(threshold_fold),
            "metrics": metrics_payload,
            "curves": curve_metrics,
        })

        for i in range(len(y_true)):
            lesion_rows.append({
                "fold": int(fold_idx),
                "subject_id": str(subject_ids[i]),
                "lesion_uid": canonical_uid(lesion_uids[i]),
                "label": int(y_true[i]),
                "prob": float(y_prob[i]),
                "threshold_used": float(threshold_fold),
                "pred": int(y_pred[i]),
            })

        all_labels_list.append(y_true)
        all_probs_list.append(y_prob)
        all_fold_ids_list.append(np.array([fold_idx] * len(y_true), dtype=int))
        all_subject_ids_list.append(subject_ids)
        all_lesion_uids_list.append(lesion_uids)

    fold_results = sorted(fold_results, key=lambda x: x["fold"])

    labels_all = np.concatenate(all_labels_list).astype(int)
    probs_all = np.concatenate(all_probs_list).astype(float)
    fold_ids_all = np.concatenate(all_fold_ids_list).astype(int)
    subject_ids_all = np.concatenate(all_subject_ids_list).astype(object)
    lesion_uids_all = np.concatenate(all_lesion_uids_list).astype(object)

    preds_all = np.zeros_like(labels_all)
    for fold_id, threshold in fold_thresholds.items():
        mask = (fold_ids_all == fold_id)
        preds_all[mask] = (probs_all[mask] >= threshold).astype(int)

    overall_cm = confusion_and_metrics(labels_all, preds_all)
    overall_curves = safe_auc_metrics(labels_all, probs_all)

    fpr_grid = np.linspace(0.0, 1.0, 200)
    fpr_proc_grid = np.linspace(0.0, 0.1, 200)
    recall_grid = np.linspace(0.0, 1.0, 200)

    fold_roc_interp = []
    fold_proc_interp = []
    fold_pr_interp = []

    for fr in fold_results:
        c = fr["curves"]
        fold_roc_interp.append(step_interp(c["fpr"], c["tpr"], fpr_grid))
        fold_proc_interp.append(step_interp(c["fpr_proc"], c["tpr_proc"], fpr_proc_grid))
        fold_pr_interp.append(step_interp(c["recall_curve"], c["precision_curve"], recall_grid))

    fold_roc_interp = np.vstack(fold_roc_interp)
    fold_proc_interp = np.vstack(fold_proc_interp)
    fold_pr_interp = np.vstack(fold_pr_interp)

    mean_roc = np.mean(fold_roc_interp, axis=0)
    mean_proc = np.mean(fold_proc_interp, axis=0)
    mean_pr = np.mean(fold_pr_interp, axis=0)

    roc_mean_auc = float(auc(fpr_grid, mean_roc))
    proc_mean_auc = float(auc(fpr_proc_grid, mean_proc))
    pr_mean_auc = float(auc(recall_grid, mean_pr))

    roc_plot_path = os.path.join(final_agg_dir, "roc_mean_std.png")
    pr_plot_path = os.path.join(final_agg_dir, "pr_mean_std.png")
    proc_plot_path = os.path.join(final_agg_dir, "proc_mean_std.png")

    save_curve_plot_with_folds(
        x_grid=fpr_grid,
        fold_curves=fold_roc_interp,
        mean_curve=mean_roc,
        title="Lesion-wise ROC curves (5-fold + mean)",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        out_path=roc_plot_path,
        diagonal=True,
        annotation_text=f"ROC AUC = {roc_mean_auc:.4f}",
    )
    save_curve_plot_with_folds(
        x_grid=fpr_proc_grid,
        fold_curves=fold_proc_interp,
        mean_curve=mean_proc,
        title="Lesion-wise pROC curves (FPR <= 0.1)",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        out_path=proc_plot_path,
        diagonal=False,
        annotation_text=f"pROC AUC = {proc_mean_auc:.4f}",
    )
    save_curve_plot_with_folds(
        x_grid=recall_grid,
        fold_curves=fold_pr_interp,
        mean_curve=mean_pr,
        title="Lesion-wise PR curves (5-fold + mean)",
        xlabel="Recall",
        ylabel="Precision",
        out_path=pr_plot_path,
        diagonal=False,
    )

    # keep legacy + plots/ copies
    os.replace(roc_plot_path, os.path.join(plots_dir, "roc_mean_std.png"))
    os.replace(proc_plot_path, os.path.join(plots_dir, "proc_mean_std.png"))
    os.replace(pr_plot_path, os.path.join(plots_dir, "pr_mean_std.png"))
    roc_plot_path = os.path.join(plots_dir, "roc_mean_std.png")
    proc_plot_path = os.path.join(plots_dir, "proc_mean_std.png")
    pr_plot_path = os.path.join(plots_dir, "pr_mean_std.png")

    # legacy expected filenames at FINAL_AGG root
    import shutil
    shutil.copy2(roc_plot_path, os.path.join(final_agg_dir, "roc_mean_std.png"))
    shutil.copy2(pr_plot_path, os.path.join(final_agg_dir, "pr_mean_std.png"))

    cm_sum = np.array([[overall_cm["tn"], overall_cm["fp"]], [overall_cm["fn"], overall_cm["tp"]]], dtype=int)
    cm_plot_path = os.path.join(plots_dir, "confusion_matrix_agg.png")
    save_confusion_image(cm_sum, cm_plot_path)
    shutil.copy2(cm_plot_path, os.path.join(final_agg_dir, "confusion_matrix_agg.png"))

    lesion_predictions_path = os.path.join(final_agg_dir, "lesion_predictions.csv")
    lesion_df = pd.DataFrame(lesion_rows)
    lesion_df.to_csv(lesion_predictions_path, index=False)

    bootstrap = subject_level_bootstrap(
        labels=labels_all,
        probs=probs_all,
        fold_ids=fold_ids_all,
        subject_ids=subject_ids_all,
        fold_thresholds=fold_thresholds,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
    )

    bootstrap_npz_path = os.path.join(final_agg_dir, "bootstrap_distributions.npz")
    np.savez(bootstrap_npz_path, **bootstrap)

    bootstrap_ci = {k: bootstrap_summary(v) for k, v in bootstrap.items()}

    subjects_unique = np.unique(subject_ids_all)
    pred_counts = []
    gold_counts = []
    for sid in subjects_unique:
        mask = (subject_ids_all == sid)
        pred_counts.append(int(np.sum(preds_all[mask] == 1)))
        gold_counts.append(int(np.sum(labels_all[mask] == 1)))
    pred_counts = np.array(pred_counts, dtype=float)
    gold_counts = np.array(gold_counts, dtype=float)

    subject_stats = {
        "pred_count_mean": float(np.mean(pred_counts)) if len(pred_counts) else 0.0,
        "pred_count_min": float(np.min(pred_counts)) if len(pred_counts) else 0.0,
        "pred_count_max": float(np.max(pred_counts)) if len(pred_counts) else 0.0,
        "pred_count_median": float(np.median(pred_counts)) if len(pred_counts) else 0.0,
        "gold_count_mean": float(np.mean(gold_counts)) if len(gold_counts) else 0.0,
        "gold_count_min": float(np.min(gold_counts)) if len(gold_counts) else 0.0,
        "gold_count_max": float(np.max(gold_counts)) if len(gold_counts) else 0.0,
        "gold_count_median": float(np.median(gold_counts)) if len(gold_counts) else 0.0,
        "mse_count": float(np.mean((pred_counts - gold_counts) ** 2)) if len(pred_counts) else 0.0,
    }
    pearson_r, pearson_ci_low, pearson_ci_high = pearson_with_fisher_ci(gold_counts, pred_counts)
    subject_stats["pearson_r"] = float(pearson_r) if np.isfinite(pearson_r) else np.nan
    subject_stats["pearson_ci_low"] = float(pearson_ci_low) if np.isfinite(pearson_ci_low) else np.nan
    subject_stats["pearson_ci_high"] = float(pearson_ci_high) if np.isfinite(pearson_ci_high) else np.nan

    subject_scatter_path = os.path.join(plots_dir, "subject_count_scatter.png")
    save_subject_scatter(gold_counts, pred_counts, subject_scatter_path)

    metric_keys = ["accuracy", "specificity", "recall", "precision", "f1", "roc_auc", "proc_auc", "pr_auc"]

    branch_aggregates = {}
    for branch_name, entries in branch_results.items():
        agg = aggregate_branch_entries(entries)
        if agg is not None:
            branch_aggregates[branch_name] = agg

    raw_prediction_columns = ["fold_id", "model_tag", "subject_id", "lesion_uid", "y_true", "y_prob", "threshold_used", "checkpoint"]
    raw_predictions_df = pd.DataFrame(raw_prediction_rows, columns=raw_prediction_columns)
    if len(raw_predictions_df) > 0:
        raw_predictions_df = raw_predictions_df.sort_values(
            by=["fold_id", "model_tag", "lesion_uid", "subject_id", "y_true", "y_prob", "checkpoint"],
            kind="mergesort",
        )
        dup_mask = raw_predictions_df.duplicated(subset=["fold_id", "model_tag", "lesion_uid"], keep=False)
        if dup_mask.any():
            dup_sample = raw_predictions_df.loc[dup_mask, ["fold_id", "model_tag", "lesion_uid"]].head(10)
            raise RuntimeError(
                "Duplicate canonical UID rows found for (fold_id, model_tag, lesion_uid). "
                "This breaks reliable pairing across experiments. Sample duplicates: "
                + dup_sample.to_dict(orient="records").__str__()
            )
        raw_predictions_df = raw_predictions_df.reset_index(drop=True)

    raw_predictions_csv_path = os.path.join(final_agg_dir, "raw_model_predictions.csv")
    if len(raw_predictions_df) > 0:
        raw_predictions_df.to_csv(raw_predictions_csv_path, index=False)
    else:
        pd.DataFrame(columns=raw_prediction_columns).to_csv(raw_predictions_csv_path, index=False)

    delong_rows = []
    wilcoxon_rows = []
    fold_auc_rows = []
    auc_bootstrap_dist_rows = []
    auc_bootstrap_summary_rows = []

    if len(raw_predictions_df) > 0:
        available_tags = set(raw_predictions_df["model_tag"].astype(str).unique().tolist())
        comparisons = [
            ("best", "best_ema"),
            ("best", "post_retrain"),
            ("best_ema", "post_retrain_ema"),
        ]
        comparisons = [(a, b) for a, b in comparisons if (a in available_tags and b in available_tags)]

        folds_sorted = sorted(raw_predictions_df["fold_id"].astype(int).unique().tolist())

        for tag in sorted(available_tags):
            for fold_id in folds_sorted:
                fold_df = raw_predictions_df[(raw_predictions_df["model_tag"] == tag) & (raw_predictions_df["fold_id"] == fold_id)]
                y_fold = fold_df["y_true"].to_numpy(dtype=int)
                p_fold = fold_df["y_prob"].to_numpy(dtype=float)
                if y_fold.size == 0 or np.unique(y_fold).size < 2:
                    roc_auc_fold = np.nan
                else:
                    roc_auc_fold = float(roc_auc_score(y_fold, p_fold))
                fold_auc_rows.append({
                    "model_tag": str(tag),
                    "fold_id": int(fold_id),
                    "roc_auc": float(roc_auc_fold) if np.isfinite(roc_auc_fold) else np.nan,
                    "n_samples": int(y_fold.size),
                    "n_pos": int(np.sum(y_fold == 1)),
                })

        fold_auc_df = pd.DataFrame(fold_auc_rows)

        for a_tag, b_tag in comparisons:
            for fold_id in folds_sorted:
                _, y_true_d, y_prob_a_d, y_prob_b_d = align_model_pair_predictions(
                    raw_predictions_df, a_tag, b_tag, fold_id=fold_id
                )
                if y_true_d.size == 0:
                    continue

                auc_a, auc_b, delta_auc, p_value = delong_roc_test(y_true_d, y_prob_a_d, y_prob_b_d)
                delong_rows.append({
                    "A_tag": str(a_tag),
                    "B_tag": str(b_tag),
                    "comparison": f"{a_tag}_vs_{b_tag}",
                    "fold_id": int(fold_id),
                    "auc_A": float(auc_a) if np.isfinite(auc_a) else np.nan,
                    "auc_B": float(auc_b) if np.isfinite(auc_b) else np.nan,
                    "delta_auc": float(delta_auc) if np.isfinite(delta_auc) else np.nan,
                    "p_value": float(p_value),
                    "n_samples": int(y_true_d.size),
                    "n_pos": int(np.sum(y_true_d == 1)),
                })

                wandb.log({
                    f"stats/delong/{a_tag}_vs_{b_tag}/fold{fold_id}/auc_A": float(auc_a) if np.isfinite(auc_a) else np.nan,
                    f"stats/delong/{a_tag}_vs_{b_tag}/fold{fold_id}/auc_B": float(auc_b) if np.isfinite(auc_b) else np.nan,
                    f"stats/delong/{a_tag}_vs_{b_tag}/fold{fold_id}/delta_auc": float(delta_auc) if np.isfinite(delta_auc) else np.nan,
                    f"stats/delong/{a_tag}_vs_{b_tag}/fold{fold_id}/p_value": float(p_value),
                    f"stats/delong/{a_tag}_vs_{b_tag}/fold{fold_id}/n_samples": int(y_true_d.size),
                    f"stats/delong/{a_tag}_vs_{b_tag}/fold{fold_id}/n_pos": int(np.sum(y_true_d == 1)),
                })

            _, y_true_all_d, y_prob_a_all_d, y_prob_b_all_d = align_model_pair_predictions(
                raw_predictions_df, a_tag, b_tag, fold_id=None
            )
            if y_true_all_d.size > 0:
                auc_a, auc_b, delta_auc, p_value = delong_roc_test(y_true_all_d, y_prob_a_all_d, y_prob_b_all_d)
                delong_rows.append({
                    "A_tag": str(a_tag),
                    "B_tag": str(b_tag),
                    "comparison": f"{a_tag}_vs_{b_tag}",
                    "fold_id": "overall",
                    "auc_A": float(auc_a) if np.isfinite(auc_a) else np.nan,
                    "auc_B": float(auc_b) if np.isfinite(auc_b) else np.nan,
                    "delta_auc": float(delta_auc) if np.isfinite(delta_auc) else np.nan,
                    "p_value": float(p_value),
                    "n_samples": int(y_true_all_d.size),
                    "n_pos": int(np.sum(y_true_all_d == 1)),
                })

                wandb.log({
                    f"stats/delong/{a_tag}_vs_{b_tag}/overall/auc_A": float(auc_a) if np.isfinite(auc_a) else np.nan,
                    f"stats/delong/{a_tag}_vs_{b_tag}/overall/auc_B": float(auc_b) if np.isfinite(auc_b) else np.nan,
                    f"stats/delong/{a_tag}_vs_{b_tag}/overall/delta_auc": float(delta_auc) if np.isfinite(delta_auc) else np.nan,
                    f"stats/delong/{a_tag}_vs_{b_tag}/overall/p_value": float(p_value),
                    f"stats/delong/{a_tag}_vs_{b_tag}/overall/n_samples": int(y_true_all_d.size),
                    f"stats/delong/{a_tag}_vs_{b_tag}/overall/n_pos": int(np.sum(y_true_all_d == 1)),
                })

            vec_a = []
            vec_b = []
            fold_ids_common = []
            for fold_id in folds_sorted:
                row_a = fold_auc_df[(fold_auc_df["model_tag"] == a_tag) & (fold_auc_df["fold_id"] == fold_id)]
                row_b = fold_auc_df[(fold_auc_df["model_tag"] == b_tag) & (fold_auc_df["fold_id"] == fold_id)]
                if len(row_a) == 0 or len(row_b) == 0:
                    continue
                vec_a.append(float(row_a.iloc[0]["roc_auc"]))
                vec_b.append(float(row_b.iloc[0]["roc_auc"]))
                fold_ids_common.append(int(fold_id))

            wilc = run_wilcoxon_fold_auc(vec_a, vec_b)
            arr_a = np.asarray(vec_a, dtype=float)
            arr_b = np.asarray(vec_b, dtype=float)
            wilcoxon_rows.append({
                "A_tag": str(a_tag),
                "B_tag": str(b_tag),
                "comparison": f"{a_tag}_vs_{b_tag}",
                "statistic": float(wilc["statistic"]),
                "p_value": float(wilc["p_value"]),
                "n_folds": int(wilc["n_folds"]),
                "meanA": float(np.nanmean(arr_a)) if arr_a.size > 0 else np.nan,
                "stdA": float(np.nanstd(arr_a)) if arr_a.size > 0 else np.nan,
                "meanB": float(np.nanmean(arr_b)) if arr_b.size > 0 else np.nan,
                "stdB": float(np.nanstd(arr_b)) if arr_b.size > 0 else np.nan,
                "fold_ids": json.dumps(fold_ids_common),
                "vecA": json.dumps([None if not np.isfinite(v) else float(v) for v in vec_a]),
                "vecB": json.dumps([None if not np.isfinite(v) else float(v) for v in vec_b]),
                "note": str(wilc["note"]),
            })

            wandb.log({
                f"stats/wilcoxon/{a_tag}_vs_{b_tag}/statistic": float(wilc["statistic"]),
                f"stats/wilcoxon/{a_tag}_vs_{b_tag}/p_value": float(wilc["p_value"]),
                f"stats/wilcoxon/{a_tag}_vs_{b_tag}/n_folds": int(wilc["n_folds"]),
                f"stats/wilcoxon/{a_tag}_vs_{b_tag}/mean_auc_A": float(np.nanmean(arr_a)) if arr_a.size > 0 else np.nan,
                f"stats/wilcoxon/{a_tag}_vs_{b_tag}/mean_auc_B": float(np.nanmean(arr_b)) if arr_b.size > 0 else np.nan,
            })

        for tag in sorted(available_tags):
            for fold_id in folds_sorted + ["overall"]:
                if fold_id == "overall":
                    sub_df = raw_predictions_df[raw_predictions_df["model_tag"] == tag]
                    seed_offset = 100000 + int(sum(ord(c) for c in str(tag)))
                else:
                    sub_df = raw_predictions_df[(raw_predictions_df["model_tag"] == tag) & (raw_predictions_df["fold_id"] == int(fold_id))]
                    seed_offset = int(fold_id) * 1000 + int(sum(ord(c) for c in str(tag)))

                y_sub = sub_df["y_true"].to_numpy(dtype=int)
                p_sub = sub_df["y_prob"].to_numpy(dtype=float)
                s_sub = sub_df["subject_id"].astype(str).to_numpy(dtype=object)

                if y_sub.size == 0 or np.unique(y_sub).size < 2:
                    roc_auc_point = np.nan
                else:
                    try:
                        roc_auc_point = float(roc_auc_score(y_sub, p_sub))
                    except ValueError:
                        roc_auc_point = np.nan

                boot_aucs = bootstrap_subject_auc_distribution(
                    y_true=y_sub,
                    y_prob=p_sub,
                    subject_ids=s_sub,
                    n_boot=args.num_bootstrap_auc,
                    seed=args.seed + seed_offset,
                )
                boot_summary = summarize_auc_bootstrap(boot_aucs)

                fold_label = "overall" if fold_id == "overall" else int(fold_id)
                for rep_idx, auc_val in enumerate(boot_aucs):
                    auc_bootstrap_dist_rows.append({
                        "model_tag": str(tag),
                        "fold_id": fold_label,
                        "replicate_idx": int(rep_idx),
                        "boot_auc": float(auc_val),
                    })

                auc_bootstrap_summary_rows.append({
                    "model_tag": str(tag),
                    "fold_id": fold_label,
                    "roc_auc_point": float(roc_auc_point) if np.isfinite(roc_auc_point) else np.nan,
                    "ci_low": float(boot_summary["ci_low"]) if np.isfinite(boot_summary["ci_low"]) else np.nan,
                    "ci_high": float(boot_summary["ci_high"]) if np.isfinite(boot_summary["ci_high"]) else np.nan,
                    "median": float(boot_summary["median"]) if np.isfinite(boot_summary["median"]) else np.nan,
                    "mean": float(boot_summary["mean"]) if np.isfinite(boot_summary["mean"]) else np.nan,
                    "n_boot": int(args.num_bootstrap_auc),
                    "n_boot_valid": int(boot_summary["n_boot_valid"]),
                    "n_subjects": int(np.unique(s_sub).size),
                    "n_samples": int(y_sub.size),
                    "n_pos": int(np.sum(y_sub == 1)),
                })

                wandb_prefix = f"stats/patient_bootstrap_auc/{tag}/{fold_label}"
                wandb.log({
                    f"{wandb_prefix}/roc_auc_point": float(roc_auc_point) if np.isfinite(roc_auc_point) else np.nan,
                    f"{wandb_prefix}/ci_low": float(boot_summary["ci_low"]) if np.isfinite(boot_summary["ci_low"]) else np.nan,
                    f"{wandb_prefix}/ci_high": float(boot_summary["ci_high"]) if np.isfinite(boot_summary["ci_high"]) else np.nan,
                    f"{wandb_prefix}/n_boot_valid": int(boot_summary["n_boot_valid"]),
                })

                finite_boot = boot_aucs[np.isfinite(boot_aucs)]
                if finite_boot.size > 0:
                    wandb.log({f"{wandb_prefix}/hist": wandb.Histogram(finite_boot)})

    delong_df = pd.DataFrame(
        delong_rows,
        columns=["A_tag", "B_tag", "comparison", "fold_id", "auc_A", "auc_B", "delta_auc", "p_value", "n_samples", "n_pos"],
    )
    wilcoxon_df = pd.DataFrame(
        wilcoxon_rows,
        columns=["A_tag", "B_tag", "comparison", "statistic", "p_value", "n_folds", "meanA", "stdA", "meanB", "stdB", "fold_ids", "vecA", "vecB", "note"],
    )
    fold_auc_df = pd.DataFrame(
        fold_auc_rows,
        columns=["model_tag", "fold_id", "roc_auc", "n_samples", "n_pos"],
    )
    auc_bootstrap_dist_df = pd.DataFrame(
        auc_bootstrap_dist_rows,
        columns=["model_tag", "fold_id", "replicate_idx", "boot_auc"],
    )
    auc_bootstrap_summary_df = pd.DataFrame(
        auc_bootstrap_summary_rows,
        columns=["model_tag", "fold_id", "roc_auc_point", "ci_low", "ci_high", "median", "mean", "n_boot", "n_boot_valid", "n_subjects", "n_samples", "n_pos"],
    )

    delong_csv_path = os.path.join(final_agg_dir, "delong_results.csv")
    wilcoxon_csv_path = os.path.join(final_agg_dir, "wilcoxon_results.csv")
    fold_auc_csv_path = os.path.join(final_agg_dir, "fold_auc_by_model.csv")
    auc_bootstrap_dist_csv_path = os.path.join(final_agg_dir, "patient_bootstrap_auc_distribution.csv")
    auc_bootstrap_summary_csv_path = os.path.join(final_agg_dir, "patient_bootstrap_auc_summary.csv")

    delong_df.to_csv(delong_csv_path, index=False)
    wilcoxon_df.to_csv(wilcoxon_csv_path, index=False)
    fold_auc_df.to_csv(fold_auc_csv_path, index=False)
    auc_bootstrap_dist_df.to_csv(auc_bootstrap_dist_csv_path, index=False)
    auc_bootstrap_summary_df.to_csv(auc_bootstrap_summary_csv_path, index=False)

    stats_summary = {
        "delong": delong_rows,
        "wilcoxon": wilcoxon_rows,
        "fold_auc": fold_auc_rows,
        "patient_bootstrap_auc_summary": auc_bootstrap_summary_rows,
    }
    stats_summary_path = os.path.join(final_agg_dir, "model_comparison_stats.json")
    with open(stats_summary_path, "w") as f:
        json.dump(stats_summary, f, indent=2)

    if wandb.run is not None:
        wandb.save(raw_predictions_csv_path)
        wandb.save(delong_csv_path)
        wandb.save(wilcoxon_csv_path)
        wandb.save(fold_auc_csv_path)
        wandb.save(auc_bootstrap_dist_csv_path)
        wandb.save(auc_bootstrap_summary_csv_path)
        wandb.save(stats_summary_path)

    if args.export_csv:
        metrics_columns = [
            "exp", "model_tag", "split", "fold", "threshold_name", "threshold", "n",
            "tp", "fp", "tn", "fn",
            "accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "roc_auc", "pr_auc",
        ]
        roc_columns = ["exp", "model_tag", "fold", "fpr", "tpr", "threshold", "curve_type"]
        pr_columns = ["exp", "model_tag", "fold", "recall", "precision", "threshold", "curve_type"]
        curve_mean_columns = [
            "exp", "model_tag", "curve_type", "x", "y_mean", "y_std", "y_low", "y_high", "method"
        ]

        metrics_rows = []
        roc_rows = []
        pr_rows = []
        curve_mean_rows = []

        grouped_entries = {}
        curve_seen = set()
        model_fold_curves = {}

        for branch_name, entries in sorted(branch_results.items()):
            threshold_name, model_tag = parse_branch_name_for_export(branch_name)
            if model_tag is None:
                continue

            key = (threshold_name, model_tag)
            grouped_entries.setdefault(key, [])

            for e in sorted(entries, key=lambda x: x["fold"]):
                grouped_entries[key].append(e)

                y_true_fold = np.asarray(e["y_true"], dtype=int).reshape(-1)
                n_fold = int(y_true_fold.size)
                tp = int(e["metrics"]["tp"])
                fp = int(e["metrics"]["fp"])
                tn = int(e["metrics"]["tn"])
                fn = int(e["metrics"]["fn"])
                assert (tp + fp + tn + fn) == n_fold, (
                    f"Confusion-matrix sum mismatch for {branch_name} fold {e['fold']}: "
                    f"tp+fp+tn+fn={tp+fp+tn+fn}, n={n_fold}"
                )

                npv = (tn / (tn + fn)) if (tn + fn) > 0 else np.nan
                metrics_rows.append({
                    "exp": args.experiment,
                    "model_tag": model_tag,
                    "split": "test",
                    "fold": int(e["fold"]),
                    "threshold_name": threshold_name,
                    "threshold": float(e["threshold"]),
                    "n": n_fold,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "accuracy": float(e["metrics"]["accuracy"]),
                    "sensitivity": float(e["metrics"]["recall"]),
                    "specificity": float(e["metrics"]["specificity"]),
                    "ppv": float(e["metrics"]["precision"]),
                    "npv": float(npv) if np.isfinite(npv) else np.nan,
                    "f1": float(e["metrics"]["f1"]),
                    "roc_auc": float(e["metrics"]["roc_auc"]),
                    "pr_auc": float(e["metrics"]["pr_auc"]),
                })

                curve_key = (model_tag, int(e["fold"]))
                if curve_key not in curve_seen:
                    curve_seen.add(curve_key)
                    curves = e.get("curves", {})
                    fpr = np.asarray(curves.get("fpr", []), dtype=float)
                    tpr = np.asarray(curves.get("tpr", []), dtype=float)
                    roc_thresholds = np.asarray(curves.get("roc_thresholds", []), dtype=float)

                    recall = np.asarray(curves.get("recall_curve", []), dtype=float)
                    precision = np.asarray(curves.get("precision_curve", []), dtype=float)
                    pr_thresholds = np.asarray(curves.get("pr_thresholds", []), dtype=float)

                    model_fold_curves[curve_key] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "recall": recall,
                        "precision": precision,
                    }

                    if fpr.size == 0 or tpr.size == 0:
                        roc_rows.append({
                            "exp": args.experiment,
                            "model_tag": model_tag,
                            "fold": int(e["fold"]),
                            "fpr": np.nan,
                            "tpr": np.nan,
                            "threshold": np.nan,
                            "curve_type": "roc",
                        })
                    else:
                        for i in range(min(fpr.size, tpr.size)):
                            thr = float(roc_thresholds[i]) if i < roc_thresholds.size else np.nan
                            roc_rows.append({
                                "exp": args.experiment,
                                "model_tag": model_tag,
                                "fold": int(e["fold"]),
                                "fpr": float(fpr[i]),
                                "tpr": float(tpr[i]),
                                "threshold": thr,
                                "curve_type": "roc",
                            })

                    if recall.size == 0 or precision.size == 0:
                        pr_rows.append({
                            "exp": args.experiment,
                            "model_tag": model_tag,
                            "fold": int(e["fold"]),
                            "recall": np.nan,
                            "precision": np.nan,
                            "threshold": np.nan,
                            "curve_type": "pr",
                        })
                    else:
                        for i in range(min(recall.size, precision.size)):
                            thr = float(pr_thresholds[i]) if i < pr_thresholds.size else np.nan
                            pr_rows.append({
                                "exp": args.experiment,
                                "model_tag": model_tag,
                                "fold": int(e["fold"]),
                                "recall": float(recall[i]),
                                "precision": float(precision[i]),
                                "threshold": thr,
                                "curve_type": "pr",
                            })

        for (threshold_name, model_tag), entries in sorted(grouped_entries.items()):
            tp_sum = int(np.sum([int(e["metrics"]["tp"]) for e in entries]))
            fp_sum = int(np.sum([int(e["metrics"]["fp"]) for e in entries]))
            tn_sum = int(np.sum([int(e["metrics"]["tn"]) for e in entries]))
            fn_sum = int(np.sum([int(e["metrics"]["fn"]) for e in entries]))

            y_true_all = np.concatenate([np.asarray(e["y_true"], dtype=int).reshape(-1) for e in entries])
            y_prob_all = np.concatenate([np.asarray(e["y_prob"], dtype=float).reshape(-1) for e in entries])
            n_all = int(y_true_all.size)
            assert (tp_sum + fp_sum + tn_sum + fn_sum) == n_all, (
                f"Confusion-matrix sum mismatch for {threshold_name}/{model_tag} ALL: "
                f"tp+fp+tn+fn={tp_sum+fp_sum+tn_sum+fn_sum}, n={n_all}"
            )

            acc = ((tp_sum + tn_sum) / n_all) if n_all > 0 else np.nan
            sensitivity = (tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else np.nan
            specificity = (tn_sum / (tn_sum + fp_sum)) if (tn_sum + fp_sum) > 0 else np.nan
            ppv = (tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else np.nan
            npv = (tn_sum / (tn_sum + fn_sum)) if (tn_sum + fn_sum) > 0 else np.nan
            f1 = (2 * ppv * sensitivity / (ppv + sensitivity)) if np.isfinite(ppv) and np.isfinite(sensitivity) and (ppv + sensitivity) > 0 else np.nan

            # For ALL rows, AUC values are computed on pooled fold predictions.
            try:
                roc_auc_all = float(roc_auc_score(y_true_all, y_prob_all))
            except ValueError:
                roc_auc_all = np.nan
            try:
                precision_curve_all, recall_curve_all, _ = precision_recall_curve(y_true_all, y_prob_all)
                pr_auc_all = float(auc(recall_curve_all, precision_curve_all))
            except ValueError:
                pr_auc_all = np.nan

            fold_thresholds_all = np.array([float(e["threshold"]) for e in entries], dtype=float)
            threshold_all = float(fold_thresholds_all[0]) if np.allclose(fold_thresholds_all, fold_thresholds_all[0]) else np.nan

            metrics_rows.append({
                "exp": args.experiment,
                "model_tag": model_tag,
                "split": "test",
                "fold": "ALL",
                "threshold_name": threshold_name,
                "threshold": threshold_all,
                "n": n_all,
                "tp": tp_sum,
                "fp": fp_sum,
                "tn": tn_sum,
                "fn": fn_sum,
                "accuracy": float(acc) if np.isfinite(acc) else np.nan,
                "sensitivity": float(sensitivity) if np.isfinite(sensitivity) else np.nan,
                "specificity": float(specificity) if np.isfinite(specificity) else np.nan,
                "ppv": float(ppv) if np.isfinite(ppv) else np.nan,
                "npv": float(npv) if np.isfinite(npv) else np.nan,
                "f1": float(f1) if np.isfinite(f1) else np.nan,
                "roc_auc": roc_auc_all,
                "pr_auc": pr_auc_all,
            })

        roc_grid = np.linspace(0.0, 1.0, 1001)
        recall_grid = np.linspace(0.0, 1.0, 1001)

        for model_tag in ["best", "best_ema", "post_retrain", "post_retrain_ema"]:
            fold_keys = sorted([k for k in model_fold_curves.keys() if k[0] == model_tag], key=lambda x: x[1])
            if len(fold_keys) == 0:
                continue

            roc_stack = []
            pr_stack = []
            for fk in fold_keys:
                c = model_fold_curves[fk]
                if c["fpr"].size > 0 and c["tpr"].size > 0:
                    roc_stack.append(step_interp(c["fpr"], c["tpr"], roc_grid))
                if c["recall"].size > 0 and c["precision"].size > 0:
                    pr_stack.append(step_interp(c["recall"], c["precision"], recall_grid))

            if len(roc_stack) > 0:
                roc_stack = np.vstack(roc_stack)
                roc_mean = np.mean(roc_stack, axis=0)
                roc_std = np.std(roc_stack, axis=0)
                roc_half = 1.96 * roc_std / np.sqrt(roc_stack.shape[0])
                for i, x in enumerate(roc_grid):
                    curve_mean_rows.append({
                        "exp": args.experiment,
                        "model_tag": model_tag,
                        "curve_type": "roc",
                        "x": float(x),
                        "y_mean": float(roc_mean[i]),
                        "y_std": float(roc_std[i]),
                        "y_low": float(roc_mean[i] - roc_half[i]),
                        "y_high": float(roc_mean[i] + roc_half[i]),
                        "method": "fold_mean_interp",
                    })

            if len(pr_stack) > 0:
                pr_stack = np.vstack(pr_stack)
                pr_mean = np.mean(pr_stack, axis=0)
                pr_std = np.std(pr_stack, axis=0)
                pr_half = 1.96 * pr_std / np.sqrt(pr_stack.shape[0])
                for i, x in enumerate(recall_grid):
                    curve_mean_rows.append({
                        "exp": args.experiment,
                        "model_tag": model_tag,
                        "curve_type": "pr",
                        "x": float(x),
                        "y_mean": float(pr_mean[i]),
                        "y_std": float(pr_std[i]),
                        "y_low": float(pr_mean[i] - pr_half[i]),
                        "y_high": float(pr_mean[i] + pr_half[i]),
                        "method": "fold_mean_interp",
                    })

        metrics_csv_path = os.path.join(export_path, "metrics_summary.csv")
        roc_csv_path = os.path.join(export_path, "roc_curve_points.csv")
        pr_csv_path = os.path.join(export_path, "pr_curve_points.csv")
        curve_mean_csv_path = os.path.join(export_path, "curve_mean.csv")

        pd.DataFrame(metrics_rows, columns=metrics_columns).to_csv(metrics_csv_path, index=False)
        pd.DataFrame(roc_rows, columns=roc_columns).to_csv(roc_csv_path, index=False)
        pd.DataFrame(pr_rows, columns=pr_columns).to_csv(pr_csv_path, index=False)
        pd.DataFrame(curve_mean_rows, columns=curve_mean_columns).to_csv(curve_mean_csv_path, index=False)

        if wandb.run is not None:
            wandb.save(metrics_csv_path)
            wandb.save(roc_csv_path)
            wandb.save(pr_csv_path)
            wandb.save(curve_mean_csv_path)
            wandb.save(os.path.join(export_path, "*.csv"))

    metrics_summary = {
        "experiment": args.experiment,
        "threshold_input": float(args.threshold),
        "thresholds_per_fold": {str(k): float(v) for k, v in fold_thresholds.items()},
        "final": {
            "roc_auc": float(overall_curves["roc_auc"]),
            "proc_auc": float(overall_curves["proc_auc"]),
            "pr_auc": float(overall_curves["pr_auc"]),
            "accuracy": overall_cm["accuracy"],
            "specificity": overall_cm["specificity"],
            "recall": overall_cm["recall"],
            "precision": overall_cm["precision"],
            "f1": overall_cm["f1"],
            "tp": overall_cm["tp"],
            "fp": overall_cm["fp"],
            "fn": overall_cm["fn"],
            "tn": overall_cm["tn"],
        },
        "cv_mean_curve_auc": {
            "roc_auc": roc_mean_auc,
            "proc_auc": proc_mean_auc,
            "pr_auc": pr_mean_auc,
        },
        "bootstrap": bootstrap_ci,
        "subject": subject_stats,
        "folds": [
            {
                "fold": int(fr["fold"]),
                "threshold": float(fr["threshold"]),
                **fr["metrics"],
            }
            for fr in fold_results
        ],
        "optimized_thresholds_pre_retrain": {
            key: {str(k): float(v) for k, v in value.items()}
            for key, value in optimized_thresholds_per_fold.items()
        },
        "additional_results": branch_aggregates,
        "model_comparison_stats": stats_summary,
    }

    metrics_summary_path = os.path.join(final_agg_dir, "metrics_summary.json")
    with open(metrics_summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # keep previous summary outputs
    summary_rows = []
    for fr in fold_results:
        m = fr["metrics"]
        summary_rows.append({
            "fold": fr["fold"],
            "threshold": fr["threshold"],
            "accuracy": m["accuracy"],
            "specificity": m["specificity"],
            "recall": m["recall"],
            "precision": m["precision"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "proc_auc": m["proc_auc"],
            "pr_auc": m["pr_auc"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
        })

    summary_rows.append({
        "fold": "overall",
        "threshold": "per-fold",
        "accuracy": overall_cm["accuracy"],
        "specificity": overall_cm["specificity"],
        "recall": overall_cm["recall"],
        "precision": overall_cm["precision"],
        "f1": overall_cm["f1"],
        "roc_auc": overall_curves["roc_auc"],
        "proc_auc": overall_curves["proc_auc"],
        "pr_auc": overall_curves["pr_auc"],
        "tp": overall_cm["tp"],
        "fp": overall_cm["fp"],
        "fn": overall_cm["fn"],
        "tn": overall_cm["tn"],
    })

    pd.DataFrame(summary_rows).to_csv(os.path.join(final_agg_dir, "final_metrics_summary.csv"), index=False)
    with open(os.path.join(final_agg_dir, "final_metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    branch_summary_rows = []
    for branch_name, agg in sorted(branch_aggregates.items()):
        for fold_data in agg["folds"]:
            branch_summary_rows.append({
                "branch": branch_name,
                "fold": fold_data["fold"],
                "threshold": fold_data["threshold"],
                "val_best_f1": fold_data["val_best_f1"],
                "accuracy": fold_data["accuracy"],
                "specificity": fold_data["specificity"],
                "recall": fold_data["recall"],
                "precision": fold_data["precision"],
                "f1": fold_data["f1"],
                "roc_auc": fold_data["roc_auc"],
                "proc_auc": fold_data["proc_auc"],
                "pr_auc": fold_data["pr_auc"],
                "tp": fold_data["tp"],
                "fp": fold_data["fp"],
                "fn": fold_data["fn"],
                "tn": fold_data["tn"],
                "checkpoint": fold_data["checkpoint"],
                "model_module": fold_data["model_module"],
            })

        branch_summary_rows.append({
            "branch": branch_name,
            "fold": "overall",
            "threshold": "per-fold",
            "val_best_f1": np.nan,
            "accuracy": agg["final"]["accuracy"],
            "specificity": agg["final"]["specificity"],
            "recall": agg["final"]["recall"],
            "precision": agg["final"]["precision"],
            "f1": agg["final"]["f1"],
            "roc_auc": agg["final"]["roc_auc"],
            "proc_auc": agg["final"]["proc_auc"],
            "pr_auc": agg["final"]["pr_auc"],
            "tp": agg["final"]["tp"],
            "fp": agg["final"]["fp"],
            "fn": agg["final"]["fn"],
            "tn": agg["final"]["tn"],
            "checkpoint": "",
            "model_module": "",
        })

    if len(branch_summary_rows) > 0:
        pd.DataFrame(branch_summary_rows).to_csv(
            os.path.join(final_agg_dir, "final_metrics_summary_branches.csv"),
            index=False,
        )

    if args.export_csv:
        raw_predictions_df.to_csv(os.path.join(export_path, "raw_model_predictions.csv"), index=False)
        delong_df.to_csv(os.path.join(export_path, "delong_results.csv"), index=False)
        wilcoxon_df.to_csv(os.path.join(export_path, "wilcoxon_results.csv"), index=False)
        fold_auc_df.to_csv(os.path.join(export_path, "fold_auc_by_model.csv"), index=False)
        auc_bootstrap_dist_df.to_csv(os.path.join(export_path, "patient_bootstrap_auc_distribution.csv"), index=False)
        auc_bootstrap_summary_df.to_csv(os.path.join(export_path, "patient_bootstrap_auc_summary.csv"), index=False)
        with open(os.path.join(export_path, "model_comparison_stats.json"), "w") as f:
            json.dump(stats_summary, f, indent=2)

    # wandb scalar logging
    wandb.log({
        "final/roc_auc": float(overall_curves["roc_auc"]),
        "final/proc_auc": float(overall_curves["proc_auc"]),
        "final/pr_auc": float(overall_curves["pr_auc"]),
        "final/accuracy": overall_cm["accuracy"],
        "final/specificity": overall_cm["specificity"],
        "final/recall": overall_cm["recall"],
        "final/precision": overall_cm["precision"],
        "final/f1": overall_cm["f1"],
        "final/tp": overall_cm["tp"],
        "final/fp": overall_cm["fp"],
        "final/fn": overall_cm["fn"],
        "final/tn": overall_cm["tn"],
        "final/roc_auc_mean_curve": roc_mean_auc,
        "final/proc_auc_mean_curve": proc_mean_auc,
        "final/pr_auc_mean_curve": pr_mean_auc,
    })

    for m in metric_keys:
        bs = bootstrap_ci[m]
        wandb.log({
            f"bootstrap/{m}_mean": bs["mean"],
            f"bootstrap/{m}_ci_low": bs["ci_low"],
            f"bootstrap/{m}_ci_high": bs["ci_high"],
        })
        finite_vals = bootstrap[m][np.isfinite(bootstrap[m])]
        if finite_vals.size > 0:
            wandb.log({f"bootstrap/{m}_hist": wandb.Histogram(finite_vals)})

    wandb.log({
        "subject/pred_count_mean": subject_stats["pred_count_mean"],
        "subject/pred_count_min": subject_stats["pred_count_min"],
        "subject/pred_count_max": subject_stats["pred_count_max"],
        "subject/pred_count_median": subject_stats["pred_count_median"],
        "subject/gold_count_mean": subject_stats["gold_count_mean"],
        "subject/gold_count_min": subject_stats["gold_count_min"],
        "subject/gold_count_max": subject_stats["gold_count_max"],
        "subject/gold_count_median": subject_stats["gold_count_median"],
        "subject/pearson_r": subject_stats["pearson_r"],
        "subject/pearson_ci_low": subject_stats["pearson_ci_low"],
        "subject/pearson_ci_high": subject_stats["pearson_ci_high"],
        "subject/mse_count": subject_stats["mse_count"],
    })

    wandb.log({
        "plots/roc": wandb.Image(roc_plot_path),
        "plots/proc": wandb.Image(proc_plot_path),
        "plots/pr": wandb.Image(pr_plot_path),
        "plots/confusion_matrix": wandb.Image(cm_plot_path),
        "plots/subject_scatter": wandb.Image(subject_scatter_path),
    })

    # keep previous cv/* logs for compatibility
    wandb.log({
        "cv/accuracy_mean": float(np.mean([fr["metrics"]["accuracy"] for fr in fold_results])),
        "cv/precision_mean": float(np.mean([fr["metrics"]["precision"] for fr in fold_results])),
        "cv/recall_mean": float(np.mean([fr["metrics"]["recall"] for fr in fold_results])),
        "cv/f1_mean": float(np.mean([fr["metrics"]["f1"] for fr in fold_results])),
        "cv/roc_auc_mean": float(np.mean([fr["metrics"]["roc_auc"] for fr in fold_results])),
        "cv/proc_auc_mean": float(np.mean([fr["metrics"]["proc_auc"] for fr in fold_results])),
        "cv/pr_auc_mean": float(np.mean([fr["metrics"]["pr_auc"] for fr in fold_results])),
    })

    for branch_name, agg in sorted(branch_aggregates.items()):
        wandb.log({
            f"threshold_strategies/{branch_name}/final/accuracy": agg["final"]["accuracy"],
            f"threshold_strategies/{branch_name}/final/specificity": agg["final"]["specificity"],
            f"threshold_strategies/{branch_name}/final/recall": agg["final"]["recall"],
            f"threshold_strategies/{branch_name}/final/precision": agg["final"]["precision"],
            f"threshold_strategies/{branch_name}/final/f1": agg["final"]["f1"],
            f"threshold_strategies/{branch_name}/final/roc_auc": agg["final"]["roc_auc"],
            f"threshold_strategies/{branch_name}/final/proc_auc": agg["final"]["proc_auc"],
            f"threshold_strategies/{branch_name}/final/pr_auc": agg["final"]["pr_auc"],
            f"threshold_strategies/{branch_name}/final/tp": agg["final"]["tp"],
            f"threshold_strategies/{branch_name}/final/fp": agg["final"]["fp"],
            f"threshold_strategies/{branch_name}/final/fn": agg["final"]["fn"],
            f"threshold_strategies/{branch_name}/final/tn": agg["final"]["tn"],
            f"threshold_strategies/{branch_name}/cv_mean_curve_auc/roc_auc": agg["cv_mean_curve_auc"]["roc_auc"],
            f"threshold_strategies/{branch_name}/cv_mean_curve_auc/proc_auc": agg["cv_mean_curve_auc"]["proc_auc"],
            f"threshold_strategies/{branch_name}/cv_mean_curve_auc/pr_auc": agg["cv_mean_curve_auc"]["pr_auc"],
        })

        for m in metric_keys:
            wandb.log({
                f"threshold_strategies/{branch_name}/cv/{m}_mean": agg["cv_mean_std"][m]["mean"],
            })

    artifact = wandb.Artifact(
        name=f"{args.experiment}_final_eval_outputs",
        type="evaluation",
        description="Final CV fold evaluations + aggregate outputs",
    )

    for fr in fold_results:
        fold_eval_dir = os.path.join(fr["fold_dir"], "final_evaluation")
        if os.path.isdir(fold_eval_dir):
            artifact.add_dir(fold_eval_dir, name=os.path.join(f"fold_{fr['fold']}", "final_evaluation"))

        log_candidates = []
        for root, _, files in os.walk(fr["fold_dir"]):
            for file in files:
                low = file.lower()
                if low.endswith(".log") or low == "logs.txt" or "log" in low:
                    log_candidates.append(os.path.join(root, file))
        for p in sorted(set(log_candidates)):
            rel_name = os.path.join(f"fold_{fr['fold']}", "logs", os.path.relpath(p, fr["fold_dir"]))
            artifact.add_file(p, name=rel_name)

    artifact.add_dir(final_agg_dir, name="FINAL_AGG")
    wandb.log_artifact(artifact)

    log_line("Aggregated metrics summary:", log_path)
    log_line(
        f"ROC_AUC={overall_curves['roc_auc']:.4f}, pROC_AUC={overall_curves['proc_auc']:.4f}, PR_AUC={overall_curves['pr_auc']:.4f}",
        log_path,
    )
    log_line(
        f"ACC={overall_cm['accuracy']:.4f}, SPEC={overall_cm['specificity']:.4f}, REC={overall_cm['recall']:.4f}, PREC={overall_cm['precision']:.4f}, F1={overall_cm['f1']:.4f}",
        log_path,
    )

    recap_metric_map = {
        "ROC_AUC": "roc_auc",
        "PR_AUC": "pr_auc",
        "ACCURACY": "accuracy",
        "F1": "f1",
        "SENSITIVITY": "recall",
        "SPECIFICITY": "specificity",
        "PPV": "precision",
    }
    recap_branches = [
        "optimized/pre_retrain/best",
        "optimized/post_retrain",
    ]

    log_line("CV recap over folds (mean ± std) for optimized branches:", log_path)
    for branch_name in recap_branches:
        agg = branch_aggregates.get(branch_name)
        if agg is None:
            log_line(f"{branch_name}: not available", log_path)
            continue

        log_line(f"{branch_name}:", log_path)
        for display_name, metric_key in recap_metric_map.items():
            mean_val = float(agg["cv_mean_std"][metric_key]["mean"])
            std_val = float(agg["cv_mean_std"][metric_key]["std"])
            log_line(f"  {display_name}: {mean_val:.4f} ± {std_val:.4f}", log_path)

    wandb.finish()
    log_line("Final evaluation complete.", log_path)


if __name__ == "__main__":
    main()
