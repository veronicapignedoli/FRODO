import os
import argparse
import numpy as np
import random
import torch
import shutil
import wandb
import sklearn
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, StepLR, ReduceLROnPlateau, PolynomialLR
from pprint import pprint
from architecture import MultiModalClassifier
from data_generator import get_patch_dataloaders
from patches_extraction import compute_patches_extraction
import time
from datetime import datetime, timedelta
import json
from training_evaluation_functions import standard_train_classifier, evaluate_classifier, collect_labels_probs, compute_paper_metrics
from losses import focal_loss
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="MRI multi-modal KD training script")
    parser.add_argument("-dt", "--dataset_type", type=str, choices=['HSMn'], default='HSMn', help="Type of dataset to use")
    parser.add_argument("-m", "--mask", type=str, default="lesion", help="Mask type")
    parser.add_argument("-ep", "--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("-drop", "--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("-mod", "--modalities", type=str, default='FLAIR,QSM', help="Modalities to use (comma-separated)")
    parser.add_argument("-f", "--filters", type=int, default=16, help="Number of filters in first downsampling stage")
    parser.add_argument("-exp", "--experiment", type=str, default="0", help="Name of the experiment")
    parser.add_argument("-batch", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-patch_extr", "--extract_patches", action='store_true', help="Extract patches from sliced data")
    parser.add_argument("-g", "--gpu", type=int, required=False, help="GPU to use")
    parser.add_argument("-patience", "--patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("-seed", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-img_size", "--img_size", type=tuple, default=(256, 256), help="Image size")
    parser.add_argument("-fold", "--fold", type=int, default=0, help="Fold for leave-one-out (0-4 for ISBI)")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("-th", "--threshold", type=float, default=0.5, help="Threshold for classification")
    parser.add_argument("-balance", "--balance_classes", action='store_true', help="Balance dataset by sampling equal number of RIM and non-RIM patches")
    parser.add_argument("--use_pos_weight", action='store_true', help="Use pos_weight in BCEWithLogitsLoss")
    parser.add_argument("--weighted_sampler", action='store_true', help="Use WeightedRandomSampler for balanced batches")
    parser.add_argument("--use_ema", action='store_true', help="Use Exponential Moving Average of model weights")
    parser.add_argument("--use_supcon", action='store_true', help="Use Supervised Contrastive Loss as auxiliary loss")
    parser.add_argument("--supcon_tau", type=float, default=0.1, help="Temperature for SupCon loss")
    parser.add_argument("--cv5", action='store_true', help="Run 5-fold outer cross-validation")
    parser.add_argument("--cv_fold", type=int, default=None, help="Run a specific outer fold (0-4)")
    parser.add_argument("--inner_val_ratio", type=float, default=0.2, help="Inner validation ratio for dev split")
    parser.add_argument("--threshold_search", action='store_true', default=True, help="Search best threshold on val")
    parser.add_argument("--no_threshold_search", action='store_false', dest='threshold_search', help="Disable threshold search and use --threshold")
    parser.add_argument("--threshold_metric", type=str, choices=['f1', 'youden', 'pr_auc'], default='f1', help="Metric for threshold selection")
    parser.add_argument("--ssl_pretrained", action='store_true', help="Load self-supervised pretrained backbone")
    parser.add_argument("--ssl_pretrain_mode", type=str, choices=['contrastive', 'multimodal'], default='contrastive', help="Pretraining mode used to produce SSL backbone")
    parser.add_argument("--ssl_pretrain_root", type=str, default='pretrain_outputs', help="Root directory containing SSL pretraining runs")
    parser.add_argument("--ssl_pretrain_tag", type=str, default='', help="Optional substring to filter SSL run folder names")
    parser.add_argument("--ssl_pretrain_explicit_path", type=str, default='', help="Explicit path to SSL backbone .pt file (overrides root/mode/tag resolution)")
    args = parser.parse_args()
    
    if isinstance(args.modalities, str):
        args.modalities = [m.strip() for m in args.modalities.split(',')]
    
    return args 

def get_dataset_info(dataset_type):
    if dataset_type == 'HSMn':
        data_path = "/data/cil/veronica/HSMn"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return data_path

#set random seed for reproducibility 
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_subjects(data_path):
    all_subjects = [f for f in sorted(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, f))]
    return {
            'train': all_subjects[:62],
            'val': all_subjects[62:75],
            'test': all_subjects[75:]
            }


def load_labels(patches_path):
    """Load labels.json"""
    labels_path = os.path.join(patches_path, 'labels.json')
    with open(labels_path, 'r') as f:
        return json.load(f)


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


def split_dev_subjects(dev_subjects, labels_dict, subject_phase_map, seed, val_ratio, max_attempts=50):
    dev_subjects = list(dev_subjects)
    for attempt in range(max_attempts):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + attempt)
        train_idx, val_idx = next(gss.split(dev_subjects, groups=dev_subjects))
        train_subjects = [dev_subjects[i] for i in train_idx]
        val_subjects = [dev_subjects[i] for i in val_idx]
        if any(subject_has_positive(s, labels_dict, subject_phase_map) for s in val_subjects):
            return train_subjects, val_subjects
    raise ValueError("Unable to create a validation split with at least one positive subject.")


def count_patches(subjects, labels_dict, subject_phase_map=None, phase=None):
    """Count total patches for given subjects.
    New structure: labels_dict[phase][subj][patch_id] = label
    """
    count = 0
    if subject_phase_map is None:
        if phase not in labels_dict:
            return 0
        for subj in subjects:
            if subj not in labels_dict[phase]:
                continue
            count += len(labels_dict[phase][subj])
        return count
    
    for subj in subjects:
        subj_phase = subject_phase_map.get(subj)
        if subj_phase is None:
            continue
        count += len(labels_dict.get(subj_phase, {}).get(subj, {}))
    return count


def select_best_threshold(y_true, y_probs, metric='f1'):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_score = -1.0
    
    if metric == 'pr_auc':
        metric = 'f1'
    
    for thresh in thresholds:
        preds = (y_probs > thresh).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == 'youden':
            tn = np.sum((y_true == 0) & (preds == 0))
            fp = np.sum((y_true == 0) & (preds == 1))
            fn = np.sum((y_true == 1) & (preds == 0))
            tp = np.sum((y_true == 1) & (preds == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            score = tpr - fpr
        else:
            score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(thresh)
    
    return best_threshold


def build_subject_paths(subject_list, sliced_base, phase, dataset_type='HSMn', modalities=None):
    if dataset_type == 'HSMn':
        result = []
        for subj in subject_list:
            paths = {
                'mask_bin': os.path.join(sliced_base, subj, phase, 'out_mask_bin'),
                'mask_clust': os.path.join(sliced_base, subj, phase, 'out_mask_clust')
            }
            for mod in modalities:
                if mod == 'T1':
                    paths['T1'] = os.path.join(sliced_base, subj, phase, 'out_T1')
                elif mod == 'FLAIR':
                    paths['FLAIR'] = os.path.join(sliced_base, subj, phase, 'out_FLAIR')
                elif mod == 'FA':
                    paths['FA'] = os.path.join(sliced_base, subj, phase, 'out_FA')
                elif mod == 'QSM':
                    paths['QSM'] = os.path.join(sliced_base, subj, phase, 'out_QSM')
                elif mod == 'QSMp':
                    paths['QSMp'] = os.path.join(sliced_base, subj, phase, 'out_QSMp')
            result.append(paths)
        return result
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def resolve_ssl_checkpoint(fold, ssl_pretrain_mode, ssl_pretrain_root, ssl_pretrain_tag="", ssl_pretrain_explicit_path=""):
    if ssl_pretrain_explicit_path:
        explicit_path = os.path.abspath(ssl_pretrain_explicit_path)
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                f"SSL explicit checkpoint not found: {explicit_path}. "
                f"Expected a valid .pt checkpoint path"
            )
        return explicit_path

    root = os.path.abspath(ssl_pretrain_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"SSL pretrain root not found: {root}. "
            f"Expected directories like <timestamp>_{ssl_pretrain_mode}_fold{fold} containing SSL backbone .pt files"
        )

    suffix = f"_{ssl_pretrain_mode}_fold{fold}"
    candidates = []
    for dirname in os.listdir(root):
        run_dir = os.path.join(root, dirname)
        if not os.path.isdir(run_dir):
            continue
        if not dirname.endswith(suffix):
            continue
        if ssl_pretrain_tag and ssl_pretrain_tag not in dirname:
            continue
        ckpt_path_old = os.path.join(run_dir, "backbone_pretrained.pt")
        if os.path.isfile(ckpt_path_old):
            candidates.append((dirname, ckpt_path_old))
            continue

        run_ckpt_candidates = []
        for filename in os.listdir(run_dir):
            if not filename.endswith(".pt"):
                continue
            if filename in {"checkpoint_last.pt", "checkpoint_best.pt"}:
                continue
            if not filename.endswith(f"{suffix}.pt"):
                continue
            run_ckpt_candidates.append(filename)

        if run_ckpt_candidates:
            run_ckpt_candidates.sort()
            ckpt_path_new = os.path.join(run_dir, run_ckpt_candidates[-1])
            candidates.append((dirname, ckpt_path_new))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No SSL checkpoint found for fold {fold}, mode '{ssl_pretrain_mode}', root '{root}', tag '{ssl_pretrain_tag}'. "
            f"Expected at least one directory matching '*_{ssl_pretrain_mode}_fold{fold}' containing either backbone_pretrained.pt or '*_{ssl_pretrain_mode}_fold{fold}.pt'"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_ssl_backbone(model, checkpoint_path, fold_id, context_label):
    ssl_state = torch.load(checkpoint_path, map_location='cpu')
    incompatible = model.load_state_dict(ssl_state, strict=False)
    missing_keys = incompatible.missing_keys
    unexpected_keys = incompatible.unexpected_keys
    print(f"[SSL] Loaded pretrained backbone ({context_label}) for fold {fold_id}: {checkpoint_path}")
    print(f"[SSL] missing_keys={len(missing_keys)} | unexpected_keys={len(unexpected_keys)}")
    try:
        wandb.log({
            f"ssl/{context_label}_enabled": 1,
            f"ssl/{context_label}_fold": int(fold_id),
            f"ssl/{context_label}_path": checkpoint_path,
            f"ssl/{context_label}_missing_keys_count": len(missing_keys),
            f"ssl/{context_label}_unexpected_keys_count": len(unexpected_keys),
        })
    except Exception as e:
        print(f"Warning: Could not log SSL load info to wandb: {e}")
    return missing_keys, unexpected_keys


def ssl_forward_sanity_check(model, loader, device, context_label):
    with torch.no_grad():
        volumes, _labels, _infos = next(iter(loader))
        volumes = [v.to(device) for v in volumes]
        logits = model(volumes)
        print(f"[SSL] Sanity check ({context_label}) logits shape: {tuple(logits.shape)}")

def main():
    # ========== SETTING ==========
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    args = parse_arguments()
    setup_seed(args.seed)
    modalities = args.modalities
    print(f"\n Experiment '{args.experiment}' starting!")
    pprint(vars(args))
    if args.balance_classes and args.weighted_sampler:
        print("[WARNING] Both --balance_classes and --weighted_sampler are enabled. This can be redundant.")
    if args.use_pos_weight and args.weighted_sampler:
        print("[WARNING] Both --use_pos_weight and --weighted_sampler are enabled. This can over-correct class imbalance.")
    base_path = get_dataset_info(args.dataset_type)
    device = torch.device(f"cuda:0")
    print(f"Using device: {device}")
    patches_path = "/data/cil/veronica/3DRim_Classification_patches"
    experiments_path = "/data/cil/veronica/Experiments/SM_experiments/3D_Rim_Classification"

    # ========== PATCH EXTRACTION ==========
    if args.extract_patches:
        print("\n" + "="*60)
        print("EXTRACTING PATCHES")
        print("="*60)
        compute_patches_extraction()
        print(" Patch extraction complete!")

    # ========== LOADING LABELS ==========
    if not os.path.exists(os.path.join(patches_path, 'labels.json')):
        raise FileNotFoundError(
            f" labels.json not found in {patches_path}.\n"
            f"Run with --extract_patches to generate it:\n"
            f"  python main.py --extract_patches"
        )
    labels_dict = load_labels(patches_path)
    rim_count = 0
    norim_count = 0
    for phase in ['train', 'val', 'test']:
        if phase not in labels_dict:
            continue
        for subj in labels_dict[phase]:
            for patch_id, label in labels_dict[phase][subj].items():
                if label == 1:
                    rim_count += 1
                else:
                    norim_count += 1

    print(f" Labels in labels.json:")
    print(f"  Rim (1): {rim_count}")
    print(f"  NoRim (0): {norim_count}")
    
    # ========== POS WEIGHT-IF ENABLED ==========
    total_samples = rim_count + norim_count
    pos_weight = (norim_count / rim_count) if rim_count > 0 else 1.0
    
    if args.use_pos_weight:
        print(f"Class Weighting for BCEWithLogitsLoss:")
        print(f"  Total samples: {total_samples}")
        print(f"  Rim samples: {rim_count} ({100*rim_count/total_samples:.1f}%)")
        print(f"  NoRim samples: {norim_count} ({100*norim_count/total_samples:.1f}%)")
        print(f"  pos_weight: {pos_weight:.2f}x")
        print(f"  Model will care {pos_weight:.0f}x more about Rim errors.")
        
    # ========== LOADING DATASET AND BUILDING DATALOADERS ==========
    if not os.path.exists(os.path.join(patches_path, 'labels.json')):
        raise FileNotFoundError(f"labels.json not found in {patches_path}. Run with --extract_patches first.")

    subject_phase_map = get_subject_phase_map(labels_dict)
    folds = get_outer_folds()
    
    if args.cv5:
        fold_ids = [0, 1, 2, 3, 4]
    elif args.cv_fold is not None:
        if args.cv_fold not in folds:
            raise ValueError("cv_fold must be in [0, 1, 2, 3, 4]")
        fold_ids = [args.cv_fold]
    else:
        fold_ids = [0]
    
    all_fold_metrics = []
    all_folds_results = []
    summary_output_dir = os.path.join(experiments_path, args.experiment)
    os.makedirs(summary_output_dir, exist_ok=True)

    for fold_id in fold_ids:
        print("\n" + "="*60)
        print(f"OUTER FOLD {fold_id}")
        print("="*60)
        
        # --- Initialize wandb run for this fold ---
        is_single_fold = (len(fold_ids) == 1)
        if is_single_fold:
            try:
                wandb.init(
                    project="Classification-w-QSM",
                    entity="RIM-project",
                    config=vars(args),
                    name=args.experiment,
                    job_type="training"
                )
                print("W&B initialized successfully.")
            except Exception as e:
                print(f"W&B initialization failed: {e}")
        else:
            fold_config = vars(args).copy()
            fold_config["fold"] = fold_id
            try:
                wandb.init(
                    project="Classification-w-QSM",
                    entity="RIM-project",
                    config=fold_config,
                    name=f"{args.experiment}_fold{fold_id}",
                    group=args.experiment,
                    job_type="training",
                    reinit=True
                )
                print(f"W&B initialized for fold {fold_id}.")
            except Exception as e:
                print(f"W&B initialization failed for fold {fold_id}: {e}")

        test_subjects = folds[fold_id]
        dev_subjects = []
        for k, fold_subjects in folds.items():
            if k != fold_id:
                dev_subjects.extend(fold_subjects)

        train_subjects, val_subjects = split_dev_subjects( #constraint: the validation split must contain at least one positive subject
            dev_subjects=dev_subjects,
            labels_dict=labels_dict,
            subject_phase_map=subject_phase_map,
            seed=args.seed,
            val_ratio=args.inner_val_ratio
        )

        subjects = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }

        fold_output_dir = os.path.join(summary_output_dir, f"fold_{fold_id}")
        os.makedirs(fold_output_dir, exist_ok=True)

        with open(os.path.join(fold_output_dir, 'split_subjects.json'), 'w') as f:
            json.dump(subjects, f, indent=2)

        num_train = count_patches(subjects['train'], labels_dict, subject_phase_map=subject_phase_map)
        num_val = count_patches(subjects['val'], labels_dict, subject_phase_map=subject_phase_map)
        num_test = count_patches(subjects['test'], labels_dict, subject_phase_map=subject_phase_map)
        
        try:
            wandb.log({
                "n_train_patients": len(train_subjects),
                "n_val_patients": len(val_subjects),
                "n_test_patients": len(test_subjects),
                "n_train_patches": num_train,
                "n_val_patches": num_val,
                "n_test_patches": num_test
            })
        except Exception as e:
            print(f"Warning: Could not log split sizes to wandb: {e}")

        print(f" Dataset statistics:")
        print(f"  Training patches: {num_train}")
        print(f"  Validation patches: {num_val}")
        print(f"  Test patches: {num_test}")

        train_loader, val_loader, test_loader = get_patch_dataloaders(
            patches_path=patches_path,
            labels_dict=labels_dict,
            subjects=subjects,
            modalities=args.modalities,
            batch_size=args.batch_size,
            num_workers=8,
            balance_classes=args.balance_classes,
            weighted_sampler=args.weighted_sampler,
            subject_phase_map=subject_phase_map
        )

        print(f"Steps per epoch (train): {len(train_loader)}")
        print(f"Steps per epoch (val): {len(val_loader)}")
        print(f"Steps per epoch (test): {len(test_loader)}")

        print("DATASET")
        print(f"Train: {len(train_loader.dataset)} patches")
        print(f"Val: {len(val_loader.dataset)} patches")
        print(f"Test: {len(test_loader.dataset)} patches")

        train_labels = []
        val_labels = []

        for _, labels, _ in train_loader:
            train_labels.extend(labels.numpy().flatten())

        for _, labels, _ in val_loader:
            val_labels.extend(labels.numpy().flatten())

        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)

        print(f"Train labels distribution:")
        print(f"  Rim (1): {np.sum(train_labels == 1)} ({100*np.mean(train_labels == 1):.1f}%)")
        print(f"  NoRim (0): {np.sum(train_labels == 0)} ({100*np.mean(train_labels == 0):.1f}%)")

        print(f"\nVal labels distribution:")
        print(f"  Rim (1): {np.sum(val_labels == 1)} ({100*np.mean(val_labels == 1):.1f}%)")
        print(f"  NoRim (0): {np.sum(val_labels == 0)} ({100*np.mean(val_labels == 0):.1f}%)")

        print(f"\nDataset overlap check:")
        print(f"  Train samples: {len(train_loader.dataset.samples)}")
        print(f"  Val samples: {len(val_loader.dataset.samples)}")

        print("\n" + "="*60)
        print("CREATING MODEL")
        print("="*60)
        
        model = MultiModalClassifier(
            in_channels_list=[1] * len(args.modalities),
            base_filters=args.filters,
            dropout=0.3
        ).to(device)
        if args.ssl_pretrained:
            ssl_checkpoint_path = resolve_ssl_checkpoint(
                fold=fold_id,
                ssl_pretrain_mode=args.ssl_pretrain_mode,
                ssl_pretrain_root=args.ssl_pretrain_root,
                ssl_pretrain_tag=args.ssl_pretrain_tag,
                ssl_pretrain_explicit_path=args.ssl_pretrain_explicit_path,
            )
            print(f"[SSL] Fold-aware load: supervised fold {fold_id} -> pretrained fold {fold_id}")
            load_ssl_backbone(model, ssl_checkpoint_path, fold_id, context_label="train")
            ssl_forward_sanity_check(model, train_loader, device, context_label="train")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)

        # Create SupCon weight parameter if enabled
        supcon_weight_param = None
        if args.use_supcon:
            # Initialize logit such that softplus(logit) ≈ 0.05-0.1
            # softplus(x) ≈ log(1 + exp(x)), so for small outputs: softplus(x) ≈ log(2) + x/2 + ...
            # To get 0.05, we want x such that softplus(x) ≈ 0.05
            # For small outputs: log(1 + exp(x)) ≈ log(1 + exp(0)) = log(2) ≈ 0.693 when x=0
            # Use x ≈ -2.5 to get softplus(-2.5) ≈ 0.0818
            supcon_logit_init = torch.tensor(-2.5, dtype=torch.float32, device=device, requires_grad=True)
            supcon_weight_param = supcon_logit_init
            print(f"SupCon initialized: softplus({supcon_logit_init.item():.4f}) ≈ {torch.nn.functional.softplus(supcon_weight_param).item():.4f}")

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
        
        # Add SupCon weight to optimizer if enabled
        if supcon_weight_param is not None:
            optimizer.add_param_group({'params': [supcon_weight_param]})
        
        scheduler = StepLR(optimizer, step_size=10, gamma=0.75)
        if args.use_pos_weight:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()

        start_time = time.time()
        best_model, ema, best_epoch = standard_train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=args.epochs,
            log_dir=fold_output_dir,
            scheduler=scheduler,
            patience=args.patience,
            use_ema=args.use_ema,
            use_supcon=args.use_supcon,
            supcon_tau=args.supcon_tau,
            supcon_weight_param=supcon_weight_param
        )

        model_path = os.path.join(fold_output_dir, 'model_best.pt')
        torch.save(best_model.state_dict(), model_path)
        if ema is not None:
            torch.save(ema.ema.state_dict(), os.path.join(fold_output_dir, 'model_best_ema.pt'))
        print(f"Model saved to {model_path}")

        end_time = time.time()
        elapsed = end_time - start_time
        print(f" Training time: {str(timedelta(seconds=int(elapsed)))}")

        val_labels_arr, val_probs_arr = collect_labels_probs(
            model=best_model,
            data_loader=val_loader,
            device=device,
            ema=ema if args.use_ema else None
        )

        best_threshold = args.threshold
        if args.threshold_search:
            best_threshold = select_best_threshold(val_labels_arr, val_probs_arr, metric=args.threshold_metric)

        with open(os.path.join(fold_output_dir, 'fold_metadata.json'), 'w') as f:
            json.dump({
                'best_epoch': best_epoch,
                'best_threshold': best_threshold
            }, f, indent=2)

        print("\n" + "="*60)
        print("RETRAINING FINAL MODEL")
        print("="*60)

        train_dev_subjects = train_subjects + val_subjects
        retrain_subjects = {'train': train_dev_subjects, 'val': [], 'test': []}
        retrain_loader, _, _ = get_patch_dataloaders(
            patches_path=patches_path,
            labels_dict=labels_dict,
            subjects=retrain_subjects,
            modalities=args.modalities,
            batch_size=args.batch_size,
            num_workers=8,
            balance_classes=args.balance_classes,
            weighted_sampler=args.weighted_sampler,
            subject_phase_map=subject_phase_map
        )

        final_model = MultiModalClassifier(
            in_channels_list=[1] * len(args.modalities),
            base_filters=args.filters,
            dropout=0.3
        ).to(device)
        if args.ssl_pretrained:
            print(f"[SSL] Fold-aware load (retrain): supervised fold {fold_id} -> pretrained fold {fold_id}")
            load_ssl_backbone(final_model, ssl_checkpoint_path, fold_id, context_label="retrain")
            ssl_forward_sanity_check(final_model, retrain_loader, device, context_label="retrain")
        optimizer = AdamW(final_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        
        # Create SupCon weight parameter for final model if enabled
        final_supcon_weight_param = None
        if args.use_supcon:
            final_supcon_logit_init = torch.tensor(-2.5, dtype=torch.float32, device=device, requires_grad=True)
            final_supcon_weight_param = final_supcon_logit_init
            optimizer.add_param_group({'params': [final_supcon_weight_param]})
        
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
        if args.use_pos_weight:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()

        final_model, final_ema, _ = standard_train_classifier(
            model=final_model,
            train_loader=retrain_loader,
            val_loader=None,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=best_epoch,
            log_dir=None,
            scheduler=scheduler,
            patience=0,
            use_ema=args.use_ema,
            use_supcon=args.use_supcon,
            supcon_tau=args.supcon_tau,
            supcon_weight_param=final_supcon_weight_param
        )
        torch.save(final_model.state_dict(), os.path.join(fold_output_dir, "model_post_retrain.pt"))
        if args.use_ema and final_ema is not None:
            torch.save(final_ema.ema.state_dict(), os.path.join(fold_output_dir, "model_post_retrain_ema.pt"))
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)

        test_metrics, test_details = evaluate_classifier(
            model=final_model,
            test_loader=test_loader,
            device=device,
            output_dir=os.path.join(fold_output_dir, 'evaluation'),
            threshold=best_threshold,
            ema=final_ema if args.use_ema else None,
            return_details=True
        )

        all_folds_results.append({
            'y_true': test_details['y_true'],
            'y_prob': test_details['y_prob'],
            'lesion_uids': test_details['lesion_uids'],
            'val_y_true': val_labels_arr.tolist(),
            'val_y_prob': val_probs_arr.tolist()
        })

        with open(os.path.join(fold_output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)

        all_fold_metrics.append(test_metrics)
        
        try:
            test_metrics_log = {f"test/{k}": v for k, v in test_metrics.items()}
            wandb.log(test_metrics_log)
        except Exception as e:
            print(f"Warning: Could not log test metrics to wandb: {e}")
        
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: Could not finish wandb run: {e}")

        print("\n Test Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        print(f"  AUC-ROC: {test_metrics['auc']:.4f}")

    if args.cv5:
        metric_keys = ['average_precision', 'auc', 'f1', 'precision', 'recall', 'accuracy']
        summary = {'mean': {}, 'std': {}}
        for key in metric_keys:
            values = [m.get(key, 0.0) for m in all_fold_metrics]
            summary['mean'][key] = float(np.mean(values))
            summary['std'][key] = float(np.std(values))

        with open(os.path.join(summary_output_dir, 'cv_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        paper_metrics = compute_paper_metrics(all_folds_results, bootstrap_iters=500, seed=args.seed)
        with open(os.path.join(summary_output_dir, 'paper_metrics_summary.json'), 'w') as f:
            json.dump(paper_metrics, f, indent=2)

        print("\nCross-validation summary (mean +/- std):")
        for key in metric_keys:
            mean_v = summary['mean'][key]
            std_v = summary['std'][key]
            print(f"  {key}: {mean_v:.4f} +/- {std_v:.4f}")
        
        try:
            wandb.init(
                project="Classification-w-QSM",
                entity="RIM-project",
                config=vars(args),
                name="Evaluation_Final_Report",
                group=args.experiment,
                job_type="evaluation",
                reinit=True
            )
            print("W&B initialized for final evaluation report.")

            for fold_log in paper_metrics['fold_logs']:
                fold_step = int(fold_log['fold'])
                wandb.log({
                    "paper-METRICS/fold/tau_opt": fold_log['tau_opt'],
                    "paper-METRICS/fold/val_f1_opt": fold_log['val_f1_opt'],
                    "paper-METRICS/fold/auc": fold_log['auc'],
                    "paper-METRICS/fold/pauc_0_1_std": fold_log['pauc_0_1_std']
                }, step=fold_step)

            final_step = len(paper_metrics['fold_logs'])
            wandb.log({
                "paper-METRICS/summary/auc_mean": paper_metrics['auc_mean'],
                "paper-METRICS/summary/auc_std": paper_metrics['auc_std'],
                "paper-METRICS/summary/pauc_0_1_mean": paper_metrics['pauc_0_1_mean'],
                "paper-METRICS/summary/pauc_0_1_std": paper_metrics['pauc_0_1_std'],
                "paper-METRICS/summary/f1": paper_metrics['global_metrics']['f1'],
                "paper-METRICS/summary/accuracy": paper_metrics['global_metrics']['accuracy'],
                "paper-METRICS/summary/sensitivity": paper_metrics['global_metrics']['sensitivity'],
                "paper-METRICS/summary/specificity": paper_metrics['global_metrics']['specificity'],
                "paper-METRICS/summary/ppv": paper_metrics['global_metrics']['ppv'],
                "paper-METRICS/summary/ci95_accuracy_low": paper_metrics['bootstrap_ci_95']['accuracy'][0],
                "paper-METRICS/summary/ci95_accuracy_high": paper_metrics['bootstrap_ci_95']['accuracy'][1],
                "paper-METRICS/summary/ci95_specificity_low": paper_metrics['bootstrap_ci_95']['specificity'][0],
                "paper-METRICS/summary/ci95_specificity_high": paper_metrics['bootstrap_ci_95']['specificity'][1],
                "paper-METRICS/summary/ci95_ppv_low": paper_metrics['bootstrap_ci_95']['ppv'][0],
                "paper-METRICS/summary/ci95_ppv_high": paper_metrics['bootstrap_ci_95']['ppv'][1],
                "paper-METRICS/summary/patient_pearson_r": paper_metrics['patient_level']['pearson_r'],
                "paper-METRICS/summary/patient_mse": paper_metrics['patient_level']['mse']
            }, step=final_step)

            wandb.finish()
            print("Final evaluation report logged to wandb.")
        except Exception as e:
            print(f"Warning: Could not log final evaluation report to wandb: {e}")

    print("\n Pipeline complete!")

if __name__ == "__main__":
    main()