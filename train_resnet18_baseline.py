import os
import argparse
import numpy as np
import random
import torch
import wandb
import json
import time
from datetime import timedelta

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from monai.networks.nets import resnet18
from torchvision.models.video import r3d_18

from data_generator import get_patch_dataloaders
from training_evaluation_functions import standard_train_classifier, evaluate_classifier, collect_labels_probs, compute_paper_metrics


def parse_arguments():
    parser = argparse.ArgumentParser(description="MRI multi-modal training script (ResNet18/R3D-18 baseline)")
    parser.add_argument("-dt", "--dataset_type", type=str, choices=['HSMn'], default='HSMn', help="Type of dataset to use")
    parser.add_argument("-m", "--mask", type=str, default="lesion", help="Mask type")
    parser.add_argument("-ep", "--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("-drop", "--dropout", type=float, default=0.3, help="Unused for resnet18 baseline")
    parser.add_argument("-mod", "--modalities", type=str, default='FLAIR,QSM', help="Modalities to use (comma-separated)")
    parser.add_argument("-f", "--filters", type=int, default=16, help="Unused for resnet18 baseline")
    parser.add_argument("-exp", "--experiment", type=str, default="0", help="Name of the experiment")
    parser.add_argument("-batch", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-g", "--gpu", type=int, required=False, help="GPU to use")
    parser.add_argument("-patience", "--patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("-seed", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-img_size", "--img_size", type=tuple, default=(256, 256), help="Image size")
    parser.add_argument("-fold", "--fold", type=int, default=0, help="Fold for leave-one-out")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("-th", "--threshold", type=float, default=0.5, help="Threshold for classification")
    parser.add_argument("-balance", "--balance_classes", action='store_true', help="Balance dataset by sampling equal number of RIM and non-RIM patches")
    parser.add_argument("--use_pos_weight", action='store_true', help="Use pos_weight in BCEWithLogitsLoss")
    parser.add_argument("--weighted_sampler", action='store_true', help="Use WeightedRandomSampler for balanced batches")
    parser.add_argument("--use_ema", action='store_true', help="Use Exponential Moving Average of model weights")
    parser.add_argument("--cv5", action='store_true', help="Run 5-fold outer cross-validation")
    parser.add_argument("--cv_fold", type=int, default=None, help="Run a specific outer fold (0-4)")
    parser.add_argument("--inner_val_ratio", type=float, default=0.2, help="Inner validation ratio for dev split")
    parser.add_argument("--threshold_search", action='store_true', default=True, help="Search best threshold on val")
    parser.add_argument("--no_threshold_search", action='store_false', dest='threshold_search', help="Disable threshold search and use --threshold")
    parser.add_argument("--threshold_metric", type=str, choices=['f1', 'youden', 'pr_auc'], default='f1', help="Metric for threshold selection")
    parser.add_argument("--model", type=str, choices=['resnet18', 'r3d_18'], default='resnet18', help="Backbone to train")
    args = parser.parse_args()

    if isinstance(args.modalities, str):
        args.modalities = [m.strip() for m in args.modalities.split(',')]

    return args


def get_dataset_info(dataset_type):
    if dataset_type == 'HSMn':
        return "/data/cil/veronica/HSMn"
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_labels(patches_path):
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


class ResNet18Baseline(nn.Module):
    def __init__(self, in_channels, model_name='resnet18'):
        super().__init__()
        self.model_name = model_name
        if model_name == 'resnet18':
            self.backbone = resnet18(
                spatial_dims=3,
                n_input_channels=in_channels,
                num_classes=1
            )
        elif model_name == 'r3d_18':
            self.backbone = r3d_18(weights=None, progress=True, num_classes=1)
            stem_conv = self.backbone.stem[0]
            if stem_conv.in_channels != in_channels:
                self.backbone.stem[0] = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=stem_conv.out_channels,
                    kernel_size=stem_conv.kernel_size,
                    stride=stem_conv.stride,
                    padding=stem_conv.padding,
                    bias=(stem_conv.bias is not None)
                )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def forward(self, volumes):
        x = torch.cat(volumes, dim=1)
        return self.backbone(x)


def main():
    args = parse_arguments()
    setup_seed(args.seed)

    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda:0")

    print(f"Using device: {device}")
    print(f"Experiment '{args.experiment}' starting with model: {args.model}")

    _ = get_dataset_info(args.dataset_type)
    patches_path = "/data/cil/veronica/3DRim_Classification_patches"
    experiments_path = "/data/cil/veronica/Experiments/SM_experiments/3D_Rim_Classification"

    labels_path = os.path.join(patches_path, 'labels.json')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found in {patches_path}.")

    labels_dict = load_labels(patches_path)
    rim_count, norim_count = 0, 0
    for phase in ['train', 'val', 'test']:
        if phase not in labels_dict:
            continue
        for subj in labels_dict[phase]:
            for _, label in labels_dict[phase][subj].items():
                if label == 1:
                    rim_count += 1
                else:
                    norim_count += 1

    total_samples = rim_count + norim_count
    pos_weight = (norim_count / rim_count) if rim_count > 0 else 1.0

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
    summary_output_dir = os.path.join(experiments_path, f"{args.experiment}_{args.model}")
    os.makedirs(summary_output_dir, exist_ok=True)

    for fold_id in fold_ids:
        print("\n" + "=" * 60)
        print(f"OUTER FOLD {fold_id}")
        print("=" * 60)

        is_single_fold = (len(fold_ids) == 1)
        if is_single_fold:
            wandb.init(
                project="Classification-w-QSM",
                entity="RIM-project",
                config=vars(args),
                name=f"{args.experiment}_{args.model}",
                job_type="training"
            )
        else:
            fold_config = vars(args).copy()
            fold_config["fold"] = fold_id
            wandb.init(
                project="Classification-w-QSM",
                entity="RIM-project",
                config=fold_config,
                name=f"{args.experiment}_{args.model}_fold{fold_id}",
                group=f"{args.experiment}_{args.model}",
                job_type="training",
                reinit=True
            )

        test_subjects = folds[fold_id]
        dev_subjects = []
        for k, fold_subjects in folds.items():
            if k != fold_id:
                dev_subjects.extend(fold_subjects)

        train_subjects, val_subjects = split_dev_subjects(
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

        wandb.log({
            "n_train_patients": len(train_subjects),
            "n_val_patients": len(val_subjects),
            "n_test_patients": len(test_subjects),
            "n_train_patches": num_train,
            "n_val_patches": num_val,
            "n_test_patches": num_test
        })

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

        model = ResNet18Baseline(in_channels=len(args.modalities), model_name=args.model).to(device)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
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
            use_supcon=False,
            supcon_tau=0.1,
            supcon_weight_param=None
        )

        model_path = os.path.join(fold_output_dir, 'model_best.pt')
        torch.save(best_model.state_dict(), model_path)
        if ema is not None:
            torch.save(ema.ema.state_dict(), os.path.join(fold_output_dir, 'model_best_ema.pt'))

        elapsed = time.time() - start_time
        print(f"Training time: {str(timedelta(seconds=int(elapsed)))}")

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
            json.dump({'best_epoch': best_epoch, 'best_threshold': best_threshold}, f, indent=2)

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

        final_model = ResNet18Baseline(in_channels=len(args.modalities), model_name=args.model).to(device)
        optimizer = AdamW(final_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
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
            use_supcon=False,
            supcon_tau=0.1,
            supcon_weight_param=None
        )

        test_metrics, test_details = evaluate_classifier(
            model=final_model,
            test_loader=test_loader,
            device=device,
            output_dir=os.path.join(fold_output_dir, 'evaluation'),
            threshold=best_threshold,
            ema=final_ema if args.use_ema else None,
            return_details=True
        )

        with open(os.path.join(fold_output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)

        all_folds_results.append({
            'y_true': test_details['y_true'],
            'y_prob': test_details['y_prob'],
            'lesion_uids': test_details['lesion_uids'],
            'val_y_true': val_labels_arr.tolist(),
            'val_y_prob': val_probs_arr.tolist()
        })

        all_fold_metrics.append(test_metrics)
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()

    if args.cv5:
        metric_keys = ['average_precision', 'auc', 'f1', 'precision', 'recall', 'accuracy']
        summary = {'mean': {}, 'std': {}, 'var': {}}
        for key in metric_keys:
            values = [m.get(key, 0.0) for m in all_fold_metrics]
            summary['mean'][key] = float(np.mean(values))
            summary['std'][key] = float(np.std(values))
            summary['var'][key] = float(np.var(values))

        with open(os.path.join(summary_output_dir, 'cv_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    paper_metrics = compute_paper_metrics(all_folds_results, bootstrap_iters=500, seed=args.seed)
    with open(os.path.join(summary_output_dir, 'paper_metrics_summary.json'), 'w') as f:
        json.dump(paper_metrics, f, indent=2)

    try:
        wandb.init(
            project="Classification-w-QSM",
            entity="RIM-project",
            config=vars(args),
            name="Evaluation_Final_Report",
            group=f"{args.experiment}_{args.model}",
            job_type="evaluation",
            reinit=True
        )

        for fold_log in paper_metrics['fold_logs']:
            fold_step = int(fold_log['fold'])
            wandb.log({
                "paper-METRICS/fold/tau_opt": fold_log['tau_opt'],
                "paper-METRICS/fold/val_f1_opt": fold_log['val_f1_opt'],
                "paper-METRICS/fold/auc": fold_log['auc'],
                "paper-METRICS/fold/pauc_0_1_std": fold_log['pauc_0_1_std']
            }, step=fold_step)

        final_step = len(paper_metrics['fold_logs'])
        summary_log = {
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
        }

        if args.cv5:
            metric_keys = ['average_precision', 'auc', 'f1', 'precision', 'recall', 'accuracy']
            summary = {'mean': {}, 'std': {}, 'var': {}}
            for key in metric_keys:
                values = [m.get(key, 0.0) for m in all_fold_metrics]
                summary['mean'][key] = float(np.mean(values))
                summary['std'][key] = float(np.std(values))
                summary['var'][key] = float(np.var(values))
                summary_log[f"paper-METRICS/summary/cv_mean/{key}"] = summary['mean'][key]
                summary_log[f"paper-METRICS/summary/cv_std/{key}"] = summary['std'][key]
                summary_log[f"paper-METRICS/summary/cv_var/{key}"] = summary['var'][key]

        wandb.log(summary_log, step=final_step)
        wandb.finish()
    except Exception as e:
        print(f"Warning: Could not log final evaluation report to wandb: {e}")

    print("Pipeline complete!")


if __name__ == "__main__":
    main()