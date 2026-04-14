import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import copy
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, precision_recall_curve, average_precision_score)

def standard_train_classifier(model, train_loader, val_loader, optimizer, criterion, device, epochs=100, log_dir=None, scheduler=None, patience=10, use_ema=False, use_supcon=False, supcon_tau=0.1, supcon_weight_param=None):
    """
    Training loop with optional EMA and SupCon support.
    
    Args:
        use_supcon: bool - Enable supervised contrastive loss
        supcon_tau: float - Temperature for SupCon (default 0.1)
        supcon_weight_param: nn.Parameter - Learnable weight (softplus of underlying logit)
    """
    from losses import supervised_contrastive_loss
    
    best_model = None
    best_ema = None
    patience_counter = 0
    
    # Initialize EMA if enabled
    ema = None
    if use_ema:
        from ema import ModelEMA
        ema = ModelEMA(model, decay=0.999)
        print("EMA enabled with decay=0.999")
    
    if use_supcon:
        print(f"SupCon enabled with tau={supcon_tau}")
        if supcon_weight_param is not None:
            print(f"SupCon weight initialized with logit={supcon_weight_param.item():.4f}")
    
    train_losses, val_losses = [], []
    best_val_pr_auc = -1.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # ========== TRAINING ==========
        model.train()
        epoch_train_loss = 0.0
        epoch_loss_cls = 0.0
        epoch_loss_supcon = 0.0
        epoch_grad_norms = []
        all_labels, all_probs = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for volumes, labels, _ in pbar:
            # volumes is a list (B, 1, D, H, W)
            volumes = [img.to(device) for img in volumes]
            labels = labels.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                if use_supcon:
                    # Return features for SupCon
                    logits, z_fused = model(volumes, return_features=True)
                    logits = logits.squeeze(1)
                    
                    # Classification loss
                    loss_cls = criterion(logits, labels)
                    
                    # SupCon loss (compute in float32 for stability)
                    with torch.autocast(device_type='cuda', enabled=False):
                        z_fused_fp32 = z_fused.float()
                        labels_int = labels.long()
                        loss_supcon = supervised_contrastive_loss(z_fused_fp32, labels_int, temperature=supcon_tau)
                    
                    # Learned weight for SupCon
                    if supcon_weight_param is not None:
                        w_supcon = torch.nn.functional.softplus(supcon_weight_param)
                    else:
                        w_supcon = torch.tensor(0.1, dtype=torch.float32, device=device)
                    
                    # Total loss
                    loss = loss_cls + w_supcon * loss_supcon
                else:
                    logits = model(volumes).squeeze(1)
                    loss = criterion(logits, labels)
                    loss_cls = loss
                    loss_supcon = torch.tensor(0.0)
                    w_supcon = torch.tensor(0.0)
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update EMA if enabled
            if ema is not None:
                ema.update(model)
            
            epoch_train_loss += loss.item()
            epoch_loss_cls += loss_cls.item()
            epoch_loss_supcon += loss_supcon.item()
            epoch_grad_norms.append(float(grad_norm))
            all_labels.extend(labels.detach().cpu().numpy().astype(np.int64))
            all_probs.extend(torch.sigmoid(logits).detach().float().cpu().numpy())
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_loss_cls = epoch_loss_cls / len(train_loader)
        avg_loss_supcon = epoch_loss_supcon / len(train_loader)
        train_losses.append(avg_train_loss)
        y_true = np.array(all_labels, dtype=np.int64)
        y_pred = (np.array(all_probs) > 0.5).astype(np.int64)
        train_acc = accuracy_score(y_true, y_pred)
        avg_grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_acc:.4f}'})

        
        
        if val_loader is not None:
            # ========== VALIDATION ==========
            model.eval()
            epoch_val_loss = 0.0
            val_labels, val_probs = [], []
            
            # Use EMA model for validation if enabled
            eval_model = ema.ema if ema is not None else model
            
            with torch.no_grad():
                for volumes, labels, _ in val_loader:
                    volumes = [v.to(device) for v in volumes]
                    labels  = labels.to(device, dtype=torch.float32)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = eval_model(volumes).squeeze(1) # (B, 1) -> (B,)
                        loss = criterion(logits, labels)
                    epoch_val_loss += loss.item()
                    val_labels.extend(labels.detach().cpu().numpy().astype(np.int64))
                    val_probs.extend(torch.sigmoid(logits).detach().float().cpu().numpy())
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                y_true = np.array(val_labels, dtype=np.int64)
                y_probs = np.array(val_probs, dtype=np.float32)
                y_pred = (y_probs > 0.5).astype(np.int64)
                val_acc = accuracy_score(y_true, y_pred)
                val_pr_auc = average_precision_score(y_true, y_probs)

    
        if val_loader is not None:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val PR-AUC: {val_pr_auc:.4f}")
        else:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")
        
        # W&B Logging
        if wandb.run is not None:
            log_payload = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/loss_cls": avg_loss_cls,
                "train/grad_norm": avg_grad_norm,
                "lr": optimizer.param_groups[0]['lr']
            }
            if use_supcon:
                log_payload["train/loss_supcon"] = avg_loss_supcon
                log_payload["train/supcon_weight"] = w_supcon.item()
                log_payload["train/supcon_tau"] = supcon_tau
            if val_loader is not None:
                log_payload.update({
                    "val/loss": avg_val_loss,
                    "val/accuracy": val_acc,
                    "val/pr_auc": val_pr_auc
                })
            wandb.log(log_payload)

        if val_loader is not None:
            # Early Stopping based on PR-AUC (higher is better)
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_model = copy.deepcopy(model.state_dict())
                if ema is not None:
                    best_ema = copy.deepcopy(ema.ema.state_dict())
                patience_counter = 0
                best_epoch = epoch + 1
                if log_dir:
                    torch.save(best_model, os.path.join(log_dir, 'checkpoint_best.pt'))
                    if ema is not None:
                        torch.save(best_ema, os.path.join(log_dir, 'checkpoint_best_ema.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(-val_pr_auc)  # Negate for mode='min' to maximize PR-AUC
            else:
                scheduler.step()

    if val_loader is not None and best_model:
        model.load_state_dict(best_model)
        if ema is not None and best_ema is not None:
            ema.ema.load_state_dict(best_ema)
    
    if val_loader is None:
        best_epoch = epochs
    
    return model, ema, best_epoch

def collect_labels_probs(model, data_loader, device, ema=None):
    """Collect labels and probabilities for a data loader."""
    eval_model = ema.ema if ema is not None else model
    eval_model.eval()
    all_labels, all_probs = [], []
    
    with torch.no_grad():
        for volumes, labels, _ in data_loader:
            volumes = [vol.to(device) for vol in volumes]
            logits = eval_model(volumes).squeeze(1)
            probs = torch.sigmoid(logits)
            all_labels.extend(labels.detach().to(torch.int64).cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels).astype(int).flatten(), np.array(all_probs).astype(float).flatten()


def _optimize_f1_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_tau = 0.5
    best_f1 = -1.0
    for tau in thresholds:
        y_pred = (y_prob > tau).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)
    return best_tau, best_f1


def _piecewise_constant_interp(x, y, x_grid):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]
    if len(unique_x) == 1:
        return np.full_like(x_grid, fill_value=float(unique_y[0]), dtype=float)
    f = interp1d(unique_x, unique_y, kind='previous', bounds_error=False, fill_value=(unique_y[0], unique_y[-1]), assume_sorted=True)
    return f(x_grid)


def _specificity_from_binary(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_paper_metrics(all_folds_results, bootstrap_iters=500, seed=42):
    fold_auc = []
    fold_pauc = []
    fold_interp_roc = []
    fold_interp_pr = []
    fold_thresholds = []
    fold_logs = []

    roc_grid = np.linspace(0.0, 1.0, 1001)
    pr_grid = np.linspace(0.0, 1.0, 1001)

    y_total = []
    yhat_total = []
    uid_total = []

    for fold_idx, fold in enumerate(all_folds_results):
        y_true_test = np.asarray(fold['y_true'], dtype=int)
        y_prob_test = np.asarray(fold['y_prob'], dtype=float)
        lesion_uids = np.asarray(fold['lesion_uids'])
        y_true_val = np.asarray(fold['val_y_true'], dtype=int)
        y_prob_val = np.asarray(fold['val_y_prob'], dtype=float)

        tau_opt, best_val_f1 = _optimize_f1_threshold(y_true_val, y_prob_val)
        fold_thresholds.append(tau_opt)

        y_pred_test = (y_prob_test > tau_opt).astype(int)
        y_total.append(y_true_test)
        yhat_total.append(y_pred_test)
        uid_total.append(lesion_uids)

        auc_i = roc_auc_score(y_true_test, y_prob_test)
        pauc_i = roc_auc_score(y_true_test, y_prob_test, max_fpr=0.1)
        fold_auc.append(float(auc_i))
        fold_pauc.append(float(pauc_i))

        fpr, tpr, _ = roc_curve(y_true_test, y_prob_test)
        interp_tpr = _piecewise_constant_interp(fpr, tpr, roc_grid)
        fold_interp_roc.append(interp_tpr)

        precision_curve, recall_curve, _ = precision_recall_curve(y_true_test, y_prob_test)
        recall_asc = recall_curve[::-1]
        precision_asc = precision_curve[::-1]
        interp_precision = _piecewise_constant_interp(recall_asc, precision_asc, pr_grid)
        fold_interp_pr.append(interp_precision)

        fold_logs.append({
            'fold': int(fold_idx),
            'tau_opt': float(tau_opt),
            'val_f1_opt': float(best_val_f1),
            'auc': float(auc_i),
            'pauc_0_1_std': float(pauc_i)
        })

    y_total = np.concatenate(y_total).astype(int)
    yhat_total = np.concatenate(yhat_total).astype(int)
    uid_total = np.concatenate(uid_total)

    global_f1 = f1_score(y_total, yhat_total, zero_division=0)
    global_acc = accuracy_score(y_total, yhat_total)
    global_recall = recall_score(y_total, yhat_total, zero_division=0)
    global_precision = precision_score(y_total, yhat_total, zero_division=0)
    global_specificity = _specificity_from_binary(y_total, yhat_total)

    unique_subjects = np.unique([uid.split('_', 1)[0] for uid in uid_total])
    rng = np.random.default_rng(seed)
    boot_acc, boot_spec, boot_ppv = [], [], []
    subject_ids = np.array([uid.split('_', 1)[0] for uid in uid_total])

    for _ in range(bootstrap_iters):
        sampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        sampled_indices = []
        for sid in sampled_subjects:
            sampled_indices.extend(np.where(subject_ids == sid)[0].tolist())
        sampled_indices = np.array(sampled_indices, dtype=int)

        y_b = y_total[sampled_indices]
        yhat_b = yhat_total[sampled_indices]
        boot_acc.append(accuracy_score(y_b, yhat_b))
        boot_spec.append(_specificity_from_binary(y_b, yhat_b))
        boot_ppv.append(precision_score(y_b, yhat_b, zero_division=0))

    ci_acc = np.percentile(np.array(boot_acc), [2.5, 97.5]).tolist()
    ci_spec = np.percentile(np.array(boot_spec), [2.5, 97.5]).tolist()
    ci_ppv = np.percentile(np.array(boot_ppv), [2.5, 97.5]).tolist()

    subject_pred_pos = {}
    subject_true_pos = {}
    for sid in unique_subjects:
        idx = np.where(subject_ids == sid)[0]
        subject_pred_pos[sid] = int(np.sum(yhat_total[idx] == 1))
        subject_true_pos[sid] = int(np.sum(y_total[idx] == 1))

    pred_counts = np.array([subject_pred_pos[sid] for sid in unique_subjects], dtype=float)
    true_counts = np.array([subject_true_pos[sid] for sid in unique_subjects], dtype=float)
    pearson_r, _ = stats.pearsonr(true_counts, pred_counts)
    mse_counts = float(np.mean((pred_counts - true_counts) ** 2))

    mean_roc = np.mean(np.stack(fold_interp_roc, axis=0), axis=0)
    mean_pr = np.mean(np.stack(fold_interp_pr, axis=0), axis=0)

    return {
        'fold_logs': fold_logs,
        'roc_curve_mean': {
            'fpr_grid': roc_grid.tolist(),
            'tpr_mean': mean_roc.tolist()
        },
        'pr_curve_mean': {
            'recall_grid': pr_grid.tolist(),
            'precision_mean': mean_pr.tolist()
        },
        'auc_mean': float(np.mean(fold_auc)),
        'auc_std': float(np.std(fold_auc)),
        'pauc_0_1_mean': float(np.mean(fold_pauc)),
        'pauc_0_1_std': float(np.std(fold_pauc)),
        'thresholds_opt': [float(t) for t in fold_thresholds],
        'global_metrics': {
            'f1': float(global_f1),
            'accuracy': float(global_acc),
            'sensitivity': float(global_recall),
            'specificity': float(global_specificity),
            'ppv': float(global_precision)
        },
        'bootstrap_ci_95': {
            'accuracy': [float(ci_acc[0]), float(ci_acc[1])],
            'specificity': [float(ci_spec[0]), float(ci_spec[1])],
            'ppv': [float(ci_ppv[0]), float(ci_ppv[1])]
        },
        'patient_level': {
            'pearson_r': float(pearson_r),
            'mse': float(mse_counts)
        }
    }

def evaluate_classifier(model, test_loader, device, output_dir=None, threshold=0.5, ema=None, return_details=False):
    """
    Valutazione finale su patch volumetriche 3D.
    If ema is provided, uses EMA model for evaluation.
    """
    # Use EMA model if provided
    eval_model = ema.ema if ema is not None else model
    eval_model.eval()
    all_labels, all_probs, all_uids = [], [], []
    
    with torch.no_grad():
        for volumes, labels, infos in tqdm(test_loader, desc="Testing"):
            volumes = [vol.to(device) for vol in volumes]
            logits = eval_model(volumes).squeeze(1)  # (B, 1) -> (B,)
            probs = torch.sigmoid(logits)
            
            all_labels.extend(labels.detach().to(torch.int64).cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())
            all_uids.extend([info['lesion_uid'] for info in infos])
    
    all_labels = np.array(all_labels).astype(int).flatten()
    all_probs = np.array(all_probs).astype(float).flatten()
    all_preds = (all_probs > threshold).astype(int)
    
    # Calcolo Metriche
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs),
        'average_precision': average_precision_score(all_labels, all_probs)
    }
    
    # ========== SCORE DISTRIBUTION ANALYSIS ==========
    # Split scores by class
    scores_pos = all_probs[all_labels == 1]  # Rim
    scores_neg = all_probs[all_labels == 0]  # NoRim
    
    # Compute overlap area and statistics only if both classes present
    overlap_area = np.nan
    mu_pos, sigma_pos = np.nan, np.nan
    mu_neg, sigma_neg = np.nan, np.nan
    fisher_ratio = np.nan
    
    if len(scores_pos) > 0 and len(scores_neg) > 0:
        # Summary statistics
        mu_pos, sigma_pos = np.mean(scores_pos), np.std(scores_pos)
        mu_neg, sigma_neg = np.mean(scores_neg), np.std(scores_neg)
        fisher_ratio = (mu_pos - mu_neg)**2 / (sigma_pos**2 + sigma_neg**2 + 1e-8)
        
        # Compute overlap area
        bins = 200
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        hist_pos, _ = np.histogram(scores_pos, bins=bin_edges, density=True)
        hist_neg, _ = np.histogram(scores_neg, bins=bin_edges, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        overlap_area = np.sum(np.minimum(hist_pos, hist_neg)) * bin_width
        
        # Plot overlapping histograms
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        ax_dist.hist(scores_pos, bins=50, density=True, alpha=0.5, label='Rim (y=1)', color='red')
        ax_dist.hist(scores_neg, bins=50, density=True, alpha=0.5, label='NoRim (y=0)', color='blue')
        ax_dist.set_xlabel('Predicted probability')
        ax_dist.set_ylabel('Density')
        ax_dist.set_title('Score Distributions s(x|y=1) vs s(x|y=0)')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        # Add overlap annotation
        ax_dist.text(0.95, 0.95, f'Overlap area = {overlap_area:.3f}',
                    transform=ax_dist.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save plot if output_dir provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig_dist.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=150, bbox_inches='tight')
        
        # W&B logging for distribution analysis
        if wandb.run is not None:
            wandb.log({
                "test/score_distribution_plot": wandb.Image(fig_dist),
                "test/scores_pos_hist": wandb.Histogram(scores_pos),
                "test/scores_neg_hist": wandb.Histogram(scores_neg),
                "test/mu_pos": mu_pos,
                "test/sigma_pos": sigma_pos,
                "test/mu_neg": mu_neg,
                "test/sigma_neg": sigma_neg,
                "test/fisher_ratio": fisher_ratio,
                "test/overlap_area": overlap_area
            })
        
        plt.close(fig_dist)

    # Creazione DataFrame per analisi lesioni
    df = pd.DataFrame({
        'lesion_uid': all_uids,
        'patch_prob': all_probs,
        'label': all_labels
    })
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_evaluation_curves(all_labels, all_probs, all_preds, output_dir)
        df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)

    # ========== W&B LOGGING ==========
    if wandb.run is not None:
        # 1. Log metriche base
        wandb.log({
            "test/accuracy": metrics['accuracy'],
            "test/precision": metrics['precision'],
            "test/recall": metrics['recall'],
            "test/f1": metrics['f1'],
            "test/auc": metrics['auc'],
            "test/average_precision": metrics['average_precision']
        })
        
        # 2. Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['NoRim', 'Rim'], 
                    yticklabels=['NoRim', 'Rim'], ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_ylabel('True Label')
        ax_cm.set_xlabel('Predicted Label')
        wandb.log({"test/confusion_matrix": wandb.Image(fig_cm)})
        plt.close(fig_cm)
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(fpr, tpr, label=f'ROC (AUC = {metrics["auc"]:.3f})', linewidth=2)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)
        wandb.log({"test/roc_curve": wandb.Image(fig_roc)})
        plt.close(fig_roc)
        
        # 4. Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        ax_pr.plot(recall_curve, precision_curve, 
                   label=f'PR (AP = {metrics["average_precision"]:.3f})', linewidth=2)
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        ax_pr.legend()
        ax_pr.grid(True, alpha=0.3)
        wandb.log({"test/precision_recall_curve": wandb.Image(fig_pr)})
        plt.close(fig_pr)
        
        # 5. Threshold Analysis
        thresholds = np.linspace(0, 1, 100)
        precisions, recalls, f1s = [], [], []
        
        for thresh in thresholds:
            preds_thresh = (all_probs > thresh).astype(int)
            precisions.append(precision_score(all_labels, preds_thresh, zero_division=0))
            recalls.append(recall_score(all_labels, preds_thresh, zero_division=0))
            f1s.append(f1_score(all_labels, preds_thresh, zero_division=0))
        
        fig_thresh, ax_thresh = plt.subplots(figsize=(10, 6))
        ax_thresh.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax_thresh.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax_thresh.plot(thresholds, f1s, label='F1-Score', linewidth=2)
        ax_thresh.axvline(x=threshold, color='r', linestyle='--', 
                         label=f'Current threshold ({threshold})', linewidth=1.5)
        ax_thresh.set_xlabel('Threshold')
        ax_thresh.set_ylabel('Score')
        ax_thresh.set_title('Threshold Analysis: Precision, Recall, F1-Score')
        ax_thresh.legend()
        ax_thresh.grid(True, alpha=0.3)
        wandb.log({"test/threshold_analysis": wandb.Image(fig_thresh)})
        plt.close(fig_thresh)
        
        # 6. Log confusion matrix come tabella wandb
        wandb.log({
            "test/confusion_matrix_table": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=['NoRim', 'Rim']
            )
        })

    print("\n--- Final Test Results (3D Patches) ---")
    for k, v in metrics.items():
        print(f" {k.capitalize()}: {v:.4f}")
        
    if return_details:
        return metrics, {
            'y_true': all_labels.astype(int).tolist(),
            'y_prob': all_probs.astype(float).tolist(),
            'lesion_uids': list(all_uids)
        }

    return metrics

def plot_evaluation_curves(y_true, y_prob, y_pred, output_dir):
    """Helper per generare i grafici di valutazione."""
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NoRim','Rim'], yticklabels=['NoRim','Rim'])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC: {roc_auc_score(y_true, y_prob):.3f}')
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title("ROC Curve")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()