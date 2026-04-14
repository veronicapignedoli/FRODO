import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os



def confusion_matrix_binary(y_true, y_pred, threshold=0.5, save_path=None):
    """
    compute confusion matrix for binary segmentation.
    y_true, y_pred: torch.Tensor with shape (B, 1, H, W) or similar
    threshold: threshold to binarize probabilities
    save_path: if specified, saves the matrix to a .txt file
    """
    # Binarize predictions
    if y_pred.dtype != torch.float32:
        y_pred = y_pred.float()
    y_prob = torch.sigmoid(y_pred) if y_pred.max() > 1 else y_pred
    y_bin = (y_prob > threshold).int()
    y_true = y_true.int()

    # Flatten
    y_true = y_true.view(-1).cpu().numpy()
    y_bin = y_bin.view(-1).cpu().numpy()

    TP = np.sum((y_true == 1) & (y_bin == 1))
    TN = np.sum((y_true == 0) & (y_bin == 0))
    FP = np.sum((y_true == 0) & (y_bin == 1))
    FN = np.sum((y_true == 1) & (y_bin == 0))

    cm = np.array([[TP, FP],
                   [FN, TN]])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("Confusion Matrix (binary segmentation)\n")
            f.write("         Pred=1   Pred=0\n")
            f.write(f"True=1   {TP:7d} {FN:7d}\n")
            f.write(f"True=0   {FP:7d} {TN:7d}\n")

    return cm

def dice_coef(y_true, y_pred, smooth=1e-4):
    y_true_f = y_true.contiguous().view(y_true.size(0), -1) #contiguous() is used to ensure that the tensor is stored in a contiguous chunk of memory, for view command to work correctly
    y_pred_f = y_pred.contiguous().view(y_pred.size(0), -1)
    intersection = (y_true_f * y_pred_f).sum(1)
    dice = (2. * intersection + smooth) / (y_true_f.sum(1) + y_pred_f.sum(1) + smooth)
    return dice.mean()

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

class DiceLoss(nn.Module):
    def __init__(self, apply_sigmoid=False):
        super(DiceLoss, self).__init__()
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs, targets, smooth=1.0):
        if self.apply_sigmoid:
            inputs = torch.sigmoid(inputs)

        inputs = inputs.contiguous().view(inputs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        dice = (2. * intersection + smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + smooth)

        return 1 - dice.mean()
    

def focal_loss(y_pred, y_true, alpha=0.80, gamma=2.0):
    """
    Focal loss for binary classification with logits.
    y_pred: raw logits (NO sigmoid)
    y_true: binary labels [0, 1]
    
    Args:
        y_pred: model predictions (logits)
        y_true: ground truth labels
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    BCE = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    p_t = torch.sigmoid(y_pred)
    p_t = torch.where(y_true == 1, p_t, 1 - p_t)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    loss = alpha * focal_weight * BCE
    
    return loss.mean()

def dice_bce_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = F.binary_cross_entropy(y_pred, y_true)
    return dice + bce

def dice_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred, alpha=alpha, gamma=gamma)
    return dice + focal

def dice_bce_loss_kd(teacher_logits, student_logits, t=1.0, a=1.0, b=0.0):
    teacher_soft = torch.sigmoid(teacher_logits / t)
    student_prob = torch.sigmoid(student_logits / t)
    
    dice = dice_loss(teacher_soft, student_prob)
    bce = F.binary_cross_entropy(student_prob, teacher_soft)
    return a * dice + b * bce


"""
def kl_divergence(y_true, y_pred, distribution='pixelwise'):
    #dim=1 is softmax over channel, i.e. pixel-wise distribution mantaining spatial structure
    #dim=(2,3) is softmax over space, you obtain a distribution of attention of filters over the image
 
    if distribution == 'pixelwise':
        y_true = F.softmax(y_true, dim=1)
        y_pred = F.softmax(y_pred, dim=1)
    elif distribution == 'spatial':
        y_true = F.softmax(y_true, dim = (2,3))
        y_pred = F.softmax(y_pred, dim = (2,3))
    y_true = torch.clamp(y_true, 1e-7, 1)
    y_pred = torch.clamp(y_pred, 1e-7, 1)
    return torch.sum(y_true * torch.log(y_true / y_pred))

"""
def kl_divergence(y_true, y_pred, distribution='pixelwise', T=3.0):
    # correct version for binary segmentation (1 output channel) and temperature scaling for knowledge distillation.    
    if distribution == 'pixelwise':
        # sigmoid with temperature to get probabilities for binary segmentation
        # treat every pixel as a Bernoulli distribution
        p = torch.sigmoid(y_true / T)
        q = torch.sigmoid(y_pred / T)
        
        # avoid log(0) by clamping probabilities
        p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
        q = torch.clamp(q, 1e-7, 1.0 - 1e-7)
        
        # KL divergence for Bernoulli distributions
        kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        return kl.mean() * (T**2)

    elif distribution == 'spatial':
        #if you want to maintain spatial logic (attention on filters),  
        #here Softmax makes sense because you normalize the entire image (dim 2,3)
        y_true_soft = F.softmax(y_true / T, dim=(2, 3))
        y_pred_soft = F.softmax(y_pred / T, dim=(2, 3))
        
        y_true_soft = torch.clamp(y_true_soft, 1e-7, 1)
        y_pred_soft = torch.clamp(y_pred_soft, 1e-7, 1)
        
        return torch.sum(y_true_soft * torch.log(y_true_soft / y_pred_soft)) * (T**2)



def l2_loss(y_true, y_pred):
    return F.mse_loss(y_pred, y_true, reduction="mean")


def l1_loss(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))



def feature_matching_loss(teacher_feat, student_feat, normalize=True):
    """
    Feature matching loss for knowledge distillation
    """
    if normalize:
        # Normalize features to have mean 0 and std 1
        teacher_feat = F.normalize(teacher_feat, p=2, dim=1)
        student_feat = F.normalize(student_feat, p=2, dim=1)
    
    # L2 loss
    return F.mse_loss(student_feat, teacher_feat)   

def spatial_feature_loss(teacher_feat, student_feat):
    """
    Feature matching that preserves spatial structure
    """
    # Do not view(-1) to preserve spatial dimensions
    B, C, H, W = teacher_feat.shape

    # Loss per channel to preserve spatial information
    loss = F.mse_loss(teacher_feat, student_feat, reduction='none')

    # Mean over batch and channels, but keep spatial info
    return loss.mean()

def attention_feature_loss(teacher_feat, student_feat):
    """
    Feature matching with weights based on feature importance
    """
    # Compute attention weights from teacher
    attention = torch.mean(teacher_feat.abs(), dim=1, keepdim=True)
    attention = F.softmax(attention.view(attention.size(0), -1), dim=1)
    attention = attention.view_as(attention)
    
    # Weighted L2 loss
    diff = (teacher_feat - student_feat) ** 2
    weighted_diff = diff * attention
    
    return weighted_diff.mean()

def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    Classical Supervised Contrastive Loss (SupCon).
    
    Args:
        embeddings: (B, D) - Normalized or unnormalized embeddings
        labels: (B,) - Class labels (binary: 0 or 1)
        temperature: float - Temperature for scaling logits
    
    Returns:
        loss: scalar tensor, mean over valid anchors
    
    Notes:
        - Embeddings are L2-normalized internally
        - Positives for anchor i are all j != i with label[j] == label[i]
        - If anchor has no positives in batch, it contributes 0
        - Uses float32 for stability
    """
    # Ensure float32 for stability
    embeddings = embeddings.float()
    labels = labels.long()
    
    B = embeddings.shape[0]
    
    # L2 normalization of embeddings
    embeddings_normalized = F.normalize(embeddings, p=2, dim=1)  # (B, D)
    
    # Compute similarity matrix: (B, B)
    # sim[i,j] = z_i^T z_j
    similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())  # (B, B)
    
    # Scale by temperature
    similarity_matrix = similarity_matrix / temperature
    
    # Mask: same label as anchor
    labels_equal = labels.unsqueeze(1) == labels.unsqueeze(0)  # (B, B)
    
    # Mask out the diagonal (anchor cannot be its own positive)
    mask_positives = labels_equal & ~torch.eye(B, dtype=torch.bool, device=embeddings.device)  # (B, B)
    
    # Mask for negatives (different label)
    mask_negatives = ~labels_equal  # (B, B)
    
    # For each anchor i, count positives
    num_positives = mask_positives.sum(dim=1)  # (B,)
    
    loss_total = 0.0
    count_valid_anchors = 0
    
    for i in range(B):
        if num_positives[i] == 0:
            # No positives for this anchor, skip
            continue
        
        count_valid_anchors += 1
        
        # Similarity scores for all samples (including anchor itself)
        sim_i = similarity_matrix[i]  # (B,)
        
        # Max trick for numerical stability
        sim_i_max = sim_i.max().detach()
        sim_i = sim_i - sim_i_max
        
        # Exponentials
        exp_sim = torch.exp(sim_i)  # (B,)
        
        # Positive part: sum of exp(sim) for positives (excluding anchor)
        positive_sum = (exp_sim * mask_positives[i]).sum()
        
        # Denominator: sum of exp(sim) for all except anchor
        denominator = (exp_sim * (mask_positives[i] | mask_negatives[i])).sum()
        
        # Avoid division by zero (though shouldn't happen with proper masking)
        if denominator > 0:
            loss_i = -torch.log(positive_sum / denominator + 1e-8)
            loss_total = loss_total + loss_i
    
    if count_valid_anchors == 0:
        # No valid anchors (e.g., batch with single class), return zero loss
        return torch.tensor(0.0, dtype=embeddings.dtype, device=embeddings.device)
    
    # Average over valid anchors
    loss = loss_total / count_valid_anchors
    
    return loss