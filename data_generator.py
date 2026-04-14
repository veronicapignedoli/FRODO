import os
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from glob import glob
from scipy.ndimage import rotate

ENABLE_EXPENSIVE_3D_ROTATIONS = False  # Temporarily disabled for CPU bottleneck / GPU utilization debugging.


def apply_3d_augmentation(volumes):
    """
    Apply 3D augmentation to a list of volumes.
    All volumes receive the same transformations to maintain consistency.
    
    Args:
        volumes: list of numpy arrays with shape (D, H, W)
    
    Returns:
        list of augmented volumes
    """
    # Random flip along each axis (50% probability each)
    if random.random() > 0.5:  # Flip along depth axis
        volumes = [np.flip(vol, axis=0).copy() for vol in volumes]
    
    if random.random() > 0.5:  # Flip along height axis
        volumes = [np.flip(vol, axis=1).copy() for vol in volumes]
    
    if random.random() > 0.5:  # Flip along width axis
        volumes = [np.flip(vol, axis=2).copy() for vol in volumes]
    
    # Small random rotations (±10 degrees on each axis)
    # Temporarily disabled by default for performance debugging.
    if ENABLE_EXPENSIVE_3D_ROTATIONS:
        if random.random() > 0.5:
            # Random rotation around z-axis (axial plane)
            angle_z = random.uniform(-10, 10)
            volumes = [rotate(vol, angle_z, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0) for vol in volumes]
        
        if random.random() > 0.5:
            # Random rotation around y-axis (coronal plane)
            angle_y = random.uniform(-10, 10)
            volumes = [rotate(vol, angle_y, axes=(0, 2), reshape=False, order=1, mode='constant', cval=0) for vol in volumes]
        
        if random.random() > 0.5:
            # Random rotation around x-axis (sagittal plane)
            angle_x = random.uniform(-10, 10)
            volumes = [rotate(vol, angle_x, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0) for vol in volumes]
    
    return volumes



class PatchDataset(Dataset):
    """
    New structure: labels_dict[phase][subj][patch_id] = label
    """
    def __init__(self, patches_path, labels_dict, subjects, modalities, phase='train',
                 augment=True, oversample_rim=False, balance_classes=False, subject_phase_map=None):
        self.patches_path = patches_path
        self.labels_dict = labels_dict
        self.modalities = modalities
        self.phase = phase
        self.augment = augment and phase == 'train'
        self.oversample_rim = oversample_rim and phase == 'train'
        self.balance_classes = balance_classes
        
        self.samples = []
        self.subject_phase_map = subject_phase_map
        
        if self.subject_phase_map is None:
            # Check if phase exists in labels_dict
            if phase not in labels_dict:
                print(f"[WARNING] Phase '{phase}' not found in labels_dict")
                return
            
            for subj in subjects:
                if subj not in labels_dict[phase]:
                    continue
                
                # New structure: labels_dict[phase][subj] contains patch_id -> label mapping
                for patch_id, label in labels_dict[phase][subj].items():
                    paths = {}
                    valid = True
                    for mod in modalities:
                        # Path structure: patches_path/phase/subj/mod/patch_id.npy
                        p = os.path.join(patches_path, phase, subj, mod, f"{patch_id}.npy")
                        if os.path.exists(p):
                            paths[mod] = p
                        else:
                            valid = False
                            break
                    
                    if valid:
                        self.samples.append({
                            'paths': paths, 
                            'label': label, 
                            'subject': subj,
                            'patch_id': patch_id
                        })
        else:
            for subj in subjects:
                subj_phase = self.subject_phase_map.get(subj)
                if subj_phase is None:
                    continue
                if subj_phase not in labels_dict:
                    continue
                if subj not in labels_dict[subj_phase]:
                    continue
                
                for patch_id, label in labels_dict[subj_phase][subj].items():
                    paths = {}
                    valid = True
                    for mod in modalities:
                        p = os.path.join(patches_path, subj_phase, subj, mod, f"{patch_id}.npy")
                        if os.path.exists(p):
                            paths[mod] = p
                        else:
                            valid = False
                            break
                    
                    if valid:
                        self.samples.append({
                            'paths': paths,
                            'label': label,
                            'subject': subj,
                            'patch_id': patch_id
                        })
        
        print(f"[PatchDataset] {phase}: Found {len(self.samples)} samples for {len(subjects)} subjects")
        
        if self.balance_classes:
            self._balance_rim_norim_samples()
        elif self.augment and self.oversample_rim:
            self._oversample_rim_samples()

    def _balance_rim_norim_samples(self):
        """Balance dataset by keeping all RIM samples and randomly sampling equal number of non-RIM samples."""
        rim_samples = [s for s in self.samples if s['label'] == 1]
        norim_samples = [s for s in self.samples if s['label'] == 0]
        
        if len(rim_samples) == 0:
            print(f"  [Balance Classes] Warning: No RIM samples found, keeping all samples")
            return
        
        print(f"  [Balance Classes] Original - RIM: {len(rim_samples)}, NoRIM: {len(norim_samples)}")
        
        # Sample non-RIM to match RIM count
        if len(norim_samples) > len(rim_samples):
            sampled_norim = random.sample(norim_samples, len(rim_samples))
        else:
            sampled_norim = norim_samples
        
        # Combine balanced samples
        self.samples = rim_samples + sampled_norim
        random.shuffle(self.samples)
        
        print(f"  [Balance Classes] Balanced - RIM: {len(rim_samples)}, NoRIM: {len(sampled_norim)} (Total: {len(self.samples)})")
    
    def _oversample_rim_samples(self):
        """Bilanciamento del dataset tramite oversampling delle patch RIM (Label 1)."""
        rim_samples = [s for s in self.samples if s['label'] == 1]
        norim_samples = [s for s in self.samples if s['label'] == 0]
        
        if len(rim_samples) == 0: return
        
        ratio = len(norim_samples) / len(rim_samples)
        if ratio > 1:
            print(f"  [Oversampling 3D] RIM: {len(rim_samples)}, NoRIM: {len(norim_samples)} (Ratio: {ratio:.2f})")
            augmented_rim = []
            for _ in range(int(ratio) - 1):
                augmented_rim.extend(rim_samples)
            self.samples.extend(augmented_rim)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Caricamento volumi (D, H, W)
        images_list = [np.load(sample['paths'][m]).astype(np.float32) for m in self.modalities]
        
        # Apply 3D augmentation if enabled
        if self.augment:
            images_list = apply_3d_augmentation(images_list)
        
        # Converti ogni modalità in un tensore (1, D, H, W) separato
        tensors = [torch.from_numpy(img).unsqueeze(0).float() for img in images_list]
        
        label = torch.tensor(sample['label'], dtype=torch.float32)
        
        return tensors, label, {
            'lesion_uid': f"{sample['subject']}_{sample['patch_id']}",
            'label': sample['label']
        }

    def __len__(self):
        return len(self.samples)


def custom_collate_fn(batch):
    """Custom collate function to handle list of tensors per sample"""
    # batch[i][0] is a list of tensors for modalities
    num_modalities = len(batch[0][0])
    
    # Stack each modality separately
    images = [torch.stack([item[0][i] for item in batch]) for i in range(num_modalities)]
    labels = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    
    return images, labels, metadata


def _build_weighted_sampler(dataset):
    labels = [s['label'] for s in dataset.samples]
    class_counts = np.bincount(labels, minlength=2)
    if class_counts[0] == 0 or class_counts[1] == 0:
        print("  [WeightedRandomSampler] Warning: one class is missing, sampler disabled")
        return None
    class_weights = {0: 1.0 / class_counts[0], 1: 1.0 / class_counts[1]}
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_patch_dataloaders(patches_path, labels_dict, subjects, modalities, batch_size=16, 
                         num_workers=16, augment=True, oversample_rim=False, balance_classes=False,
                         weighted_sampler=False, pin_memory=True, subject_phase_map=None):
    """Create dataloaders for train/val/test."""
    
    train_dataset = PatchDataset(
        patches_path, labels_dict, subjects['train'], modalities, 
        phase='train', augment=augment, oversample_rim=oversample_rim, balance_classes=balance_classes,
        subject_phase_map=subject_phase_map
    )
    val_dataset = PatchDataset(
        patches_path, labels_dict, subjects['val'], modalities, 
        phase='val', augment=False, oversample_rim=False, balance_classes=balance_classes,
        subject_phase_map=subject_phase_map
    )
    test_dataset = PatchDataset(
        patches_path, labels_dict, subjects['test'], modalities, 
        phase='test', augment=False, oversample_rim=False, balance_classes=balance_classes,
        subject_phase_map=subject_phase_map
    )
    
    train_sampler = None
    train_shuffle = True
    if weighted_sampler:
        train_sampler = _build_weighted_sampler(train_dataset)
        if train_sampler is not None:
            train_shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_shuffle, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader