#!/usr/bin/env python3
"""
Script to compute normalization statistics for 3D MRI volumes.
This script analyzes the entire HSMn dataset to extract global statistics
for each modality, which can then be used for normalization during patch extraction.

The statistics are computed on the training set only to avoid data leakage.
"""

import os
import numpy as np
import nibabel as nb
from glob import glob
from tqdm import tqdm
import json

# Configuration
BASE_PATH = '/home/veronica/veronica/Datasets/MSclerosis/HSMn'
MODALITIES = ['T1', 'FLAIR', 'QSM', 'QSMp']
OUTPUT_FILE = '/home/veronica/veronica/3DRim_classifier /normalization_stats_3D.json'

def split_subjects(data_path):
    """Split subjects into train/val/test following the same logic as patches_extraction.py"""
    all_subjects = [f for f in sorted(os.listdir(data_path)) 
                   if os.path.isdir(os.path.join(data_path, f)) and f != '116']
    return {
        'train': all_subjects[:62],
        'val': all_subjects[62:75],
        'test': all_subjects[75:]
    }

def compute_volume_statistics():
    """
    Compute comprehensive statistics for each modality on the training set.
    Returns a dictionary with statistics for each modality.
    """
    print("="*80)
    print("COMPUTING NORMALIZATION STATISTICS FOR 3D VOLUMES")
    print("="*80)
    
    subjects = split_subjects(BASE_PATH)
    print(f"\nAnalyzing {len(subjects['train'])} training subjects...")
    print(f"Train subjects: {subjects['train'][:5]}...{subjects['train'][-5:]}")
    
    # Initialize statistics collectors
    stats = {mod: {'values': []} for mod in MODALITIES}
    
    # Process training subjects only
    for subj in tqdm(subjects['train'], desc="Processing training volumes"):
        subj_path = os.path.join(BASE_PATH, subj)
        if not os.path.isdir(subj_path):
            continue
        
        for mod in MODALITIES:
            # Determine file path based on modality
            if mod == 'T1' or mod == 'FLAIR':
                mod_pattern = os.path.join(subj_path, f"{mod}_bet.nii.gz")
            elif mod == 'QSMp':
                mod_pattern = os.path.join(subj_path, f"QSM_param_to_T1.nii.gz")
            else:  # QSM
                mod_pattern = os.path.join(subj_path, f"{mod}_to_T1.nii.gz")
            
            mod_files = glob(mod_pattern)
            if not mod_files:
                print(f"Warning: {mod} file not found for subject {subj}")
                continue
            
            # Load volume
            vol = nb.load(mod_files[0]).get_fdata()
            
            # Collect non-zero voxels (exclude background)
            non_zero_voxels = vol[vol != 0]
            if len(non_zero_voxels) > 0:
                stats[mod]['values'].append(non_zero_voxels)
    
    # Compute final statistics
    normalization_stats = {}
    
    print("\n" + "="*80)
    print("COMPUTED STATISTICS (Training Set - Non-zero Voxels Only)")
    print("="*80)
    print(f"{'Modality':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'P1':<12} {'P99':<12} {'Max':<12} {'Voxels':<15}")
    print("-"*80)
    
    for mod in MODALITIES:
        if not stats[mod]['values']:
            print(f"Warning: No data found for {mod}")
            continue
        
        # Concatenate all values from all training subjects
        all_values = np.concatenate(stats[mod]['values'])
        
        # Compute statistics
        mean_val = float(np.mean(all_values))
        std_val = float(np.std(all_values))
        min_val = float(np.min(all_values))
        max_val = float(np.max(all_values))
        p1_val = float(np.percentile(all_values, 1))
        p99_val = float(np.percentile(all_values, 99))
        median_val = float(np.median(all_values))
        num_voxels = int(len(all_values))
        
        normalization_stats[mod] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'p1': p1_val,
            'p99': p99_val,
            'median': median_val,
            'num_voxels': num_voxels,
            'description': f'Statistics computed on {len(subjects["train"])} training subjects (non-zero voxels only)'
        }
        
        # Print statistics
        print(f"{mod:<10} {mean_val:<12.2f} {std_val:<12.2f} {min_val:<12.2f} "
              f"{p1_val:<12.2f} {p99_val:<12.2f} {max_val:<12.2f} {num_voxels:<15,}")
    
    print("="*80)
    
    return normalization_stats

def save_statistics(stats, output_file):
    """Save statistics to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n[✓] Statistics saved to: {output_file}")

def print_recommended_normalization():
    """Print recommended normalization strategies for each modality"""
    print("\n" + "="*80)
    print("RECOMMENDED NORMALIZATION STRATEGIES")
    print("="*80)
    print("""
Based on the computed statistics, here are the recommended normalization approaches:

1. T1 & FLAIR (Morphological sequences):
   - Use Z-score normalization: (x - mean) / std
   - These are intensity-based, and relative context matters
   
2. QSM & QSMp (Quantitative sequences):
   - QSM: Z-score normalization with global statistics
   - QSMp: Z-score normalization with global statistics
   - These are quantitative, absolute values matter

For Z-score normalization, use the mean and std from the statistics above.
    """)
    print("="*80)

def main():
    """Main function"""
    # Compute statistics
    stats = compute_volume_statistics()
    
    # Save to JSON
    save_statistics(stats, OUTPUT_FILE)
    
    # Print recommendations
    print_recommended_normalization()
    
    print("\n[✓] Done! You can now use these statistics in patches_extraction.py")
    print(f"    Copy the statistics from {OUTPUT_FILE} into the NORMALIZATION_STATS dictionary.")

if __name__ == "__main__":
    main()
