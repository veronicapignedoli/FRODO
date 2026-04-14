"""
Script to generate histograms of normalized patch values
Only considers lesion voxels (masked regions)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def load_patch_with_mask(patches_path, phase, subject, patch_id, modality):
    """
    Load a patch and its corresponding mask
    
    Args:
        patches_path: Base path to patches
        phase: 'train', 'val', or 'test'
        subject: Subject ID
        patch_id: Patch ID (e.g., 'LID100')
        modality: Modality name (e.g., 'FLAIR', 'QSM', 'T1', 'QSMp')
    
    Returns:
        patch: numpy array of patch values
        mask: numpy array of mask (lesion region)
    """
    patch_path = os.path.join(patches_path, phase, subject, modality, f"{patch_id}.npy")
    mask_path = os.path.join(patches_path, phase, subject, 'mask_clust', f"{patch_id}.npy")
    
    if not os.path.exists(patch_path) or not os.path.exists(mask_path):
        return None, None
    
    patch = np.load(patch_path)
    mask = np.load(mask_path)
    
    return patch, mask


def get_masked_values(patch, mask):
    """
    Extract only lesion voxel values from patch using mask
    
    Args:
        patch: 3D numpy array of patch values
        mask: 3D numpy array of mask (lesion=1, background=0)
    
    Returns:
        masked_values: 1D array of voxel values inside the lesion
    """
    # Ensure mask is binary
    mask_binary = (mask > 0).astype(bool)
    
    # Extract values where mask is True
    masked_values = patch[mask_binary]
    
    return masked_values


def collect_all_masked_values(patches_path, labels_dict, modalities, phases=['train', 'val', 'test'], 
                               label_filter=None, max_patches_per_modality=None):
    """
    Collect all masked (lesion-only) values for each modality across all patches
    
    Args:
        patches_path: Base path to patches
        labels_dict: Dictionary with labels
        modalities: List of modalities to process
        phases: List of phases to include
        label_filter: If specified, only include patches with this label (0 or 1)
        max_patches_per_modality: Maximum number of patches to process per modality (for speed)
    
    Returns:
        values_dict: Dictionary with modality -> list of masked values
        stats_dict: Dictionary with statistics per modality
    """
    values_dict = {mod: [] for mod in modalities}
    patch_counts = {mod: 0 for mod in modalities}
    
    print(f"\nCollecting masked lesion values for modalities: {modalities}")
    print(f"Phases: {phases}")
    if label_filter is not None:
        print(f"Filtering patches with label: {label_filter}")
    
    total_patches = 0
    for phase in phases:
        if phase not in labels_dict:
            continue
        for subject in labels_dict[phase]:
            for patch_id, label in labels_dict[phase][subject].items():
                # Filter by label if specified
                if label_filter is not None and label != label_filter:
                    continue
                
                total_patches += 1
    
    print(f"Total patches to process: {total_patches}")
    
    with tqdm(total=total_patches * len(modalities), desc="Processing patches") as pbar:
        for phase in phases:
            if phase not in labels_dict:
                continue
            
            for subject in labels_dict[phase]:
                for patch_id, label in labels_dict[phase][subject].items():
                    # Filter by label if specified
                    if label_filter is not None and label != label_filter:
                        pbar.update(len(modalities))
                        continue
                    
                    for modality in modalities:
                        # Check if we've reached the limit for this modality
                        if max_patches_per_modality and patch_counts[modality] >= max_patches_per_modality:
                            pbar.update(1)
                            continue
                        
                        # Load patch and mask
                        patch, mask = load_patch_with_mask(patches_path, phase, subject, patch_id, modality)
                        
                        if patch is not None and mask is not None:
                            # Get masked values (only lesion voxels)
                            masked_vals = get_masked_values(patch, mask)
                            
                            if len(masked_vals) > 0:
                                values_dict[modality].extend(masked_vals.flatten())
                                patch_counts[modality] += 1
                        
                        pbar.update(1)
    
    # Compute statistics
    stats_dict = {}
    for modality in modalities:
        vals = np.array(values_dict[modality])
        if len(vals) > 0:
            stats_dict[modality] = {
                'count': len(vals),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'median': float(np.median(vals)),
                'q25': float(np.percentile(vals, 25)),
                'q75': float(np.percentile(vals, 75)),
                'patches_processed': patch_counts[modality]
            }
        else:
            stats_dict[modality] = {'count': 0, 'patches_processed': 0}
    
    return values_dict, stats_dict


def plot_histograms(values_dict, stats_dict, output_dir, label_filter=None, bins=100):
    """
    Create and save histogram plots for each modality
    
    Args:
        values_dict: Dictionary with modality -> list of values
        stats_dict: Dictionary with statistics
        output_dir: Directory to save plots
        label_filter: Label filter used (for title)
        bins: Number of bins for histogram
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Individual histograms for each modality
    for modality, values in values_dict.items():
        if len(values) == 0:
            print(f"Warning: No values for {modality}, skipping histogram")
            continue
        
        vals = np.array(values)
        stats = stats_dict[modality]
        
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        n, bins_edges, patches = plt.hist(vals, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        
        # Add statistics text
        stats_text = (
            f"Count: {stats['count']:,} voxels\n"
            f"Patches: {stats['patches_processed']}\n"
            f"Mean: {stats['mean']:.4f}\n"
            f"Std: {stats['std']:.4f}\n"
            f"Min: {stats['min']:.4f}\n"
            f"Max: {stats['max']:.4f}\n"
            f"Median: {stats['median']:.4f}"
        )
        
        plt.text(0.98, 0.97, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9,
                family='monospace')
        
        # Add vertical lines for mean and median
        plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.4f}")
        plt.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.4f}")
        
        label_str = f" (Label {label_filter})" if label_filter is not None else ""
        plt.title(f'Histogram of Normalized {modality} Values in Lesion Regions{label_str}', fontsize=14, fontweight='bold')
        plt.xlabel('Normalized Intensity Value', fontsize=12)
        plt.ylabel('Frequency (Number of Voxels)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        label_suffix = f"_label{label_filter}" if label_filter is not None else ""
        filename = f'histogram_{modality}{label_suffix}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    
    # Combined histogram (all modalities in one plot)
    plt.figure(figsize=(14, 8))
    
    colors = ['steelblue', 'coral', 'green', 'purple', 'orange', 'brown']
    for idx, (modality, values) in enumerate(values_dict.items()):
        if len(values) == 0:
            continue
        
        vals = np.array(values)
        color = colors[idx % len(colors)]
        
        plt.hist(vals, bins=bins, alpha=0.5, label=modality, color=color, edgecolor='black')
    
    label_str = f" (Label {label_filter})" if label_filter is not None else ""
    plt.title(f'Combined Histograms of Normalized Values in Lesion Regions{label_str}', fontsize=14, fontweight='bold')
    plt.xlabel('Normalized Intensity Value', fontsize=12)
    plt.ylabel('Frequency (Number of Voxels)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save combined figure
    label_suffix = f"_label{label_filter}" if label_filter is not None else ""
    filename = f'histogram_combined{label_suffix}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def save_statistics(stats_dict, output_dir, label_filter=None):
    """Save statistics to JSON file"""
    label_suffix = f"_label{label_filter}" if label_filter is not None else ""
    stats_file = os.path.join(output_dir, f'statistics{label_suffix}.json')
    
    with open(stats_file, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    
    print(f"\nStatistics saved to: {stats_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("STATISTICS SUMMARY (Lesion-Masked Voxels Only)")
    print("="*70)
    for modality, stats in stats_dict.items():
        if stats['count'] == 0:
            print(f"\n{modality}: No data")
            continue
        
        print(f"\n{modality}:")
        print(f"  Patches processed: {stats['patches_processed']}")
        print(f"  Total voxels:      {stats['count']:,}")
        print(f"  Mean ± Std:        {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range:             [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Median (IQR):      {stats['median']:.4f} [{stats['q25']:.4f}, {stats['q75']:.4f}]")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Generate histograms of normalized patch values (lesion-masked)')
    parser.add_argument('--patches_path', type=str,
                        default='/home/veronica/veronica/Datasets/MSclerosis/3DRim_Classification_patches',
                        help='Path to patches directory')
    parser.add_argument('--modalities', type=str, nargs='+',
                        default=['T1', 'FLAIR', 'QSM', 'QSMp'],
                        help='Modalities to analyze')
    parser.add_argument('--phases', type=str, nargs='+',
                        default=['train', 'val', 'test'],
                        help='Phases to include')
    parser.add_argument('--output_dir', type=str,
                        default='histogram_analysis',
                        help='Output directory for histograms')
    parser.add_argument('--bins', type=int, default=100,
                        help='Number of bins for histograms')
    parser.add_argument('--label_filter', type=int, default=None, choices=[0, 1],
                        help='Only process patches with this label (0=no-rim, 1=rim+)')
    parser.add_argument('--max_patches', type=int, default=None,
                        help='Maximum patches per modality (for faster testing)')
    parser.add_argument('--separate_by_label', action='store_true',
                        help='Create separate histograms for rim+ and no-rim patches')
    
    args = parser.parse_args()
    
    # Load labels
    labels_path = os.path.join(args.patches_path, 'labels.json')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found at {labels_path}")
    
    print(f"Loading labels from: {labels_path}")
    with open(labels_path, 'r') as f:
        labels_dict = json.load(f)
    
    print(f"\nConfiguration:")
    print(f"  Patches path: {args.patches_path}")
    print(f"  Modalities: {args.modalities}")
    print(f"  Phases: {args.phases}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Bins: {args.bins}")
    print(f"  Label filter: {args.label_filter}")
    print(f"  Max patches per modality: {args.max_patches}")
    
    if args.separate_by_label:
        # Create histograms separately for each label
        for label in [0, 1]:
            label_name = "rim_positive" if label == 1 else "no_rim"
            print(f"\n{'='*70}")
            print(f"Processing {label_name} patches (label={label})")
            print(f"{'='*70}")
            
            output_dir = os.path.join(args.output_dir, label_name)
            
            # Collect values
            values_dict, stats_dict = collect_all_masked_values(
                patches_path=args.patches_path,
                labels_dict=labels_dict,
                modalities=args.modalities,
                phases=args.phases,
                label_filter=label,
                max_patches_per_modality=args.max_patches
            )
            
            # Plot histograms
            plot_histograms(values_dict, stats_dict, output_dir, label_filter=label, bins=args.bins)
            
            # Save statistics
            save_statistics(stats_dict, output_dir, label_filter=label)
    
    else:
        # Create histograms for all patches or with specified filter
        print(f"\n{'='*70}")
        print("Processing all patches" if args.label_filter is None else f"Processing patches with label={args.label_filter}")
        print(f"{'='*70}")
        
        # Collect values
        values_dict, stats_dict = collect_all_masked_values(
            patches_path=args.patches_path,
            labels_dict=labels_dict,
            modalities=args.modalities,
            phases=args.phases,
            label_filter=args.label_filter,
            max_patches_per_modality=args.max_patches
        )
        
        # Plot histograms
        plot_histograms(values_dict, stats_dict, args.output_dir, label_filter=args.label_filter, bins=args.bins)
        
        # Save statistics
        save_statistics(stats_dict, args.output_dir, label_filter=args.label_filter)
    
    print(f"\n✓ Analysis complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
