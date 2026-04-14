import os
import numpy as np
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import shutil
import nibabel as nb
from scipy.ndimage import center_of_mass, binary_dilation, generate_binary_structure

# Configuration /data/cil/veronica/HSMn
CONFIG = {
    'PATCH_SIZE': (64, 64, 64),  
    'MODALITIES': ['T1', 'FLAIR', 'QSM', 'QSMp'],
    'BASE_PATH': '/data/cil/veronica/HSMn',
    'EXCEL_PATH': '/nethome/vpignedoli/3DRim_classifier/DB_RIM_classification_provv.xlsx', 
    'OUTPUT_PATH': '/data/cil/veronica/3DRim_Classification_patches', 
    'ID_COLUMN': 'Lesion_label',
    'LABEL_COLUMN': 'RIM',
    'MIN_BLOB_SIZE': 110,  # Minimum number of voxels for a valid lesion
    'APPLY_MASK_MULTIPLICATION': False,  # Set to False to disable masking
    'APPLY_LARGEMASK_GEOM': True,  # Apply morphologically dilated mask (perilesional buffer)
    'ADAPTIVE_DILATION': True,
    'ADAPTIVE_K': 0.75,
    'ADAPTIVE_R_MIN': 5,
    'ADAPTIVE_R_MAX': 16,
    'DILATION_RADIUS': 20,  # Radius in voxels for morphological dilation 
    'CLEAN_OUTPUT': True,
}

def norm_img_int_rng(img, v_min=None, v_max=None, p_low=0.5, p_high=99.5):
    """
    Linear scaling to [0, 1].
    If v_min/v_max are provided, performs Fixed Range Scaling (Quantitative).
    If they are None, calculates robust percentiles on the current volume (Morphological).
    """
    if v_min is None or v_max is None:
        # Robust calculation based on the volume content (excluding background at 0)
        mask = img > 0
        roi_values = img[mask] if np.any(mask) else img
        v_min = np.percentile(roi_values, p_low)
        v_max = np.percentile(roi_values, p_high)
    
    # Clipping and scaling
    img = np.clip(img, v_min, v_max)
    return (img - v_min) / (v_max - v_min + 1e-8)


def normalize_volume(vol, modality):
    """
    Normalizes a 3D volume based on modality type.
    morphologic sequences (T1, FLAIR): per-volume normalization (relative context).
    (QSM): fixed range normalization (absolute context).
    """
    if modality in ['T1', 'FLAIR']:
        # Morphologic: per-volume normalization
        return norm_img_int_rng(vol, p_low=0.5, p_high=99.5)
    elif modality == 'QSM':
        # Quantitative: fixed range [-300, 300]
        return norm_img_int_rng(vol, v_min=-300, v_max=300)
    elif modality == 'QSMp':
        # Quantitative: fixed range [0, 150]
        return norm_img_int_rng(vol, v_min=0, v_max=150)
    else:
        # Default: return as is
        return vol


def split_subjects(data_path):
    all_subjects = [f for f in sorted(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, f))]
    return {
            'train': all_subjects[:62],
            'val': all_subjects[62:75],
            'test': all_subjects[75:]
            }

def extract_patch(volume, center, patch_size):
    """
    Extract a patch of fixed dimension centered in 'center'
    Padding in case of exit from image boundaries.
    Ensures output is always exactly patch_size.
    """
    z, y, x = center
    dz, dy, dx = [s // 2 for s in patch_size]
    
    # Ensure integer coordinates
    z, y, x = int(round(z)), int(round(y)), int(round(x))
    
    z_start, z_end = z - dz, z + dz
    y_start, y_end = y - dy, y + dy
    x_start, x_end = x - dx, x + dx
    
    # Calculate padding needed
    pad_z = [max(0, -z_start), max(0, z_end - volume.shape[0])]
    pad_y = [max(0, -y_start), max(0, y_end - volume.shape[1])]
    pad_x = [max(0, -x_start), max(0, x_end - volume.shape[2])]
    
    # Extract patch with clipping
    patch = volume[
        max(0, z_start):min(volume.shape[0], z_end),
        max(0, y_start):min(volume.shape[1], y_end),
        max(0, x_start):min(volume.shape[2], x_end)
    ]

    # Apply padding to maintain constant size
    if any(p > 0 for p in pad_z + pad_y + pad_x):
        patch = np.pad(patch, (pad_z, pad_y, pad_x), mode='constant', constant_values=0)
    
    # Final check: ensure exact size (handle rounding issues)
    if patch.shape != tuple(patch_size):
        # Crop or pad to exact size
        final_patch = np.zeros(patch_size, dtype=patch.dtype)
        
        # Calculate valid region
        z_slice = slice(0, min(patch.shape[0], patch_size[0]))
        y_slice = slice(0, min(patch.shape[1], patch_size[1]))
        x_slice = slice(0, min(patch.shape[2], patch_size[2]))
        
        final_patch[z_slice, y_slice, x_slice] = patch[z_slice, y_slice, x_slice]
        return final_patch
    
    return patch

def compute_patches_extraction():
    try:
        df = pd.read_excel(CONFIG['EXCEL_PATH'])
        df[CONFIG['LABEL_COLUMN']] = df[CONFIG['LABEL_COLUMN']].replace('?', 0)
        
        label_map = {}
        for _, row in df.iterrows():
            patient_id = str(row['ID']).zfill(3)
            lesion_id = int(row[CONFIG['ID_COLUMN']])
            rim_label = int(row[CONFIG['LABEL_COLUMN']])
            label_map[(patient_id, lesion_id)] = rim_label
        
        print(f"[✓] Loaded {len(label_map)} lesion labels from Excel")
    except Exception as e:
        print(f"[CRITICAL] Errore caricamento Excel: {e}")
        return


    if CONFIG['CLEAN_OUTPUT'] and os.path.exists(CONFIG['OUTPUT_PATH']):
        shutil.rmtree(CONFIG['OUTPUT_PATH'])
    os.makedirs(CONFIG['OUTPUT_PATH'], exist_ok=True)


    labels_dict = {'train': {}, 'val': {}, 'test': {}}
    debug_stats = {
        'total_subjects': 0, 'subjects_with_phases': 0,
        'total_lesions_found': 0, 'lesions_with_label': 0,
        'lesions_without_label': 0, 'lesions_too_small': 0,
        'lesions_too_small_rim': 0,
        'lesions_too_small_norim': 0,
        'subjects_processed': 0
    }
    stats = {'Rim': 0, 'NoRim': 0, 'Unknown': 0, 'TooSmall': 0, 'TooSmall_Rim': 0, 'TooSmall_NoRim': 0}

    subjects = split_subjects(CONFIG['BASE_PATH'])
    print(f"Train subjects ({len(subjects['train'])}): {subjects['train']}")
    print(f"Val subjects ({len(subjects['val'])}): {subjects['val']}")
    print(f"Test subjects ({len(subjects['test'])}): {subjects['test']}")

    debug_stats['total_subjects'] = len(subjects)


    for phase in ['train', 'val', 'test']:
        for subj in tqdm(subjects[phase], desc=f"Processing {phase.upper()} subjects"):
            subj_path = os.path.join(CONFIG['BASE_PATH'], subj)
            
            # Verifica che la cartella del soggetto esista
            if not os.path.isdir(subj_path):
                continue
            
            debug_stats['subjects_processed'] += 1
            
            # Carica il file NIfTI della maschera clusterizzata (direttamente in subj_path)
            mask_files = glob(os.path.join(subj_path, "FLAIR_lesion_clust_to_T1.nii.gz"))
            if not mask_files:
                print(f"[WARNING] Mask file not found for subject {subj}")
                continue
            
            # Carichiamo la maschera volumetrica
            mask_nifti = nb.load(mask_files[0])
            mask_data = mask_nifti.get_fdata()
            lesion_ids = np.unique(mask_data)
            lesion_ids = lesion_ids[lesion_ids > 0]  # Escludiamo background

            # Carichiamo i volumi 3D per ogni modalità e li normalizziamo
            volumes = {}
            valid_subject = True
            for mod in CONFIG['MODALITIES']:
                # ON T1 SPACE
                if mod == 'T1' or mod == 'FLAIR':
                    mod_pattern = os.path.join(subj_path, f"{mod}_bet.nii.gz")
                elif mod == 'QSMp':
                    mod_pattern = os.path.join(subj_path, f"QSM_param_to_T1.nii.gz")
                else:
                    mod_pattern = os.path.join(subj_path, f"{mod}_to_T1.nii.gz")
                mod_files = glob(mod_pattern)
                if mod_files:
                    # Carica il volume e normalizza
                    vol_raw = nb.load(mod_files[0]).get_fdata()
                    volumes[mod] = normalize_volume(vol_raw, mod)
                else:
                    print(f"[WARNING] {mod} file not found for subject {subj}")
                    valid_subject = False
                    break
            
            if not valid_subject:
                continue

            # Inizializza dizionario per questo soggetto
            labels_dict[phase][subj] = {}

            # LOOP SULLE LESIONI 
            for lid in lesion_ids:
                mask_blob = (mask_data == lid)
                blob_size = np.sum(mask_blob)  # volume in voxel

                composite_key = (subj, int(lid))
                
                # Check size before processing
                if blob_size < CONFIG['MIN_BLOB_SIZE']:
                    stats['TooSmall'] += 1
                    debug_stats['lesions_too_small'] += 1
                    
                    # Track if this small lesion was labeled as Rim or NoRim
                    if composite_key in label_map:
                        label = label_map[composite_key]
                        if label == 1:
                            stats['TooSmall_Rim'] += 1
                            debug_stats['lesions_too_small_rim'] += 1
                        else:
                            stats['TooSmall_NoRim'] += 1
                            debug_stats['lesions_too_small_norim'] += 1
                    continue

                if composite_key not in label_map:
                    stats['Unknown'] += 1
                    debug_stats['lesions_without_label'] += 1
                    continue

                label = label_map[composite_key]
                debug_stats['lesions_with_label'] += 1
                debug_stats['total_lesions_found'] += 1
                
                # Centro di massa 3D (z, y, x)
                center = center_of_mass(mask_blob)
                patch_id = f"LID{int(lid)}"
                
                # Estrazione Patch 3D per ogni modalità
                # OUTPUT: OUTPUT_PATH/phase/subj/mod/patchID.npy
                for mod in CONFIG['MODALITIES']:
                    mod_dir = os.path.join(CONFIG['OUTPUT_PATH'], phase, subj, mod)
                    os.makedirs(mod_dir, exist_ok=True)
                    
                    p = extract_patch(volumes[mod], center, CONFIG['PATCH_SIZE'])
                    
                    if CONFIG.get('APPLY_MASK_MULTIPLICATION'):
                        m_patch = extract_patch(mask_blob.astype(np.float32), center, CONFIG['PATCH_SIZE'])
                        p = p * (m_patch > 0.5)
                    elif CONFIG.get('APPLY_LARGEMASK_GEOM'):
                        # Morphological dilation: M' = M ⊕ B (spherical structuring element)
                        if CONFIG.get('ADAPTIVE_DILATION', False):
                            rho = ((3.0 * blob_size) / (4.0 * np.pi)) ** (1.0 / 3.0)
                            radius = int(round(np.clip(
                                CONFIG.get('ADAPTIVE_K', 0.75) * rho,
                                CONFIG.get('ADAPTIVE_R_MIN', 5),
                                CONFIG.get('ADAPTIVE_R_MAX', 16)
                            )))
                        else:
                            radius = int(CONFIG.get('DILATION_RADIUS', 10))
                        struct_elem = generate_binary_structure(3, 1)  # 3D connectivity
                        mask_dilated = binary_dilation(mask_blob, structure=struct_elem, iterations=radius)
                        m_patch_dilated = extract_patch(mask_dilated.astype(np.float32), center, CONFIG['PATCH_SIZE'])
                        p = p * (m_patch_dilated > 0.5)

                    np.save(os.path.join(mod_dir, f"{patch_id}.npy"), p.astype(np.float32))

                # Salvataggio Maschera Binaria Patch 3D
                mask_patch_dir = os.path.join(CONFIG['OUTPUT_PATH'], phase, subj, 'mask_clust')
                os.makedirs(mask_patch_dir, exist_ok=True)
                p_mask = extract_patch(mask_blob.astype(np.float32), center, CONFIG['PATCH_SIZE'])
                np.save(os.path.join(mask_patch_dir, f"{patch_id}.npy"), p_mask.astype(np.float32))

                # Update Dict
                labels_dict[phase][subj][patch_id] = int(label)
                stats['Rim' if label == 1 else 'NoRim'] += 1

    # SALVATAGGIO FINALE
    with open(os.path.join(CONFIG['OUTPUT_PATH'], 'labels.json'), 'w') as f:
        json.dump(labels_dict, f, indent=2)

    print("\n" + "="*30 + " EXTRACTION COMPLETE " + "="*30)
    print(f"Total subjects processed: {debug_stats['subjects_processed']}/{debug_stats['total_subjects']}")
    print(f"Total patches extracted: {stats['Rim'] + stats['NoRim']}")
    print(f"  Rim: {stats['Rim']} | NoRim: {stats['NoRim']}")
    print(f"\nLesions excluded:")
    print(f"  TooSmall (< {CONFIG['MIN_BLOB_SIZE']} voxels): {stats['TooSmall']} total")
    print(f"    → Rim: {stats['TooSmall_Rim']}")
    print(f"    → NoRim: {stats['TooSmall_NoRim']}")
    print(f"    → Unknown label: {stats['TooSmall'] - stats['TooSmall_Rim'] - stats['TooSmall_NoRim']}")
    print(f"  Unknown label: {stats['Unknown']}")
    print(f"\nLabels saved to: {os.path.join(CONFIG['OUTPUT_PATH'], 'labels.json')}")