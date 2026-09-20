# Data format

## 1. Why the dataset is not included

FRODO is trained on paired QSM/FLAIR MRI lesion patches from a private clinical
cohort of multiple sclerosis patients. This data cannot be publicly released:
it is patient imaging data collected under an institutional ethics committee
approval that does not permit public redistribution. To use this code with
your own data, prepare it in the format described below, or use
`generate_synthetic_data.py` to try the full pipeline without any real data.

## 2. Expected directory structure

```
data_root/
├── patches/
│   ├── <lesion_id>_qsm.npy     # float32, shape (64, 64, 64)
│   ├── <lesion_id>_flair.npy   # float32, shape (64, 64, 64)
│   └── <lesion_id>_mask.npy    # float32 or bool, shape (64, 64, 64)
└── labels.csv
```

Every `lesion_id` referenced in `labels.csv` must have all three `.npy` files
present under `patches/`.

## 3. `labels.csv` format

| column         | type | description                                   |
|----------------|------|------------------------------------------------|
| `lesion_id`    | str  | unique identifier, matches the patch filenames |
| `label`        | int  | `1` = Rim+, `0` = Rim-                          |
| `participant_id` | str | identifies the patient the lesion belongs to  |
| `fold`         | int  | outer cross-validation fold, `0`-`4`            |

Fold assignment must be done at the participant level (all lesions from one
participant belong to the same fold) to avoid leakage between train and test.

Example:

```csv
lesion_id,label,participant_id,fold
L0000,0,P000,0
L0001,1,P000,0
L0002,0,P007,3
```

## 4. Patch format

- One `.npy` file per modality per lesion, plus one lesion mask.
- All arrays: `float32`, shape `(64, 64, 64)`.
- `<lesion_id>_mask.npy` is the (already dilated) lesion mask used to zero
  out background outside the lesion's perilesional region; see `frodo/dataset.py`.

## 5. Normalization conventions

Patches are expected **raw** (not pre-normalized). Normalization is applied
in `frodo/dataset.py`:

- **FLAIR**: robust percentile normalization, computed per participant across
  all of that participant's lesion patches (0.5th-99.5th percentile), then
  clipped and rescaled to `[0, 1]`.
- **QSM**: clipped to `[-300, 300]`, then linearly rescaled to `[0, 1]`.

Lesions whose mask has fewer than 110 voxels are excluded automatically when
the dataset is constructed (a count of excluded lesions is printed).

## 6. Testing the pipeline with synthetic data

```bash
python generate_synthetic_data.py --output_dir data/synthetic
python train_ssl.py --data_dir data/synthetic --output_dir checkpoints/ssl --max_epochs 2
for fold in 0 1 2 3 4; do
    python train.py --data_dir data/synthetic --ssl_checkpoint checkpoints/ssl/best.pt \
        --fold $fold --output_dir checkpoints/finetuning --max_epochs 2
done
python evaluate.py --data_dir data/synthetic \
    --checkpoints_dir checkpoints/finetuning --output_dir results/
```

`evaluate.py` expects one checkpoint per fold (`fold_0.pt` ... `fold_4.pt`), so
`train.py` must be run once per fold first.

Metrics on synthetic data are meaningless (the "lesions" are random blobs) —
this only verifies that the pipeline runs end to end.

## 7. Adapting `frodo/dataset.py` to a different data format

If your data doesn't match this layout, the integration point is
`frodo/dataset.py`'s `LesionPatchDataset`:

- `_path()` builds the file paths for a given lesion id and modality — change
  this if your files are named or organized differently.
- `_normalize_flair()` / `_normalize_qsm()` implement the normalization
  above — adjust if your data requires different ranges or per-volume
  (rather than per-participant) statistics.
- `load_labels()` just reads a CSV with the four columns above — replace it
  with your own metadata source, as long as the returned `DataFrame` has
  `lesion_id`, `label`, `participant_id`, and `fold` columns.

No other file needs to change: `train_ssl.py`, `train.py`, and `evaluate.py`
only depend on this interface.
