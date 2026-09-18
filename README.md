# FRODO

Official PyTorch implementation of **"3D Classification of Paramagnetic Rim
Lesions in Multiple Sclerosis via Asymmetric QSM–FLAIR Modeling"**
(Pignedoli et al., MICCAI 2026).

FRODO classifies individual multiple sclerosis lesions as paramagnetic-rim
positive (Rim+) or negative (Rim-) from paired 3D QSM and FLAIR MRI patches,
using an asymmetric two-stream architecture with FiLM-based cross-modal
fusion, self-supervised pretraining, and supervised contrastive fine-tuning.

## 📁 Repository structure

```
FRODO/
├── frodo/
│   ├── model.py            # MultiModalClassifier architecture + optional EMA wrapper
│   ├── losses.py            # BCE, supervised contrastive loss, composite loss, SSL loss
│   ├── dataset.py            # data loading, normalization, masking, sampler
│   └── augmentations.py       # 3D flip/rotation augmentations
├── train_ssl.py                # Stage 1: self-supervised pretraining
├── train.py                     # Stage 2: supervised fine-tuning (one fold at a time)
├── evaluate.py                   # 5-fold evaluation, lesion- and person-level metrics
├── generate_synthetic_data.py     # synthetic data generator for testing, no real data needed
├── data/README.md                  # expected data format
└── requirements.txt
```

## ⚙️ Installation

```bash
git clone https://github.com/veronicapignedoli/FRODO.git
cd FRODO
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 🧠 Data

The clinical dataset used in the paper is private (institutional ethics
approval, no public redistribution). See [`data/README.md`](data/README.md)
for the expected data format and how to adapt `frodo/dataset.py` to your own
cohort.

## 🚀 Usage

**Stage 1 — self-supervised pretraining:**

```bash
python train_ssl.py --data_dir data/synthetic --output_dir checkpoints/ssl
```

**Stage 2 — supervised fine-tuning (one fold at a time):**

```bash
python train.py --data_dir data/synthetic --ssl_checkpoint checkpoints/ssl/best.pt \
    --fold 0 --output_dir checkpoints/finetuning
```

Repeat for `--fold 1` through `--fold 4`.

**Evaluation (all 5 folds):**

```bash
python evaluate.py --data_dir data/synthetic \
    --checkpoints_dir checkpoints/finetuning --output_dir results/
```

**Quick smoke test with synthetic data:**

```bash
python generate_synthetic_data.py --output_dir data/synthetic
python train_ssl.py --data_dir data/synthetic --output_dir checkpoints/ssl --max_epochs 2
python train.py --data_dir data/synthetic --ssl_checkpoint checkpoints/ssl/best.pt \
    --fold 0 --output_dir checkpoints/finetuning --max_epochs 2
python evaluate.py --data_dir data/synthetic \
    --checkpoints_dir checkpoints/finetuning --output_dir results/
```

Every script is self-documenting via `--help`.

## 📖 Citation

```bibtex
@inproceedings{pignedoli2026frodo,
  title     = {3D Classification of Paramagnetic Rim Lesions in Multiple Sclerosis
               via Asymmetric QSM--FLAIR Modeling},
  author    = {Pignedoli, Veronica and Boffa, Giacomo and Noceti, Nicoletta and
               Inglese, Matilde and Odone, Francesca and Moro, Matteo},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

## 📄 License

Released under the [MIT License](LICENSE).
