# FRODO 💍 (Asymmetric QSM-FLAIR Modeling for Rim+ Lesion Classification)

Official PyTorch implementation of the paper: **"3D Classification of Paramagnetic Rim Lesions in Multiple Sclerosis via Asymmetric QSM-FLAIR Modeling"** (MICCAI 2026).

---
## 🧙‍♂️ Why "FRODO"?
**FRODO** stands for **F**usion framework for **R**im lesion classificati**O**n using multimodal **D**eep-learning neur**O**imaging.

But just like Frodo Baggins, this model has one specific mission: **to find the rings**. In Multiple Sclerosis, **Paramagnetic Rim Lesions ($Rim^+$)** appear on susceptibility-sensitive MRI scans as distinct, ring-like structures. FRODO is an asymmetric 3D multimodal framework tailored specifically to hunt down and classify these chronic active inflammatory biomarkers from Quantitative Suceptibility Mapping (QSM) and FLAIR MRI.

---

## 📝 Abstract
Paramagnetic rim lesions ($Rim^+$) identified on susceptibility-sensitive MRI have recently emerged as a specific biomarker of chronic active inflammation in Multiple Sclerosis (MS) and are associated with long-term disability progression. However, susceptibility imaging and expert interpretation remain limited to specialized centers, visual assessment is time-consuming and variable, and the low prevalence of $Rim^+$ lesions poses severe class imbalance challenges for automated analysis.

We propose **FRODO**, a 3D **F**usion framework for **R**im lesion classificati**O**n using multimodal **D**eep-learning neur**O**imaging, designed for lesion-level $Rim^+/Rim^-$ classification from Quantitative Susceptibility Mapping (QSM) and FLAIR MRI.

The architecture explicitly models modality asymmetry by treating QSM as the primary susceptibility-driven signal and conditioning it with FLAIR-derived structural context. To improve robustness under limited data, we employ self-supervised multimodal pretraining followed by supervised fine-tuning with contrastive regularization.

The method was evaluated on a clinically acquired cohort of 88 people with MS with expert lesion annotations as reference standard. Results highlight improved performance compared to prior architectures, supporting the effectiveness of asymmetric multimodal modeling for automated chronic active lesion identification.

## 🏗️ Architecture Overview
FRODO processes 3D lesion patches using an asymmetric approach:
1. **Primary Stream (QSM):** Extracts the core susceptibility-driven features of the rim.
2. **Conditioning Stream (FLAIR):** Provides structural and anatomical context to modulate the QSM features.
3. **Contrastive Regularization:** Maximizes robustness against severe class imbalance.

---

## 📁 Repository Structure

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
for fold in 0 1 2 3 4; do
    python train.py --data_dir data/synthetic --ssl_checkpoint checkpoints/ssl/best.pt \
        --fold $fold --output_dir checkpoints/finetuning --max_epochs 2
done
python evaluate.py --data_dir data/synthetic \
    --checkpoints_dir checkpoints/finetuning --output_dir results/
```

`evaluate.py` expects one checkpoint per fold (`fold_0.pt` ... `fold_4.pt`) in `--checkpoints_dir`, so `train.py` must be run once per fold first.

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
