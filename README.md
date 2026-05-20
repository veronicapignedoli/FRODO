# FRODO 💍 (Asymmetric QSM-FLAIR Modeling for Rim+ Lesion Classification)

Official PyTorch implementation of the paper: **"3D Classification of Paramagnetic Rim Lesions in Multiple Sclerosis via Asymmetric QSM-FLAIR Modeling"** (MICCAI 2026).

---

## 🧙‍♂️ Why "FRODO"?
**FRODO** stands for **F**lair-conditioned **R**im **O**riented **D**etection & **O**ptimization (or simply: *The Ring Bearer*). Just like Frodo Baggins, this model has one specific mission: **to find the rings**. 
In Multiple Sclerosis, **Paramagnetic Rim Lesions ($Rim^+$)** appear on MRI scans as distinct hyperintense rings. FRODO is a 3D multimodal framework tailored to hunt down and classify these chronic active inflammatory biomarkers.

---

## 📝 Abstract
Paramagnetic rim lesions (Rim$^+$) identified on susceptibility-sensitive MRI have recently emerged as a specific biomarker of chronic active inflammation in Multiple Sclerosis (MS) and are associated with long-term disability progression. However, susceptibility imaging and expert interpretation remain limited to specialized centers, visual assessment is time-consuming and variable, and the low prevalence of Rim$^+$ lesions poses severe class imbalance challenges for automated analysis.
We propose FRODO, a 3D Fusion framework for Rim lesion classificatiOn using multimodal Deep-learning neurOimaging, designed for lesion-level Rim$^+$/Rim$^-$ classification from Quantitative Susceptibility Mapping (QSM) and FLAIR MRI.
%We propose a 3D multimodal deep learning framework for lesion-level Rim$^+$/Rim$^-$ classification from Quantitative Susceptibility Mapping (QSM) and FLAIR MRI. 
The architecture explicitly models modality asymmetry by treating QSM as the primary susceptibility-driven signal and conditioning it with FLAIR-derived structural context. To improve robustness under limited data, we employ self-supervised multimodal pretraining followed by supervised fine-tuning with contrastive regularization.
The method was evaluated on a clinically acquired cohort of 88 people with MS with expert lesion annotations as reference standard. Results highlight improved performance compared to prior architectures, supporting the effectiveness of asymmetric multimodal modeling for automated chronic active lesion identification.

## 🏗️ Architecture Overview
FRODO processes 3D lesion patches using an asymmetric approach:
1. **Primary Stream (QSM):** Extracts the core susceptibility-driven features of the rim.
2. **Conditioning Stream (FLAIR):** Provides structural and anatomical context to modulate the QSM features.
3. **Contrastive Regularization:** Maximizes robustness against severe class imbalance.

---

## 📁 Repository Structure
