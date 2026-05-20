# FRODO 💍 (Asymmetric QSM-FLAIR Modeling for Rim+ Lesion Classification)

Official PyTorch implementation of the paper: **"3D Classification of Paramagnetic Rim Lesions in Multiple Sclerosis via Asymmetric QSM-FLAIR Modeling"** (MICCAI 2026).

---
## 🧙‍♂️ Why "FRODO"?
**FRODO** stands for **F**usion framework for **R**im lesion classificati**O**n using multimodal **D**eep-learning neur**O**imaging. 

But just like Frodo Baggins, this model has one specific mission: **to find the rings**. 
In Multiple Sclerosis, **Paramagnetic Rim Lesions ($Rim^+$)** appear on susceptibility-sensitive MRI scans as distinct, ring-like structures. FRODO is an asymmetric 3D multimodal framework tailored specifically to hunt down and classify these chronic active inflammatory biomarkers from Quantitative Susceptibility Mapping (QSM) and FLAIR MRI.
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
