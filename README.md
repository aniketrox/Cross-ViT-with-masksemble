# ðŸ”¬ Cross-ViT with Masksembles
### Multi-Scale Vision Transformers with Fast Uncertainty Quantification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/aniketrox/Cross-ViT-with-masksemble?style=for-the-badge&color=gold)](https://github.com/aniketrox/Cross-ViT-with-masksemble/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge)](https://github.com/aniketrox/Cross-ViT-with-masksemble/pulls)

A unified deep learning framework integrating dual-scale Vision Transformers (Cross-ViT) with Masksemble-based ensemble inference for robust classification and calibrated uncertainty estimation.

---

## ðŸŒŸ Key Features

- ðŸ”€ Dual-Branch Multi-Scale ViT: Jointly extracts fine-grained patch details and broad contextual features simultaneously.
- âš¡ Linear-Time Cross-Attention: Fuses representations between different patch resolutions using an efficient token-exchange mechanism.
- ðŸŽ¯ Lightweight Masksembles: Emulates deep ensembles in a single forward/backward pipeline using structured binary masks without the NÃ— training cost.
- ðŸ©º Uncertainty Quantification (UQ): Computes predictive entropy and variance scores to flag out-of-distribution (OOD) or low-confidence medical/clinical samples.
- ðŸ“Š Modular PyTorch Pipeline: Clean training scripts, custom data loaders, and plug-and-play config support.

---

## ðŸ—ï¸ Architecture Overview

```
Input Image (H x W x 3)
   â”‚
   â”œâ”€â”€â”€â–º Small Patch Branch (Fine Scale)  â”€â”€â–º [Patch Embed] â”€â”€â–º Transformer Blocks â”€â”€â”
   â”‚                                                                                 â–¼
   â”‚                                                                         Cross-Attention
   â”‚                                                                           Token Fusion
   â”‚                                                                                 â–²
   â””â”€â”€â”€â–º Large Patch Branch (Coarse Scale) â”€â”€â–º [Patch Embed] â”€â”€â–º Transformer Blocks â”€â”€â”˜
                                                                                     â”‚
                                                                           Fused Representation
                                                                                     â”‚
                                                                           â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                                                           â–¼                   â–¼
                                                                     [Masksemble Head]   [Classifier]
                                                                           â”‚                   â”‚
                                                                           â–¼                   â–¼
                                                                  Ensemble Predictions & Uncertainty
```

### How Masksembles Work in Cross-ViT
1. Multi-Scale Feature Extraction: Small patches (Ps = 8x8 or 12x12) capture localized textures, while large patches (Pl = 16x16 or 24x24) preserve global structure.
2. Cross-Scale Interaction: The class token of each branch queries the patch tokens of the opposite branch via cross-attention.
3. Structured Masking: Fixed pseudo-random binary masks {M_1, M_2, ..., M_B} are applied to intermediate feature activations to simulate B distinct subnetworks for robust ensemble predictions.

---

## ðŸ“‚ Repository Structure

```tree
Cross-ViT-with-masksemble/
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ crossvit.py           # Cross-ViT backbone implementation
â”‚   â”œâ”€â”€ masksembles.py        # Masksemble layer & binary mask generator
â”‚   â””â”€â”€ crossvit_masksemble.py# End-to-end model wrapper
â”œâ”€â”€ datasets/
â”‚   â””â”€â”€ dataset_loader.py     # Custom dataloaders (HAM10000 / ISIC / ImageNet)
â”œâ”€â”€ utils/
â”‚   â”œâ”€â”€ metrics.py            # Accuracy, F1, ECE, and uncertainty metrics
â”‚   â””â”€â”€ loss.py               # Custom ensemble loss functions
â”œâ”€â”€ train.py                  # Training pipeline
â”œâ”€â”€ evaluate.py               # Evaluation & uncertainty calibration script
â”œâ”€â”€ requirements.txt          # Dependencies
â””â”€â”€ README.md                 # Project documentation
```

---

## ðŸš€ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/aniketrox/Cross-ViT-with-masksemble.git
cd Cross-ViT-with-masksemble
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Dependencies
```txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.22.0
scikit-learn>=1.0.0
pandas>=1.4.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

---

## ðŸ’» Training & Evaluation

### Training the Model
```bash
python train.py \
  --data_dir ./data/HAM10000 \
  --img_size 224 \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --n_masks 4 \
  --scale 2.0 \
  --output_dir ./checkpoints
```

### Evaluating Performance & Uncertainty
```bash
python evaluate.py \
  --checkpoint ./checkpoints/best_model.pth \
  --data_dir ./data/HAM10000 \
  --compute_uncertainty True \
  --save_plots True
```

---

## ðŸ“ˆ Uncertainty Quantification & Calibration

The model outputs predictions across ensemble masks M to compute:
- Predictive Mean Probability
- Predictive Entropy (Epistemic + Aleatoric)
- Expected Calibration Error (ECE)

| Model Variant | Accuracy (%) | Macro F1 | ECE (â†“) | Params (M) |
| :--- | :---: | :---: | :---: | :---: |
| Standard ViT-B/16 | 84.2 | 0.812 | 0.084 | ~86M |
| Standard Cross-ViT | 86.8 | 0.845 | 0.062 | ~43M |
| Cross-ViT + Masksembles (Ours) | 88.9 | 0.873 | 0.028 | ~44M |

---

## ðŸ¤ Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

---

## ðŸ“œ License
This project is licensed under the MIT License.

## ðŸ“š References & Acknowledgments
- CrossViT: Chen et al., "CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification", ICCV 2021.
- Masksembles: Durasov et al., "Masksembles for Uncertainty Estimation", CVPR 2021.
- Built upon PyTorch Image Models (timm).
