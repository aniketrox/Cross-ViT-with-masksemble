<div align="center">

# Cross-ViT with Masksembles
### Multi-Scale Vision Transformers with Fast Uncertainty Quantification

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License"></a>
  <a href="https://github.com/aniketrox/Cross-ViT-with-masksemble/stargazers"><img src="https://img.shields.io/github/stars/aniketrox/Cross-ViT-with-masksemble?style=for-the-badge&color=gold" alt="Stars"></a>
  <a href="https://github.com/aniketrox/Cross-ViT-with-masksemble/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge" alt="PRs Welcome"></a>
</p>

<p align="center">
  <img src="https://github.com/aniketrox/Cross-ViT-with-masksemble/blob/main/architecture/model.png" alt="Cross-ViT Architecture Banner" width="850"/>
</p>

<p align="center">
  <b>A unified deep learning framework integrating dual-scale Vision Transformers (Cross-ViT) with Masksemble-based ensemble inference for robust classification and calibrated uncertainty estimation.</b>
</p>

[Key Features](#-key-features) â€¢ [Architecture](#-architecture-overview) â€¢ [Getting Started](#-getting-started) â€¢ [Training & Evaluation](#-training--evaluation) â€¢ [Benchmark](#-benchmark-comparison) â€¢ [Contributing](#-contributing)

---

</div>

## Key Features

* **Dual-Branch Multi-Scale ViT:** Jointly extracts fine-grained patch details (small patches) and broad contextual semantics (large patches) simultaneously.
* **Linear-Time Cross-Attention:** Efficiently fuses multi-scale representations via class token querying with linear O(N) complexity.
* **Fast Masksembles:** Emulates deep ensembles in a single forward/backward pass using fixed structured binary masks without the Nx training penalty.
* **Uncertainty Quantification (UQ):** Computes calibrated predictive entropy and variance scores to detect out-of-distribution (OOD) and ambiguous inputs.
* **Production-Ready Pipeline:** Fully modular PyTorch scripts, data augmentation, checkpointing, and evaluation metrics.

---

## Architecture Overview

<div align="center">
  <img src="https://raw.githubusercontent.com/IBM/CrossViT/main/.github/crossvit_arch.png" alt="Cross-ViT Detailed Mechanism" width="800"/>
</div>

### How Masksembles Work in Cross-ViT

```
Input Image (H x W x 3)
   |
   +---> Small Patch Branch (Fine Scale)  --> [Patch Embed] --> Transformer Blocks --+
   |                                                                                 |
   |                                                                         Cross-Attention
   |                                                                           Token Fusion
   |                                                                                 |
   +---> Large Patch Branch (Coarse Scale) -> [Patch Embed] --> Transformer Blocks --+
                                                                                     |
                                                                           Fused Representation
                                                                                     |
                                                                           +---------+---------+
                                                                           |                   |
                                                                     [Masksemble Head]   [Classifier]
                                                                           |                   |
                                                                           v                   v
                                                                  Ensemble Predictions & Uncertainty
```

1. **Multi-Scale Feature Extraction:** Small patches (8x8 or 12x12) capture localized texture anomalies, while large patches (16x16 or 24x24) preserve global anatomical structure.
2. **Cross-Scale Interaction:** The class token of each branch queries the patch tokens of the opposite branch via cross-attention.
3. **Structured Masking:** Fixed pseudo-random binary masks {M_1, M_2, ..., M_B} are applied to intermediate feature activations to simulate B distinct subnetworks.

---

## Repository Structure

```tree
Cross-ViT-with-masksemble/
|-- models/
|   |-- crossvit.py           # Cross-ViT backbone implementation
|   |-- masksembles.py        # Masksemble layer & binary mask generator
|   +-- crossvit_masksemble.py# End-to-end model wrapper
|-- datasets/
|   +-- dataset_loader.py     # Custom dataloaders (HAM10000 / ISIC / ImageNet)
|-- utils/
|   |-- metrics.py            # Accuracy, F1, ECE, and uncertainty metrics
|   +-- loss.py               # Custom ensemble loss functions
|-- train.py                  # Training pipeline
|-- evaluate.py               # Evaluation & uncertainty calibration script
|-- requirements.txt          # Dependencies
+-- README.md                 # Project documentation
```

---

## Getting Started

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

### 3. Requirements
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

## Training & Evaluation

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

## Benchmark Comparison

| Model Architecture | Accuracy (%) | Macro F1 | ECE (Down) | Total Params |
| :--- | :---: | :---: | :---: | :---: |
| Standard ViT-B/16 | 84.2% | 0.812 | 0.084 | ~86M |
| Standard Cross-ViT | 86.8% | 0.845 | 0.062 | ~43M |
| **Cross-ViT + Masksembles (Ours)** | **88.9%** | **0.873** | **0.028** | **~44M** |

---

## Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [Issues page](https://github.com/aniketrox/Cross-ViT-with-masksemble/issues).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References & Acknowledgments

* **CrossViT:** Chen et al., *"CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification"*, ICCV 2021.
* **Masksembles:** Durasov et al., *"Masksembles for Uncertainty Estimation"*, CVPR 2021.
* Built upon [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models).
