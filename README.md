# Image Generation with Diffusion Transformers (DiT/SiT)

This repository contains an implementation and evaluation of Diffusion Transformers, specifically the SiT (Scalable Image Transformer) variant, as described in the research paper "Representation Entanglement for 
Generation: Training Diffusion Transformers Is Much Easier Than You Think". The project explores replacing the traditional U-Net backbone in diffusion models with a Vision Transformer (ViT) architecture, enhanced by
REG (Representation Enhancement Guidance).


# 🚀 Key Features

## SiT Architecture Implementation: 
A denoising network built on a Scalable Image Transformer backbone instead of a standard U-Net.

## REG Integration:
Incorporates Representation Enhancement Guidance to stabilize training and improve convergence by aligning internal transformer representations with a vision foundation model.

## Adaptive Hardware Scaling: 
Configurable model depth and hidden dimensions to study the effect of scaling on image fidelity.

## Dual-Baseline Comparison:
Includes a complete U-Net diffusion baseline for side-by-side qualitative and quantitative analysis.

## Euler-Maruyama Sampling: 
Implements the $v$-prediction objective with a discrete Euler-Maruyama sampler for high-quality image synthesis.


# 📂 Repository Structure

```
├── Diffusion-Transformers.ipynb # Implementation of SiT, REG, and U-Net models
├── outputs_improved/            # Generated samples, loss curves, and evaluation metrics
├── data/                        # CIFAR-10 subset (Cats and Dogs)
└── Report.pdf                   # Comparative research analysis
```

# ⚙️ Installation & Setup

## 1. Prerequisites
Environment: Python 3.10+ (Google Colab T4 GPU or better recommended).
Dataset: A subset of CIFAR-10 restricted to the "cat" and "dog" classes.

## 2. Install Dependencies
pip install -q torch torchvision torch-fidelity einops tqdm matplotlib


# 🛠️ Technical Workflow

## 1. Model Design
The system employs a SimpleVAE to encode $32 \times 32$ images into a latent space. The SiT encoder then processes these latents by splitting them into patches and applying positional encodings.

## 2. REG Mechanism
The training objective is regularized by the REG module. It uses a small frozen CNN backbone (Vision Foundation) to produce class tokens and feature maps. The SiT model is trained to minimize prediction loss while maximizing representational alignment with these foundation features.

## 3. Sampling Logic
Images are generated through iterative denoising. The system supports Classifier-Free Guidance (CFG) on the class token to amplify specific features during the reverse diffusion process.


## 📊 Performance & Evaluation

The models were evaluated on 500–1000 generated samples from the CIFAR-10 subset (Cats vs. Dogs). **Lower FID** and **Higher IS** indicate better generative performance.

| Model | FID (Lower is better) | Inception Score (IS) | Observation |
| :--- | :---: | :---: | :--- |
| **SiT + REG** | ~401.96 | ~1.94 | Superior global coherence and structural alignment. |
| **U-Net Baseline** | ~442.46 | ~1.68 | Faster training speed but significantly less visual detail. |


# Training Stability
## SiT+REG: 
Showed faster convergence in early epochs due to the representation alignment regularizer.

## U-Net:
Exhibited typical diffusion training behavior but struggled with fine-grained categorical diversity on limited data.

# Conclusion
This reproduction confirms that Transformers are highly scalable backbones for diffusion. The integration of REG is critical; it significantly eases the training of Diffusion Transformers by providing a 
semantic "anchor," allowing the model to focus on generating complex structures rather than learning representations from scratch.

# 🎓 Author
M Abdurrahman Khan Department of Computer Science
National University of Computer and Emerging Sciences (FAST), Pakistan
Contact: {i221148}@nu.edu.pk
