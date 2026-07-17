# Continuous 3-D Latent Diffusion

Official implementation of **Continuous 3-D Latent Diffusion for Medical Generation and Reconstruction**.

The framework combines a compact volumetric autoencoder, a 3-D latent diffusion model, and a coordinate-conditioned implicit decoder for:

- High-resolution CT and MRI volume generation
- Sparse-view CT reconstruction
- Accelerated MRI reconstruction
- Continuous, memory-efficient volumetric decoding

## Overview

The proposed decoder evaluates a shared latent feature grid and predicts voxel intensities from spatial coordinates. This avoids repeatedly passing overlapping sub-volumes through the complete decoder and enables full-volume processing on a single GPU.

The framework supports:

- CT volumes of up to \(512^3\) voxels
- MRI volumes of \(256^3\) voxels
- Unconditional 3-D volume generation
- Measurement-guided reconstruction using a frozen latent prior

## Repository status

The code, configuration files, and pretrained models are being prepared for release.

## Installation

Installation instructions will be added soon.

## Usage

Training and inference instructions will be added soon.

## Citation

```bibtex
@article{mellak2026continuous,
  title={Continuous 3-D Latent Diffusion for Medical Generation and Reconstruction},
  author={Mellak, Youness and De Paepe, Antoine and Visvikis, Dimitris and Bousse, Alexandre},
  year={2026}
}
