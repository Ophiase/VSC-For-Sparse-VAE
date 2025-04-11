# 📚 [Variational Sparse Coding (VSC) - Statistical Course Project](https://ophiase.github.io/VSC-For-Sparse-VAE)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11.5+](https://img.shields.io/badge/Python-3.11.5%2B-yellow.svg)](https://www.python.org/)


This repository contains the implementation of **Variational Sparse Coding (VSC)**, a variant of Variational Autoencoders (VAEs) that introduces sparsity in the latent space using a **Spike-and-Slab prior**. The project is part of a university course we had on high-dimensional statistics.

## Overview

The goal of this project is to explore **VSC** as a generative model that addresses key limitations of traditional VAEs, such as posterior collapse and lack of interpretability in the latent space. The implementation includes:

- **AutoEncoder (AE)**: Baseline model for dimensionality reduction.
- - **Variational AutoEncoder (VAE)**: Probabilistic extension of AE.
- **Variational Sparse Coding (VSC)**: Sparse latent space with disentangled features.
    - **VSC with Warm-Up (VSC-WU)**: Improved training strategy to avoid posterior collapse (ie. all the information collapse to a few dimensions).

## Pipeline

To run the project, use the following commands:

```bash
make pip          # Install dependencies
make download     # Download datasets (MNIST, Fashion-MNIST)
make train        # Train the models
make graphics     # Generate visualizations
make update_web   # Build and update the online report
```

## Online Report

The full report, including theoretical insights, implementation details, and experimental results, is available here:  
[https://ophiase.github.io/VSC-For-Sparse-VAE](https://ophiase.github.io/VSC-For-Sparse-VAE)

## Datasets

- **MNIST**: Handwritten digits (0–9).
- **Fashion-MNIST**: Clothing items (e.g., T-shirts, trousers).

## Structure

- **`logic/`**: Core implementation (models, training, data loading).
- **`script/`**: Scripts for training, visualization, and dataset management.
- **`web/`**: Quarto-based report generation.

## License

This project is for educational purposes only.
