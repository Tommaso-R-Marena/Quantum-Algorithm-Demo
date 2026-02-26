# Quantum Protein Folding Framework (QPF)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CI: Pytest](https://github.com/user/Quantum-Algorithm-Demo/actions/workflows/pytest.yml/badge.svg)](https://github.com/user/Quantum-Algorithm-Demo/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/user/Quantum-Algorithm-Demo/blob/main/notebooks/quantum_protein_folding_demo.ipynb)

A research-grade framework for **real-world protein structure prediction** using a hybrid quantum-classical pipeline. This project implements **Quantum Fragment Assembly (QFA)**, a novel approach that combines variational quantum algorithms with classical optimization to solve the combinatorial challenge of protein folding.

## 🌟 Overview

The Quantum Protein Folding Framework (QPF) bridges the gap between theoretical quantum chemistry and practical bioinformatics. By decomposing protein backbones into overlapping fragments and encoding their conformations into a Quadratic Unconstrained Binary Optimization (QUBO) problem, QPF leverages the power of VQE and QAOA while maintaining high structural accuracy through local refinement.

### Key Innovations
*   **Quantum Fragment Assembly (QFA)**: Efficiently maps the backbone conformation search space to a qubit Hamiltonian.
*   **Hybrid Optimization**: Combines Quantum Variational Algorithms with a high-performance **L-BFGS-B** classical refiner.
*   **Analytical Gradients**: Uses custom derivatives for the 8-term coarse-grained force field (DFIRE2, Lennard-Jones, etc.) to achieve AlphaFold-level precision.
*   **Confidence Quantification**: Implements a distance-weighted neighborhood density heuristic to provide pLDDT-style confidence scores.

## 🚀 Quick Start

### Interactive Demo
Experience the framework immediately without local setup:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/user/Quantum-Algorithm-Demo/blob/main/notebooks/quantum_protein_folding_demo.ipynb)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/user/Quantum-Algorithm-Demo.git
cd Quantum-Algorithm-Demo

# Install dependencies
pip install numpy scipy matplotlib pennylane plotly pytest
```

### Running a Prediction

Predict the structure of Chignolin using Simulated Annealing (classical baseline):

```bash
PYTHONPATH=./src python -m quantum_fold.main --mode real --benchmark chignolin --method sa
```

Run a Quantum Variational Eigensolver (VQE) experiment:

```bash
PYTHONPATH=./src python -m quantum_fold.main --mode real --seq YYDPETGTWY --method vqe --shots 300
```

## 🏗 Architecture

The framework is organized for scalability and clarity:

*   **`src/`**: The core package logic.
    *   **`core/`**: Physical models, including the 8-term force field, NeRF coordinate generation, and fragment library logic.
    *   **`algorithms/`**: The hybrid pipeline, implementing Quantum Fragment Assembly (VQE/QAOA), diffusion priors, and L-BFGS-B refinement.
    *   **`utils/`**: Bioinformatics utilities for PDB I/O, TM-score/RMSD metrics, and publication-quality 3D visualizations.
*   **`tests/`**: Rigorous unit test suite.
*   **`notebooks/`**: Interactive examples and tutorials.

## 📊 Visualizations

QPF generates high-resolution 3D visualizations and interactive HTML reports. Predicted structures are colored by their **pLDDT confidence scores**, following the standard AlphaFold convention:
*   🔵 **Blue** (>90): Very high confidence
*   🌐 **Cyan** (70-90): Confident
*   🟡 **Yellow** (50-70): Low confidence
*   🟠 **Orange** (<50): Very low confidence

## 🧪 Testing

```bash
PYTHONPATH=./src python -m pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Developed for the intersection of Quantum Computing and Structural Biology.*
