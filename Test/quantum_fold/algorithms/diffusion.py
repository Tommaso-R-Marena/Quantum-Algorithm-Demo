"""
diffusion.py
Denoising Diffusion Backbone Sampler for protein structure generation.

Implements a simple score-based generative model over Calpha inter-residue
distance matrices. The model learns to denoise progressively corrupted
distance matrices, producing plausible backbone structures.

Components:
  1. Forward process: adds Gaussian noise to native distance matrix
  2. Denoising MLP: predicts noise to remove at each step
  3. Reverse process: iterative denoising from random noise
  4. Distance geometry: reconstructs 3D coords from distance matrix (MDS)

This is a lightweight implementation suitable for small proteins (< 50 residues).
For production use, replace the MLP with a proper SE(3)-equivariant network.

References:
  [1] Ho et al., NeurIPS 2020 — DDPM
  [2] Wu et al., arXiv:2209.15171 (2022) — diffusion for protein
  [3] Anand & Huang, NeurIPS Workshop (2018) — distance matrix generation
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple
from ..core.backbone import ca_distance_matrix, kabsch_rmsd


class DiffusionBackboneSampler:
    """
    Denoising diffusion model over Calpha distance matrices.

    The model works on the upper triangle of the distance matrix
    (flattened into a 1D vector).

    Parameters
    ----------
    n_residues : int
    n_timesteps : int (default 50)
    beta_start : float (default 0.0001)
    beta_end : float (default 0.02)
    hidden_dim : int (default 128)
    seed : int
    """

    def __init__(
        self,
        n_residues: int,
        n_timesteps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        hidden_dim: int = 128,
        seed: int = 42,
    ):
        self.n_residues = n_residues
        self.n_timesteps = n_timesteps
        self.dim = n_residues * (n_residues - 1) // 2  # upper triangle

        # Noise schedule
        self.betas = np.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

        # MLP parameters (randomly initialised for now)
        self.rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim

        # Simple 2-layer MLP: input → hidden → output
        # Input: flattened distance matrix + timestep embedding
        input_dim = self.dim + 1  # +1 for timestep
        self.W1 = self.rng.normal(0, 0.01, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.normal(0, 0.01, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = self.rng.normal(0, 0.01, (hidden_dim, self.dim))
        self.b3 = np.zeros(self.dim)

        self._trained = False

    def _distmat_to_vector(self, D: np.ndarray) -> np.ndarray:
        """Upper triangle of distance matrix → flat vector."""
        n = self.n_residues
        indices = np.triu_indices(n, k=1)
        return D[indices]

    def _vector_to_distmat(self, v: np.ndarray) -> np.ndarray:
        """Flat vector → symmetric distance matrix."""
        n = self.n_residues
        D = np.zeros((n, n))
        indices = np.triu_indices(n, k=1)
        D[indices] = v
        D += D.T
        return D

    def _forward_noise(
        self, x0: np.ndarray, t: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise at timestep t: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps."""
        alpha_bar = self.alpha_bars[t]
        eps = self.rng.normal(0, 1, x0.shape)
        x_t = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * eps
        return x_t, eps

    def _predict_noise(self, x_t: np.ndarray, t: int) -> np.ndarray:
        """MLP forward pass to predict noise."""
        t_embed = np.array([t / self.n_timesteps])
        inp = np.concatenate([x_t, t_embed])

        # Layer 1
        h = inp @ self.W1 + self.b1
        h = np.maximum(h, 0)  # ReLU

        # Layer 2
        h = h @ self.W2 + self.b2
        h = np.maximum(h, 0)

        # Output
        out = h @ self.W3 + self.b3
        return out

    def train(
        self,
        native_distmats: List[np.ndarray],
        n_epochs: int = 100,
        lr: float = 0.001,
    ):
        """
        Train the denoising model on native distance matrices.

        Parameters
        ----------
        native_distmats : list of np.ndarray
            Each shape (n_residues, n_residues).
        n_epochs : int
        lr : float
        """
        x0_samples = [self._distmat_to_vector(D) for D in native_distmats]

        for epoch in range(n_epochs):
            total_loss = 0.0
            for x0 in x0_samples:
                t = self.rng.integers(0, self.n_timesteps)
                x_t, eps = self._forward_noise(x0, t)

                # Forward pass
                eps_pred = self._predict_noise(x_t, t)

                # MSE loss
                loss = np.mean((eps_pred - eps) ** 2)
                total_loss += loss

                # Compute gradients (manual backprop for simple MLP)
                self._backward_step(x_t, t, eps, eps_pred, lr)

            if epoch % max(1, n_epochs // 5) == 0:
                avg_loss = total_loss / max(len(x0_samples), 1)
                print(f"  Epoch {epoch}: loss={avg_loss:.6f}")

        self._trained = True

    def _backward_step(
        self,
        x_t: np.ndarray, t: int,
        eps: np.ndarray, eps_pred: np.ndarray,
        lr: float,
    ):
        """Simple gradient descent step (manual backprop)."""
        t_embed = np.array([t / self.n_timesteps])
        inp = np.concatenate([x_t, t_embed])

        # Forward with saved intermediates
        z1 = inp @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0)
        out = h2 @ self.W3 + self.b3

        # Backprop
        d_out = 2 * (out - eps) / len(eps)

        # Layer 3
        d_W3 = np.outer(h2, d_out)
        d_b3 = d_out
        d_h2 = d_out @ self.W3.T

        # ReLU
        d_z2 = d_h2 * (z2 > 0)

        # Layer 2
        d_W2 = np.outer(h1, d_z2)
        d_b2 = d_z2
        d_h1 = d_z2 @ self.W2.T

        # ReLU
        d_z1 = d_h1 * (z1 > 0)

        # Layer 1
        d_W1 = np.outer(inp, d_z1)
        d_b1 = d_z1

        # Update
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        """
        Generate distance matrices by iterative denoising.

        Returns
        -------
        list of np.ndarray, each shape (n_residues, n_residues)
        """
        results = []

        for _ in range(n_samples):
            # Start from pure noise
            x = self.rng.normal(0, 1, self.dim)

            # Reverse process
            for t in range(self.n_timesteps - 1, -1, -1):
                eps_pred = self._predict_noise(x, t)

                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]

                # Denoising step
                x = (1.0 / np.sqrt(alpha)) * (
                    x - (self.betas[t] / np.sqrt(1 - alpha_bar)) * eps_pred
                )

                # Add noise (except at t=0)
                if t > 0:
                    noise = self.rng.normal(0, 1, self.dim)
                    x += np.sqrt(self.betas[t]) * noise

            # Ensure distances are positive
            x = np.abs(x)

            # Convert to distance matrix
            D = self._vector_to_distmat(x)
            results.append(D)

        return results

    def sample_with_prior(
        self,
        sequence: str,
        n_samples: int = 5,
    ) -> List[np.ndarray]:
        """
        Generate distance matrices with sequence-aware priors.

        Uses the expected Calpha distances based on secondary structure
        prediction as the starting point, with noise added.
        """
        from ..core.residue import predict_secondary_structure

        ss = predict_secondary_structure(sequence)
        n = len(sequence)

        # Build prior distance matrix from SS
        D_prior = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                seq_sep = j - i
                # Base distance from sequence separation (random coil)
                d_base = 3.8 * np.sqrt(seq_sep)

                # SS corrections
                if all(ss[k] == "H" for k in range(i, j + 1)):
                    # Alpha helix: 1.5 Å per residue along axis
                    d_base = min(d_base, 1.5 * seq_sep)
                elif all(ss[k] == "E" for k in range(i, j + 1)):
                    # Beta strand: 3.3 Å per residue
                    d_base = min(d_base, 3.3 * seq_sep)

                D_prior[i, j] = d_base
                D_prior[j, i] = d_base

        results = []
        for _ in range(n_samples):
            D_noisy = D_prior + self.rng.normal(0, 1.5, (n, n))
            D_noisy = np.abs(D_noisy)
            D_noisy = (D_noisy + D_noisy.T) / 2
            np.fill_diagonal(D_noisy, 0)
            results.append(D_noisy)

        return results


def distance_to_coords(
    D: np.ndarray,
    n_dims: int = 3,
) -> np.ndarray:
    """
    Reconstruct 3D coordinates from a distance matrix using
    classical Multidimensional Scaling (MDS / Gram matrix method).

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Pairwise distance matrix.
    n_dims : int (default 3)

    Returns
    -------
    coords : np.ndarray, shape (n, n_dims)
    """
    n = len(D)
    D2 = D ** 2

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Gram matrix (double centering)
    B = -0.5 * H @ D2 @ H

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Take top n_dims positive eigenvalues
    idx = np.argsort(eigenvalues)[::-1][:n_dims]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp negative eigenvalues to 0
    eigenvalues = np.maximum(eigenvalues, 0)

    # Reconstruct coordinates
    coords = eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]

    return coords
