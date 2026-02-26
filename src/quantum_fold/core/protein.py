"""
protein.py
Protein sequence representation and energy models for lattice folding.

Implements three interaction models:
  1. **HP model** (Dill, 1985): H-H contact = −1, all others = 0
  2. **HP+ model**: H-H = −1, H-P = −0.5 (captures partial hydrophobic effect)
  3. **Miyazawa–Jernigan (MJ) potential** (Miyazawa & Jernigan, 1996):
     20×20 statistical contact energy matrix derived from PDB structures.

The HP model is the standard benchmark for quantum protein folding;
the MJ potential enables chemically realistic simulations for short peptides.

References:
  [1] Dill, Biochemistry 24, 1501 (1985)
  [2] Miyazawa & Jernigan, J. Mol. Biol. 256, 623 (1996)
  [3] Perdomo-Ortiz et al., Sci. Rep. 2, 571 (2012)
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Miyazawa–Jernigan statistical contact energies (simplified 20×20 matrix)
# ═══════════════════════════════════════════════════════════════════════════
# Values in units of kT (≈0.6 kcal/mol at 300 K).
# Source: Miyazawa & Jernigan, J. Mol. Biol. 256, 623 (1996), Table III.
# Only the lower triangle is stored; matrix is symmetric.

_MJ_RESIDUES = "ARNDCQEGHILKMFPSTWYV"
_MJ_RES_INDEX = {aa: i for i, aa in enumerate(_MJ_RESIDUES)}

# Condensed MJ contact energies (ε_ij), multiply by −1 to get attraction.
# We store the *negative* contact energy so that lower = more stable.
_MJ_MATRIX = np.array([
    # A      R      N      D      C      Q      E      G      H      I      L      K      M      F      P      S      T      W      Y      V
    [-2.72, -1.83, -1.84, -1.70, -3.07, -1.89, -1.51, -2.31, -2.41, -3.42, -3.21, -1.31, -2.89, -3.32, -2.13, -2.01, -2.32, -2.99, -2.78, -3.25],  # A
    [-1.83, -1.55, -1.97, -2.28, -1.77, -2.20, -2.79, -1.72, -2.47, -1.90, -2.04, -2.31, -1.71, -2.16, -1.64, -1.82, -1.80, -1.47, -1.41, -1.91],  # R
    [-1.84, -1.97, -1.68, -1.68, -2.01, -1.71, -1.51, -2.08, -2.41, -2.18, -1.88, -1.68, -1.80, -2.12, -1.55, -2.05, -2.11, -1.70, -1.76, -2.06],  # N
    [-1.70, -2.28, -1.68, -1.21, -2.18, -1.46, -1.02, -1.59, -2.01, -1.82, -1.51, -1.60, -1.52, -1.76, -1.16, -1.60, -1.60, -1.31, -1.45, -1.72],  # D
    [-3.07, -1.77, -2.01, -2.18, -4.53, -2.30, -1.82, -2.78, -2.84, -3.89, -3.74, -1.54, -3.52, -3.98, -2.47, -2.47, -2.69, -3.37, -3.08, -3.69],  # C
    [-1.89, -2.20, -1.71, -1.46, -2.30, -1.54, -1.42, -1.88, -1.98, -2.27, -2.12, -1.80, -1.79, -2.29, -1.52, -1.66, -1.74, -1.84, -1.80, -2.17],  # Q
    [-1.51, -2.79, -1.51, -1.02, -1.82, -1.42, -0.91, -1.22, -1.62, -1.69, -1.36, -1.68, -1.27, -1.53, -0.98, -1.32, -1.29, -0.96, -1.05, -1.55],  # E
    [-2.31, -1.72, -2.08, -1.59, -2.78, -1.88, -1.22, -2.24, -2.15, -2.67, -2.49, -1.27, -2.34, -2.70, -1.77, -2.01, -2.10, -2.46, -2.29, -2.59],  # G
    [-2.41, -2.47, -2.41, -2.01, -2.84, -1.98, -1.62, -2.15, -2.54, -3.14, -3.08, -1.79, -2.67, -3.34, -1.98, -2.11, -2.25, -2.78, -2.97, -2.98],  # H
    [-3.42, -1.90, -2.18, -1.82, -3.89, -2.27, -1.69, -2.67, -3.14, -4.16, -4.04, -1.56, -3.68, -4.03, -2.45, -2.39, -2.73, -3.52, -3.36, -3.98],  # I
    [-3.21, -2.04, -1.88, -1.51, -3.74, -2.12, -1.36, -2.49, -3.08, -4.04, -3.74, -1.36, -3.56, -3.94, -2.36, -2.19, -2.51, -3.40, -3.17, -3.75],  # L
    [-1.31, -2.31, -1.68, -1.60, -1.54, -1.80, -1.68, -1.27, -1.79, -1.56, -1.36, -1.01, -1.29, -1.63, -0.97, -1.32, -1.34, -0.82, -1.01, -1.50],  # K
    [-2.89, -1.71, -1.80, -1.52, -3.52, -1.79, -1.27, -2.34, -2.67, -3.68, -3.56, -1.29, -3.07, -3.56, -2.19, -2.15, -2.42, -3.08, -2.89, -3.39],  # M
    [-3.32, -2.16, -2.12, -1.76, -3.98, -2.29, -1.53, -2.70, -3.34, -4.03, -3.94, -1.63, -3.56, -4.19, -2.54, -2.42, -2.72, -3.53, -3.43, -3.82],  # F
    [-2.13, -1.64, -1.55, -1.16, -2.47, -1.52, -0.98, -1.77, -1.98, -2.45, -2.36, -0.97, -2.19, -2.54, -1.57, -1.57, -1.73, -2.13, -1.93, -2.33],  # P
    [-2.01, -1.82, -2.05, -1.60, -2.47, -1.66, -1.32, -2.01, -2.11, -2.39, -2.19, -1.32, -2.15, -2.42, -1.57, -1.67, -1.87, -2.08, -1.96, -2.30],  # S
    [-2.32, -1.80, -2.11, -1.60, -2.69, -1.74, -1.29, -2.10, -2.25, -2.73, -2.51, -1.34, -2.42, -2.72, -1.73, -1.87, -2.12, -2.32, -2.16, -2.58],  # T
    [-2.99, -1.47, -1.70, -1.31, -3.37, -1.84, -0.96, -2.46, -2.78, -3.52, -3.40, -0.82, -3.08, -3.53, -2.13, -2.08, -2.32, -3.21, -2.99, -3.29],  # W
    [-2.78, -1.41, -1.76, -1.45, -3.08, -1.80, -1.05, -2.29, -2.97, -3.36, -3.17, -1.01, -2.89, -3.43, -1.93, -1.96, -2.16, -2.99, -2.78, -3.13],  # Y
    [-3.25, -1.91, -2.06, -1.72, -3.69, -2.17, -1.55, -2.59, -2.98, -3.98, -3.75, -1.50, -3.39, -3.82, -2.33, -2.30, -2.58, -3.29, -3.13, -3.79],  # V
], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Energy model interface
# ═══════════════════════════════════════════════════════════════════════════

class EnergyModel:
    """Base class for pairwise contact energy models."""

    def contact_energy(self, res_i: str, res_j: str) -> float:
        """Return the energy of a topological contact between residues i and j."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class HPModel(EnergyModel):
    """
    Standard HP (hydrophobic–polar) model.
    H-H contact = −1, all others = 0.
    """
    def contact_energy(self, res_i: str, res_j: str) -> float:
        if res_i == "H" and res_j == "H":
            return -1.0
        return 0.0

    @property
    def name(self) -> str:
        return "HP"


class HPPlusModel(EnergyModel):
    """
    Extended HP+ model.
    H-H contact = −1, H-P contact = −0.5, P-P = 0.
    """
    def contact_energy(self, res_i: str, res_j: str) -> float:
        if res_i == "H" and res_j == "H":
            return -1.0
        if (res_i == "H" and res_j == "P") or (res_i == "P" and res_j == "H"):
            return -0.5
        return 0.0

    @property
    def name(self) -> str:
        return "HP+"


class MJModel(EnergyModel):
    """
    Miyazawa–Jernigan statistical contact potential.
    Uses the 20×20 matrix from Miyazawa & Jernigan (1996).
    Residues must be standard one-letter amino acid codes.
    """
    def contact_energy(self, res_i: str, res_j: str) -> float:
        i_idx = _MJ_RES_INDEX.get(res_i.upper())
        j_idx = _MJ_RES_INDEX.get(res_j.upper())
        if i_idx is None or j_idx is None:
            return 0.0  # unknown residue
        return float(_MJ_MATRIX[i_idx, j_idx])

    @property
    def name(self) -> str:
        return "MJ"


# ═══════════════════════════════════════════════════════════════════════════
# Protein class
# ═══════════════════════════════════════════════════════════════════════════

class Protein:
    """
    Represents a protein sequence with an associated energy model.

    Parameters
    ----------
    sequence : str
        Residue sequence. For HP model, characters must be 'H' or 'P'.
        For MJ model, standard one-letter amino acid codes are accepted.
    energy_model : str or EnergyModel
        "HP" (default), "HP+", "MJ", or a custom EnergyModel instance.
    """

    def __init__(self, sequence: str, energy_model: str | EnergyModel = "HP"):
        self.sequence = sequence.upper()
        self.n = len(self.sequence)

        if isinstance(energy_model, str):
            model_map = {"HP": HPModel, "HP+": HPPlusModel, "MJ": MJModel}
            cls = model_map.get(energy_model.upper())
            if cls is None:
                raise ValueError(f"Unknown energy model '{energy_model}'. Choose from {list(model_map)}")
            self.energy_model: EnergyModel = cls()
        else:
            self.energy_model = energy_model

    def is_hp(self) -> bool:
        """Check if the sequence is a valid HP sequence."""
        return set(self.sequence).issubset({"H", "P"})

    def get_interaction_energy(self, idx_i: int, idx_j: int, dist_sq: int) -> float:
        """
        Returns the pairwise interaction energy between beads i and j.

        Only non-bonded (|i−j| ≥ 2) topological contacts (dist² = 1) contribute.
        """
        if abs(idx_i - idx_j) <= 1:
            return 0.0
        if dist_sq == 1:
            return self.energy_model.contact_energy(
                self.sequence[idx_i], self.sequence[idx_j]
            )
        return 0.0

    def evaluate_energy(
        self,
        coords: List[np.ndarray],
        collision_penalty: float = 1000.0,
    ) -> float:
        """
        Evaluate the total energy of a lattice conformation.

        Parameters
        ----------
        coords : list of np.ndarray
            3D integer coordinates for each bead.
        collision_penalty : float
            Penalty added for each pair of beads that overlap (same site).

        Returns
        -------
        energy : float
            Total energy = Σ contact energies + Σ collision penalties.
        """
        if len(coords) != self.n:
            raise ValueError(
                f"Expected {self.n} coordinates, got {len(coords)}"
            )

        energy = 0.0
        for i in range(self.n):
            for j in range(i + 2, self.n):
                diff = coords[i] - coords[j]
                d2 = int(np.sum(diff * diff))

                # Collision (overlap)
                if d2 == 0:
                    energy += collision_penalty

                # Contact interaction
                energy += self.get_interaction_energy(i, j, d2)

        return energy

    def evaluate_energy_decomposed(
        self,
        coords: List[np.ndarray],
        collision_penalty: float = 1000.0,
    ) -> Dict[str, float]:
        """
        Return decomposed energy contributions.

        Returns
        -------
        dict with keys:
          "contact"   : sum of contact energies
          "collision"  : sum of collision penalties
          "total"      : contact + collision
          "n_contacts" : number of topological contacts
          "n_collisions" : number of overlapping pairs
        """
        if len(coords) != self.n:
            raise ValueError(f"Expected {self.n} coordinates, got {len(coords)}")

        contact_e = 0.0
        collision_e = 0.0
        n_contacts = 0
        n_collisions = 0

        for i in range(self.n):
            for j in range(i + 2, self.n):
                diff = coords[i] - coords[j]
                d2 = int(np.sum(diff * diff))

                if d2 == 0:
                    collision_e += collision_penalty
                    n_collisions += 1

                e_ij = self.get_interaction_energy(i, j, d2)
                if e_ij != 0.0:
                    contact_e += e_ij
                    n_contacts += 1

        return {
            "contact": contact_e,
            "collision": collision_e,
            "total": contact_e + collision_e,
            "n_contacts": n_contacts,
            "n_collisions": n_collisions,
        }

    def max_possible_contacts(self) -> int:
        """
        Upper bound on the number of H-H contacts for this sequence.
        Used for computing approximation ratios and for branch-and-bound pruning.

        For a sequence of length N with n_H hydrophobic residues,
        the theoretical maximum is min(n_H * (n_H − 1) / 2, 2*N − 6)
        (limited by lattice geometry for 3D cubic lattice).
        """
        if isinstance(self.energy_model, (HPModel, HPPlusModel)):
            n_h = self.sequence.count("H")
            geometric_max = 2 * self.n - 6 if self.n >= 4 else 0
            pair_max = n_h * (n_h - 1) // 2
            return min(geometric_max, pair_max)
        return self.n * (self.n - 1) // 2  # generic upper bound

    def __repr__(self) -> str:
        return f"Protein('{self.sequence}', model={self.energy_model.name})"
