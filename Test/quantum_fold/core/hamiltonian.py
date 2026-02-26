"""
hamiltonian.py
Constructs the QUBO / Ising Hamiltonian for lattice protein folding.

Two approaches are supported:
  1. **Diagonal Hamiltonian** (for hybrid CVaR-VQE / QAOA):
     The cost function is diagonal in the computational basis.
     We measure bitstrings and evaluate the classical cost function.
     This is the standard approach from Robert et al. (2021).

  2. **Pauli-Z Hamiltonian** (for operator-based VQE):
     The diagonal cost vector is decomposed into a sum of Pauli-Z
     products via Walsh–Hadamard transform. This produces a
     qml.Hamiltonian that can be used with analytic gradients.

References:
  [1] Robert et al., npj Quantum Inf. 7, 38 (2021)
  [2] Lucas, Front. Phys. 2, 5 (2014)  — QUBO formulations
  [3] Hadfield et al., Algorithms 12, 34 (2019)  — QAOA
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict
from itertools import product

from .protein import Protein
from .lattice import CubicLattice
from .encoding import (
    bitstring_to_coords,
    enumerate_all_bitstrings,
    compute_classical_cost_vector,
)


class QUBOBuilder:
    """
    Constructs the QUBO matrix and Ising Hamiltonian for protein folding.

    The cost function E(z) maps each computational-basis state |z⟩ to
    a real energy. Since this is diagonal, it can be written as:

        H = Σ_α  h_α  Π_{i ∈ α} Z_i

    where the sum runs over all subsets α of qubits and h_α are real
    coefficients determined by the Walsh–Hadamard transform of the
    cost vector.

    Parameters
    ----------
    protein : Protein
        The protein instance.
    bits_per_link : int
        2 (default) or 3.
    collision_penalty : float
        Penalty weight for overlapping beads.
    """

    def __init__(
        self,
        protein: Protein,
        bits_per_link: int = 2,
        collision_penalty: float = 100.0,
    ):
        self.protein = protein
        self.bits_per_link = bits_per_link
        self.collision_penalty = collision_penalty
        self.n_qubits = CubicLattice.n_qubits(protein.n, bits_per_link)
        self._cost_vector: Optional[np.ndarray] = None
        self._pauli_coeffs: Optional[Dict[Tuple[int, ...], float]] = None

    @property
    def cost_vector(self) -> np.ndarray:
        """Lazy-computed cost vector (diagonal of the Hamiltonian)."""
        if self._cost_vector is None:
            self._cost_vector = compute_classical_cost_vector(
                n_beads=self.protein.n,
                sequence=self.protein.sequence,
                collision_penalty=self.collision_penalty,
                bits_per_link=self.bits_per_link,
                energy_model=self.protein.energy_model.name,
            )
        return self._cost_vector

    def ground_state_energy(self) -> float:
        """Return the minimum energy (ground state)."""
        return float(np.min(self.cost_vector))

    def ground_state_degeneracy(self) -> int:
        """Number of bitstrings achieving the ground-state energy."""
        e_min = self.ground_state_energy()
        return int(np.sum(np.abs(self.cost_vector - e_min) < 1e-10))

    def spectral_gap(self) -> float:
        """Energy gap between ground state and first excited state."""
        unique_energies = np.unique(self.cost_vector)
        if len(unique_energies) < 2:
            return 0.0
        return float(unique_energies[1] - unique_energies[0])

    def energy_landscape(self) -> Dict[str, np.ndarray]:
        """
        Return the full energy landscape for analysis.

        Returns
        -------
        dict with:
          "energies"  : sorted unique energies
          "counts"    : degeneracy of each energy level
          "min"       : ground-state energy
          "gap"       : spectral gap
        """
        energies, counts = np.unique(self.cost_vector, return_counts=True)
        return {
            "energies": energies,
            "counts": counts,
            "min": float(energies[0]),
            "gap": float(energies[1] - energies[0]) if len(energies) > 1 else 0.0,
        }

    # --- Walsh–Hadamard decomposition into Pauli-Z terms ------------------

    def pauli_decomposition(
        self, threshold: float = 1e-12
    ) -> Dict[Tuple[int, ...], float]:
        """
        Decompose the diagonal cost function into a sum of Pauli-Z products.

        The cost vector f(z) ∈ ℝ^{2^n} is expanded as:

            f(z) = Σ_S  ĥ(S)  Π_{i∈S} (-1)^{z_i}

        where S ⊆ {0, …, n−1} and ĥ(S) is computed via the Walsh–Hadamard
        transform: ĥ = H_n · f / 2^n.

        Returns
        -------
        coeffs : dict mapping qubit-index tuples to float coefficients.
            Empty tuple () → identity (energy offset).
            (i,) → Z_i coefficient.
            (i, j) → Z_i Z_j coefficient, etc.
        """
        if self._pauli_coeffs is not None:
            return self._pauli_coeffs

        n = self.n_qubits
        N = 2 ** n
        f = self.cost_vector.copy()

        # Fast Walsh–Hadamard transform (in-place)
        h = 1
        while h < N:
            for i in range(0, N, h * 2):
                for j in range(i, i + h):
                    x = f[j]
                    y = f[j + h]
                    f[j] = x + y
                    f[j + h] = x - y
            h *= 2
        f /= N  # normalise

        # Extract non-zero coefficients
        coeffs = {}
        for idx in range(N):
            if abs(f[idx]) > threshold:
                # idx encodes the subset S via its binary representation
                S = tuple(b for b in range(n) if (idx >> (n - 1 - b)) & 1)
                coeffs[S] = float(f[idx])

        self._pauli_coeffs = coeffs
        return coeffs

    def build_pennylane_hamiltonian(self):
        """
        Build a PennyLane qml.Hamiltonian from the Pauli decomposition.

        Returns
        -------
        H : qml.Hamiltonian
            Sum of Pauli-Z tensor products.
        """
        import pennylane as qml

        decomp = self.pauli_decomposition()
        coeffs = []
        ops = []

        for qubits, coeff in decomp.items():
            coeffs.append(coeff)
            if len(qubits) == 0:
                ops.append(qml.Identity(0))
            else:
                paulis = [qml.PauliZ(q) for q in qubits]
                op = paulis[0]
                for p in paulis[1:]:
                    op = op @ p
                ops.append(op)

        return qml.Hamiltonian(coeffs, ops)

    def build_qubo_matrix(self) -> np.ndarray:
        """
        Build the QUBO matrix Q such that E(x) = x^T Q x + const.

        Variables x_i ∈ {0, 1} correspond to the bitstring.
        This is useful for quantum annealing and classical QUBO solvers.

        Returns
        -------
        Q : np.ndarray, shape (n_qubits, n_qubits)
        """
        decomp = self.pauli_decomposition()
        n = self.n_qubits
        Q = np.zeros((n, n), dtype=np.float64)

        # Transform: Z_i = 1 − 2x_i, so Π Z's become polynomial in x's.
        # For up to 2-local terms (dominant for lattice problems):
        for qubits, coeff in decomp.items():
            if len(qubits) == 0:
                continue  # constant offset
            elif len(qubits) == 1:
                i = qubits[0]
                Q[i, i] += -2 * coeff  # Z_i = 1 − 2x_i
            elif len(qubits) == 2:
                i, j = qubits
                # Z_i Z_j = (1−2x_i)(1−2x_j) = 1 − 2x_i − 2x_j + 4x_i x_j
                Q[i, i] += -2 * coeff
                Q[j, j] += -2 * coeff
                Q[i, j] += 4 * coeff
            # Higher-order terms are present but cannot be directly
            # represented in standard QUBO. They can be reduced via
            # auxiliary variables (not implemented here for clarity).

        return Q
