"""
qaoa_runner.py
QAOA (Quantum Approximate Optimization Algorithm) for lattice protein folding.

Implements QAOA with the diagonal cost Hamiltonian H_C (from hamiltonian.py)
and a standard X-mixer H_M = Σ X_i. The algorithm alternates layers of
exp(-iγ H_C) and exp(-iβ H_M) for p layers, then measures and evaluates.

For small systems (n_qubits ≤ 12), we construct the full Pauli-Z
Hamiltonian from the Walsh–Hadamard decomposition. For larger systems,
we fall back to the hybrid CVaR approach (measure + classical cost).

References:
  [1] Farhi et al., arXiv:1411.4028 (2014)  — original QAOA
  [2] Hadfield et al., Algorithms 12, 34 (2019)  — QAOA for constrained
  [3] Robert et al., npj Quantum Inf. 7, 38 (2021)
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import Dict, List, Optional
import time

from ..core.encoding import bitstring_to_coords
from ..core.protein import Protein
from ..core.lattice import CubicLattice
from ..core.hamiltonian import QUBOBuilder


class QAOAFolder:
    """
    QAOA-based protein folder.

    Parameters
    ----------
    protein : Protein
    params : dict
        Configuration:
          p_layers      : int   — QAOA depth (default 3)
          shots         : int   — measurement shots (default 1000)
          max_iter      : int   — optimizer iterations (default 100)
          bits_per_link : int   — 2 (default) or 3
          cvar_alpha    : float — CVaR quantile (default 0.10)
          collision_penalty : float (default 100.0)
          optimizer     : str   — "COBYLA" or "SPSA" (default "COBYLA")
          seed          : int   — random seed
    """

    def __init__(self, protein: Protein, params: Dict):
        self.protein = protein
        self.params = params
        self.bits_per_link = params.get("bits_per_link", 2)
        self.n_qubits = CubicLattice.n_qubits(protein.n, self.bits_per_link)
        self.shots = params.get("shots", 1000)
        self.collision_penalty = params.get("collision_penalty", 100.0)
        self.seed = params.get("seed", None)

        if self.n_qubits == 0:
            raise ValueError("Sequence too short for QAOA")

        self.dev = qml.device(
            "default.qubit", wires=self.n_qubits, shots=self.shots
        )

        # Build Hamiltonian
        self.qubo = QUBOBuilder(
            protein,
            bits_per_link=self.bits_per_link,
            collision_penalty=self.collision_penalty,
        )

        # Try to build the PennyLane Hamiltonian for operator QAOA
        self._use_operator_mode = self.n_qubits <= 16
        if self._use_operator_mode:
            try:
                self.hamiltonian = self.qubo.build_pennylane_hamiltonian()
            except Exception:
                self._use_operator_mode = False
                self.hamiltonian = None

    def qaoa_circuit(self, gammas: np.ndarray, betas: np.ndarray):
        """
        QAOA ansatz: p layers of cost + mixer unitaries.

        |ψ(γ,β)⟩ = Π_{l=1}^{p} exp(-iβ_l H_M) exp(-iγ_l H_C) |+⟩^n

        Parameters
        ----------
        gammas : array of shape (p,)
        betas  : array of shape (p,)
        """
        p = len(gammas)

        # Initial state: |+⟩^n
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)

        for layer in range(p):
            # Cost unitary: exp(-iγ H_C)
            if self._use_operator_mode and self.hamiltonian is not None:
                qml.ApproxTimeEvolution(
                    self.hamiltonian, gammas[layer], n=1
                )
            else:
                # Fallback: diagonal cost via phase oracle
                self._cost_layer_diagonal(gammas[layer])

            # Mixer unitary: exp(-iβ H_M) where H_M = Σ X_i
            for i in range(self.n_qubits):
                qml.RX(2 * betas[layer], wires=i)

    def _cost_layer_diagonal(self, gamma: float):
        """
        Apply the cost layer using RZ gates for diagonal terms.

        For a diagonal Hamiltonian H = Σ h_S Π Z_i, we have:
          exp(-iγ H) = Π exp(-iγ h_S Π Z_i)

        Single-Z terms → RZ; ZZ terms → CNOT-RZ-CNOT decomposition.
        """
        decomp = self.qubo.pauli_decomposition()

        for qubits, coeff in decomp.items():
            if len(qubits) == 0:
                # Global phase (identity) — skip
                continue
            elif len(qubits) == 1:
                qml.RZ(2 * gamma * coeff, wires=qubits[0])
            elif len(qubits) == 2:
                q0, q1 = qubits
                qml.CNOT(wires=[q0, q1])
                qml.RZ(2 * gamma * coeff, wires=q1)
                qml.CNOT(wires=[q0, q1])
            else:
                # Higher-order terms: CNOT ladder
                for k in range(len(qubits) - 1):
                    qml.CNOT(wires=[qubits[k], qubits[k + 1]])
                qml.RZ(2 * gamma * coeff, wires=qubits[-1])
                for k in range(len(qubits) - 2, -1, -1):
                    qml.CNOT(wires=[qubits[k], qubits[k + 1]])

    def run(self) -> Dict:
        """
        Run QAOA optimisation.

        Returns
        -------
        dict with keys:
          best_energy, best_coords, best_fold, valid_rate,
          energies (cost history), optimal_gammas, optimal_betas,
          time_seconds, ground_state_energy, approximation_ratio
        """
        p_layers = self.params.get("p_layers", 3)
        max_iter = self.params.get("max_iter", 100)
        cvar_alpha = self.params.get("cvar_alpha", 0.10)

        rng = np.random.default_rng(self.seed)

        # Initial parameters
        init_gammas = pnp.array(
            rng.uniform(0, np.pi, p_layers), requires_grad=True
        )
        init_betas = pnp.array(
            rng.uniform(0, np.pi, p_layers), requires_grad=True
        )

        # Combined parameter vector
        init_params = pnp.concatenate([init_gammas, init_betas])

        start_time = time.time()

        print(f"┌──────────────────────────────────────────────────")
        print(f"│ QAOA  N={self.protein.n}  Qubits={self.n_qubits}  p={p_layers}")
        print(f"│ Shots={self.shots}  α={cvar_alpha}")
        print(f"│ Operator mode: {self._use_operator_mode}")
        print(f"└──────────────────────────────────────────────────")

        # Cost function
        history = []

        def cost_fn(combined_params):
            gammas = combined_params[:p_layers]
            betas = combined_params[p_layers:]

            @qml.qnode(self.dev)
            def circuit():
                self.qaoa_circuit(gammas, betas)
                return qml.sample()

            samples = circuit()
            if samples.ndim == 1:
                if self.n_qubits == 1:
                    samples = samples.reshape(-1, 1)
                else:
                    samples = samples.reshape(1, -1)

            energies = []
            for sample in samples:
                coords = bitstring_to_coords(
                    sample, self.protein.n, self.bits_per_link
                )
                e = self.protein.evaluate_energy(
                    coords, collision_penalty=self.collision_penalty
                )
                energies.append(float(e))

            energies = np.array(energies)
            k = max(1, int(np.ceil(cvar_alpha * len(energies))))
            cvar = float(np.mean(np.sort(energies)[:k]))
            history.append(cvar)
            return cvar

        # Optimiser
        opt = qml.SPSAOptimizer(maxiter=max_iter)
        params = init_params

        for i in range(max_iter):
            params, cost = opt.step_and_cost(cost_fn, params)
            if i % max(1, max_iter // 10) == 0:
                print(f"  Iter {i:4d}/{max_iter}: CVaR={cost:+8.3f}")

        elapsed = time.time() - start_time

        # Final sampling
        opt_gammas = params[:p_layers]
        opt_betas = params[p_layers:]

        @qml.qnode(self.dev)
        def final_circuit():
            self.qaoa_circuit(opt_gammas, opt_betas)
            return qml.sample()

        final_samples = final_circuit()
        if final_samples.ndim == 1:
            if self.n_qubits == 1:
                final_samples = final_samples.reshape(-1, 1)
            else:
                final_samples = final_samples.reshape(1, -1)

        best_e = float("inf")
        best_coords = []
        valid_count = 0

        for sample in final_samples:
            coords = bitstring_to_coords(
                sample, self.protein.n, self.bits_per_link
            )
            e = self.protein.evaluate_energy(
                coords, collision_penalty=self.collision_penalty
            )
            if CubicLattice.is_self_avoiding(coords):
                valid_count += 1
            if e < best_e:
                best_e = e
                best_coords = [c.copy() for c in coords]

        valid_rate = valid_count / max(len(final_samples), 1)

        # Ground state for approximation ratio
        gs_energy = self.qubo.ground_state_energy()

        result = {
            "best_energy": float(best_e),
            "best_coords": best_coords,
            "best_fold": [c.tolist() for c in best_coords],
            "valid_rate": valid_rate,
            "energies": history,
            "optimal_gammas": list(opt_gammas),
            "optimal_betas": list(opt_betas),
            "time_seconds": elapsed,
            "ground_state_energy": gs_energy,
            "spectral_gap": self.qubo.spectral_gap(),
        }

        # Approximation ratio (meaningful only for valid folds)
        if gs_energy < 0:
            result["approximation_ratio"] = float(best_e) / gs_energy
        else:
            result["approximation_ratio"] = 1.0

        print(f"  ✓ Best energy: {best_e:.3f} (exact: {gs_energy:.3f})")
        print(f"  ✓ Approx ratio: {result['approximation_ratio']:.3f}")
        print(f"  ✓ Valid rate: {valid_rate*100:.1f}%")
        print(f"  ✓ Time: {elapsed:.2f}s")

        return result
