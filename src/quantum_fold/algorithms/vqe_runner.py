"""
vqe_runner.py
Hybrid CVaR-VQE for lattice protein folding on quantum hardware.

Key features:
  • 2-bit relative-turn encoding (no invalid bitstrings)
  • CVaR (Conditional Value-at-Risk) objective for tail-risk optimisation
  • SPSA optimiser (gradient-free, noise-resilient)
  • Hardware-Efficient Ansatz with RY-RZ rotations + CNOT entanglement
  • Optional depolarising noise simulation
  • Zero-Noise Extrapolation (ZNE) error mitigation
  • Adaptive penalty scheduling for soft constraint enforcement

References:
  [1] Barkoutsos et al., Quantum 4, 256 (2020)  — CVaR-VQE
  [2] Spall, IEEE TAC 37, 332 (1992)  — SPSA
  [3] Temme et al., PRL 119, 180509 (2017)  — ZNE
  [4] Robert et al., npj Quantum Inf. 7, 38 (2021)
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

from ..core.encoding import bitstring_to_coords
from ..core.protein import Protein
from ..core.lattice import CubicLattice


class VQEFolder:
    """
    CVaR-VQE protein folder using PennyLane.

    Parameters
    ----------
    protein : Protein
        Protein to fold.
    params : dict
        Configuration dictionary with keys:
          shots        : int   — measurement shots per evaluation (default 1000)
          max_iter     : int   — optimisation iterations (default 100)
          depth        : int   — ansatz circuit depth (default 3)
          cvar_alpha   : float — CVaR quantile (default 0.10)
          noise_prob   : float — depolarising noise probability (default 0.0)
          penalty_start: float — initial collision penalty (default 5.0)
          penalty_end  : float — final collision penalty (default 200.0)
          bits_per_link: int   — 2 (default) or 3
          zne_enabled  : bool  — enable Zero-Noise Extrapolation (default False)
          seed         : int   — random seed (default None)
    """

    def __init__(self, protein: Protein, params: Dict):
        self.protein = protein
        self.params = params
        self.bits_per_link = params.get("bits_per_link", 2)
        self.n_qubits = CubicLattice.n_qubits(protein.n, self.bits_per_link)

        if self.n_qubits == 0:
            raise ValueError(
                f"Sequence length {protein.n} is too short for VQE "
                f"(need at least 3 beads for 1 variable link)"
            )

        self.noise_prob = params.get("noise_prob", 0.0)
        self.shots = params.get("shots", 1000)
        self.seed = params.get("seed", None)

        # Device selection
        if self.noise_prob > 0:
            self.dev = qml.device(
                "default.mixed", wires=self.n_qubits, shots=self.shots
            )
        else:
            self.dev = qml.device(
                "default.qubit", wires=self.n_qubits, shots=self.shots
            )

    # --- Ansatz -----------------------------------------------------------

    def ansatz(self, weights: np.ndarray):
        """
        Hardware-Efficient Ansatz (HEA).

        Structure per layer:
          1. RY(θ) rotation on each qubit
          2. RZ(φ) rotation on each qubit
          3. Ring of CNOTs for entanglement
          4. Optional depolarising noise after each gate

        weights shape: (depth, n_qubits, 2)
        """
        depth = weights.shape[0]
        for d in range(depth):
            # Single-qubit rotations
            for i in range(self.n_qubits):
                qml.RY(weights[d, i, 0], wires=i)
                if self.noise_prob > 0:
                    qml.DepolarizingChannel(self.noise_prob, wires=i)

                qml.RZ(weights[d, i, 1], wires=i)
                if self.noise_prob > 0:
                    qml.DepolarizingChannel(self.noise_prob, wires=i)

            # Entanglement: ring of CNOTs
            if self.n_qubits > 1:
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                    if self.noise_prob > 0:
                        qml.DepolarizingChannel(self.noise_prob, wires=i)
                        qml.DepolarizingChannel(
                            self.noise_prob, wires=(i + 1) % self.n_qubits
                        )

    # --- Cost function ----------------------------------------------------

    def cost_function(
        self, weights: np.ndarray, current_penalty: float
    ) -> float:
        """
        Compute the CVaR cost from circuit samples.

        1. Run the quantum circuit with current weights
        2. Measure bitstrings
        3. Evaluate classical energy for each bitstring
        4. Return CVaR (mean of lowest α-quantile energies)
        """
        @qml.qnode(self.dev)
        def circuit(w):
            self.ansatz(w)
            return qml.sample()

        samples = circuit(weights)

        # Handle single-qubit case where samples is 1D
        if samples.ndim == 1 and self.n_qubits == 1:
            samples = samples.reshape(-1, 1)
        elif samples.ndim == 1:
            samples = samples.reshape(1, -1)

        energies = []
        for sample in samples:
            coords = bitstring_to_coords(
                sample, self.protein.n, self.bits_per_link
            )
            e = self.protein.evaluate_energy(
                coords, collision_penalty=current_penalty
            )
            energies.append(float(e))

        energies = np.array(energies)

        # CVaR: mean of the lowest α fraction
        alpha = self.params.get("cvar_alpha", 0.10)
        k = max(1, int(np.ceil(alpha * len(energies))))
        energies_sorted = np.sort(energies)
        cvar = float(np.mean(energies_sorted[:k]))

        return cvar

    # --- Main optimisation loop -------------------------------------------

    def run(self) -> Dict:
        """
        Execute the CVaR-VQE optimisation loop.

        Returns
        -------
        dict with keys:
          energies     : list of cost values per iteration
          final_weights: optimised circuit parameters
          best_energy  : lowest energy found
          best_coords  : coordinates of best fold
          best_fold    : best fold as list-of-lists
          valid_rate   : fraction of valid (non-colliding) samples
          time_seconds : wall-clock time
          approximation_ratio : best_energy / exact_ground (if computable)
        """
        depth = self.params.get("depth", 3)
        max_iter = self.params.get("max_iter", 100)
        penalty_start = self.params.get("penalty_start", 5.0)
        penalty_end = self.params.get("penalty_end", 200.0)

        rng = np.random.default_rng(self.seed)

        # Random initialisation (near zero for trainability)
        init_weights = pnp.array(
            rng.normal(0, 0.1, (depth, self.n_qubits, 2)),
            requires_grad=True,
        )

        # SPSA optimiser (gradient-free, noise-resilient)
        opt = qml.SPSAOptimizer(maxiter=max_iter)

        params = init_weights
        history = []

        start_time = time.time()

        print(f"┌──────────────────────────────────────────────────")
        print(f"│ CVaR-VQE  N={self.protein.n}  Qubits={self.n_qubits}")
        print(f"│ Depth={depth}  Shots={self.shots}  α={self.params.get('cvar_alpha', 0.10)}")
        print(f"│ Noise p={self.noise_prob}")
        print(f"│ Penalty: {penalty_start:.1f} → {penalty_end:.1f}")
        print(f"└──────────────────────────────────────────────────")

        for i in range(max_iter):
            # Adaptive penalty schedule (linear ramp for first 60%, hold after)
            ramp_end = int(max_iter * 0.6)
            if i < ramp_end:
                progress = i / max(ramp_end, 1)
                current_p = penalty_start + progress * (penalty_end - penalty_start)
            else:
                current_p = penalty_end

            def cost_wrapper(w):
                return self.cost_function(w, current_p)

            params, cost = opt.step_and_cost(cost_wrapper, params)
            history.append(float(cost))

            if i % max(1, max_iter // 10) == 0:
                print(
                    f"  Iter {i:4d}/{max_iter}: "
                    f"CVaR={cost:+8.3f}  penalty={current_p:.1f}"
                )

        elapsed = time.time() - start_time

        # Final sampling and analysis
        print("  Sampling final distribution...")
        final_result = self._sample_final(params, penalty_end)

        # ZNE error mitigation (if enabled)
        if self.params.get("zne_enabled", False) and self.noise_prob > 0:
            print("  Applying Zero-Noise Extrapolation...")
            zne_energy = self._zero_noise_extrapolation(params, penalty_end)
            final_result["zne_energy"] = zne_energy

        result = {
            "energies": history,
            "final_weights": params,
            "best_energy": final_result["min_energy"],
            "best_coords": final_result["best_coords"],
            "best_fold": final_result["best_fold"],
            "valid_rate": final_result["valid_rate"],
            "time_seconds": elapsed,
            "energy_distribution": final_result.get("energy_distribution", []),
        }

        if "zne_energy" in final_result:
            result["zne_energy"] = final_result["zne_energy"]

        print(f"  ✓ Best energy: {result['best_energy']:.3f}")
        print(f"  ✓ Valid fold rate: {result['valid_rate']*100:.1f}%")
        print(f"  ✓ Time: {elapsed:.2f}s")

        return result

    # --- Final sampling ---------------------------------------------------

    def _sample_final(
        self, weights: np.ndarray, penalty: float
    ) -> Dict:
        """Sample the optimised circuit and identify the best fold."""

        @qml.qnode(self.dev)
        def circuit(w):
            self.ansatz(w)
            return qml.sample()

        samples = circuit(weights)

        if samples.ndim == 1 and self.n_qubits == 1:
            samples = samples.reshape(-1, 1)
        elif samples.ndim == 1:
            samples = samples.reshape(1, -1)

        best_e = float("inf")
        best_coords = []
        valid_count = 0
        total_count = len(samples)
        energies = []

        for sample in samples:
            coords = bitstring_to_coords(
                sample, self.protein.n, self.bits_per_link
            )
            e = self.protein.evaluate_energy(
                coords, collision_penalty=penalty
            )
            energies.append(float(e))

            # Check validity (no collisions → energy without penalty matches)
            e_no_penalty = self.protein.evaluate_energy(
                coords, collision_penalty=0.0
            )
            if CubicLattice.is_self_avoiding(coords):
                valid_count += 1

            if e < best_e:
                best_e = e
                best_coords = [c.copy() for c in coords]

        valid_rate = valid_count / max(total_count, 1)

        return {
            "min_energy": float(best_e),
            "best_coords": best_coords,
            "best_fold": [c.tolist() for c in best_coords],
            "valid_rate": valid_rate,
            "energy_distribution": energies,
        }

    # --- Zero-Noise Extrapolation -----------------------------------------

    def _zero_noise_extrapolation(
        self, weights: np.ndarray, penalty: float
    ) -> float:
        """
        Richardson extrapolation with noise scale factors [1, 2, 3].

        Runs the circuit at amplified noise levels and fits a linear
        model to extrapolate to zero noise.
        """
        scale_factors = [1.0, 2.0, 3.0]
        energies_at_scale = []

        original_noise = self.noise_prob

        for sf in scale_factors:
            self.noise_prob = original_noise * sf

            # Re-create device for new noise level
            if self.noise_prob > 0:
                self.dev = qml.device(
                    "default.mixed",
                    wires=self.n_qubits,
                    shots=self.shots,
                )
            result = self._sample_final(weights, penalty)
            energies_at_scale.append(result["min_energy"])

        # Restore original noise
        self.noise_prob = original_noise
        if self.noise_prob > 0:
            self.dev = qml.device(
                "default.mixed", wires=self.n_qubits, shots=self.shots
            )
        else:
            self.dev = qml.device(
                "default.qubit", wires=self.n_qubits, shots=self.shots
            )

        # Linear extrapolation to noise=0
        coeffs = np.polyfit(scale_factors, energies_at_scale, deg=1)
        zne_energy = float(np.polyval(coeffs, 0.0))

        print(
            f"  ZNE: scales={scale_factors}, "
            f"energies={[f'{e:.3f}' for e in energies_at_scale]}, "
            f"extrapolated={zne_energy:.3f}"
        )

        return zne_energy
