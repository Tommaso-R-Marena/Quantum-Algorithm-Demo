"""
fragment_qopt.py
Quantum Fragment Assembly Optimiser.

THIS IS THE PRIMARY NOVEL CONTRIBUTION: formulates the protein fragment
assembly problem as a QUBO and solves it with VQE/QAOA.

Problem:
  Given K fragments, each with M_k conformations, choose one conformation
  per fragment to minimise the total energy:
    E_total = sum_k E_internal(k, x_k) + sum_{k,l} E_pairwise(k,x_k, l,x_l)
  where x_k in {0, 1, ..., M_k-1}.

QUBO encoding:
  Each fragment k uses ceil(log2(M_k)) binary qubits.
  The conformation index is encoded as an unsigned integer in binary.
  The total number of qubits = sum_k ceil(log2(M_k)).

References:
  [1] Perdomo-Ortiz et al., Sci. Rep. 2, 571 (2012) — quantum protein folding
  [2] Robert et al., npj Quantum Inf. 7, 38 (2021) — resource-efficient encoding
  [3] Barkoutsos et al., Quantum 4, 256 (2020) — CVaR-VQE for combinatorial
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import time
from typing import Dict, List, Optional, Tuple

from ..core.fragment_library import FragmentLibrary, Fragment
from ..core.force_field import CoarseGrainedForceField
from ..core.backbone import kabsch_rmsd


class FragmentQUBO:
    """
    Formulates the protein fragment assembly problem as a Quadratic
    Unconstrained Binary Optimization (QUBO) task.

    The cost function is defined as:
        H = Σ_k H_internal(k, x_k) + Σ_{k,l} H_pairwise(k, x_k, l, x_l)
    where x_k is the binary-encoded index of the conformation for fragment k.
    Each conformation index x_k is mapped to a set of qubits using a
    resource-efficient integer encoding.

    Parameters
    ----------
    library : FragmentLibrary
        Fragment library containing discrete conformations.
    force_field : CoarseGrainedForceField, optional
        Force field used to evaluate interaction energies.
    """

    def __init__(
        self,
        library: FragmentLibrary,
        force_field: Optional[CoarseGrainedForceField] = None,
    ):
        self.library = library
        self.ff = force_field or CoarseGrainedForceField()

        # Build energy matrices
        self.energy_data = library.build_assembly_energy_matrix(self.ff)

        # Qubit allocation
        self.n_fragments = self.energy_data["n_fragments"]
        self.confs_per_frag = self.energy_data["conformations_per_fragment"]
        self.qubits_per_frag = [
            int(np.ceil(np.log2(max(m, 2)))) for m in self.confs_per_frag
        ]
        self.total_qubits = sum(self.qubits_per_frag)

        # Qubit offset for each fragment
        self.qubit_offsets = []
        offset = 0
        for q in self.qubits_per_frag:
            self.qubit_offsets.append(offset)
            offset += q

    def decode_bitstring(self, bitstring: np.ndarray) -> List[int]:
        """
        Decode a bitstring into fragment conformation choices.

        Returns
        -------
        choices : list of int
            Conformation index for each fragment.
        """
        choices = []
        for k in range(self.n_fragments):
            start = self.qubit_offsets[k]
            n_bits = self.qubits_per_frag[k]
            bits = bitstring[start:start + n_bits]

            # Binary to integer
            idx = 0
            for b in bits:
                idx = 2 * idx + int(b)

            # Clamp to valid range
            idx = min(idx, self.confs_per_frag[k] - 1)
            choices.append(idx)

        return choices

    def evaluate(self, bitstring: np.ndarray) -> float:
        """
        Evaluate the total energy for a given bitstring encoding.
        """
        choices = self.decode_bitstring(bitstring)
        return self.evaluate_choices(choices)

    def evaluate_choices(self, choices: List[int]) -> float:
        """Evaluate the total energy for given fragment choices."""
        E = 0.0

        # Internal energies
        for k, ci in enumerate(choices):
            ci_clamped = min(ci, len(self.energy_data["internal"][k]) - 1)
            E += self.energy_data["internal"][k][ci_clamped]

        # Pairwise energies
        for (ki, kj), E_pair in self.energy_data["pairwise"].items():
            ci = min(choices[ki], E_pair.shape[0] - 1)
            cj = min(choices[kj], E_pair.shape[1] - 1)
            E += E_pair[ci, cj]

        return E

    def cost_vector(self) -> np.ndarray:
        """
        Compute the full cost vector over all 2^n_qubits bitstrings.
        For small systems only (n_qubits <= 20).
        """
        n = self.total_qubits
        if n > 20:
            raise ValueError(f"Cost vector too large: 2^{n} entries")

        costs = np.zeros(2 ** n, dtype=np.float64)
        for i in range(2 ** n):
            bits = np.array(
                [int(b) for b in format(i, f"0{n}b")], dtype=np.int64
            )
            costs[i] = self.evaluate(bits)
        return costs

    def ground_state(self) -> Tuple[np.ndarray, float, List[int]]:
        """
        Find the ground state by exhaustive enumeration.
        Returns (bitstring, energy, choices).
        """
        n = self.total_qubits
        if n > 20:
            raise ValueError(f"Exhaustive search too large: 2^{n}")

        best_e = float("inf")
        best_bits = None
        best_choices = None

        for i in range(2 ** n):
            bits = np.array(
                [int(b) for b in format(i, f"0{n}b")], dtype=np.int64
            )
            e = self.evaluate(bits)
            if e < best_e:
                best_e = e
                best_bits = bits.copy()
                best_choices = self.decode_bitstring(bits)

        return best_bits, best_e, best_choices


class QuantumFragmentAssembler:
    """
    Solve the fragment assembly QUBO using VQE or QAOA.

    Parameters
    ----------
    qubo : FragmentQUBO
    method : str
        "vqe" or "qaoa"
    params : dict
        shots, max_iter, depth, cvar_alpha, seed
    """

    def __init__(
        self,
        qubo: FragmentQUBO,
        method: str = "vqe",
        params: Optional[Dict] = None,
    ):
        self.qubo = qubo
        self.method = method
        self.params = params or {}
        self.n_qubits = qubo.total_qubits

        shots = self.params.get("shots", 500)
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)

    def run(self) -> Dict:
        """
        Run the quantum optimisation.

        Returns
        -------
        dict with:
          best_energy, best_choices, best_bitstring,
          assembled_coords, energies (history), time_seconds,
          ground_state_energy, approximation_ratio
        """
        max_iter = self.params.get("max_iter", 80)
        depth = self.params.get("depth", 3)
        cvar_alpha = self.params.get("cvar_alpha", 0.15)
        shots = self.params.get("shots", 500)
        seed = self.params.get("seed", 42)

        rng = np.random.default_rng(seed)

        print(f"  QFA: {self.n_qubits} qubits, {self.qubo.n_fragments} fragments")
        print(f"  Method: {self.method}, depth={depth}, shots={shots}")

        start_time = time.time()

        if self.method == "vqe":
            result = self._run_vqe(max_iter, depth, cvar_alpha, rng)
        else:
            result = self._run_qaoa(max_iter, depth, cvar_alpha, rng)

        elapsed = time.time() - start_time
        result["time_seconds"] = elapsed

        # Assemble the best structure
        result["assembled_coords"] = self.qubo.library.assemble(result["best_choices"])

        # Ground state (if tractable)
        if self.n_qubits <= 16:
            _, gs_e, gs_choices = self.qubo.ground_state()
            result["ground_state_energy"] = gs_e
            result["ground_state_choices"] = gs_choices
            if abs(gs_e) > 1e-10:
                result["approximation_ratio"] = result["best_energy"] / gs_e
            else:
                result["approximation_ratio"] = 1.0

        print(f"  Best energy: {result['best_energy']:.3f}")
        print(f"  Time: {elapsed:.2f}s")

        return result

    def _run_vqe(
        self,
        max_iter: int,
        depth: int,
        cvar_alpha: float,
        rng: np.random.Generator,
    ) -> Dict:
        """CVaR-VQE for fragment QUBO."""
        n = self.n_qubits
        n_params = depth * n * 3  # RY-RZ-CNOT per layer

        init_params = pnp.array(
            rng.uniform(-np.pi, np.pi, n_params), requires_grad=True
        )

        history = []

        def ansatz(params):
            idx = 0
            for layer in range(depth):
                for q in range(n):
                    qml.RY(params[idx], wires=q)
                    idx += 1
                for q in range(n):
                    qml.RZ(params[idx], wires=q)
                    idx += 1
                for q in range(n):
                    qml.CNOT(wires=[q, (q + 1) % n])
                for q in range(n):
                    qml.RY(params[idx], wires=q)
                    idx += 1

        @qml.qnode(self.dev)
        def circuit(params):
            ansatz(params)
            return qml.sample()

        def cost_fn(params):
            samples = circuit(params)
            if samples.ndim == 1:
                samples = samples.reshape(-1 if n == 1 else 1, n)

            energies = []
            for sample in samples:
                e = self.qubo.evaluate(sample)
                energies.append(e)

            energies = np.array(energies)
            k = max(1, int(np.ceil(cvar_alpha * len(energies))))
            cvar = float(np.mean(np.sort(energies)[:k]))
            history.append(cvar)
            return cvar

        # Optimise
        opt = qml.SPSAOptimizer(maxiter=max_iter)
        params = init_params

        for i in range(max_iter):
            params, cost = opt.step_and_cost(cost_fn, params)
            if i % max(1, max_iter // 5) == 0:
                print(f"    VQE iter {i:4d}: CVaR={cost:+8.3f}")

        # Final sampling
        @qml.qnode(self.dev)
        def final_circuit(params):
            idx = 0
            for layer in range(depth):
                for q in range(n):
                    qml.RY(params[idx], wires=q)
                    idx += 1
                for q in range(n):
                    qml.RZ(params[idx], wires=q)
                    idx += 1
                for q in range(n):
                    qml.CNOT(wires=[q, (q + 1) % n])
                for q in range(n):
                    qml.RY(params[idx], wires=q)
                    idx += 1
            return qml.sample()

        final_samples = final_circuit(params)
        if final_samples.ndim == 1:
            final_samples = final_samples.reshape(-1 if n == 1 else 1, n)

        best_e = float("inf")
        best_bits = None
        for sample in final_samples:
            e = self.qubo.evaluate(sample)
            if e < best_e:
                best_e = e
                best_bits = np.array(sample, dtype=np.int64)

        best_choices = self.qubo.decode_bitstring(best_bits)

        return {
            "best_energy": best_e,
            "best_choices": best_choices,
            "best_bitstring": list(best_bits),
            "energies": history,
        }

    def _run_qaoa(
        self,
        max_iter: int,
        p_layers: int,
        cvar_alpha: float,
        rng: np.random.Generator,
    ) -> Dict:
        """QAOA for fragment QUBO."""
        n = self.n_qubits
        history = []

        init_params = pnp.array(
            rng.uniform(0, np.pi, 2 * p_layers), requires_grad=True
        )

        @qml.qnode(self.dev)
        def circuit(params):
            gammas = params[:p_layers]
            betas = params[p_layers:]

            # Initial superposition
            for q in range(n):
                qml.Hadamard(wires=q)

            for layer in range(p_layers):
                # Cost layer: ZZ interactions
                for q in range(n):
                    qml.RZ(2 * gammas[layer], wires=q)
                for q in range(n - 1):
                    qml.CNOT(wires=[q, q + 1])
                    qml.RZ(gammas[layer] * 0.5, wires=q + 1)
                    qml.CNOT(wires=[q, q + 1])

                # Mixer layer
                for q in range(n):
                    qml.RX(2 * betas[layer], wires=q)

            return qml.sample()

        def cost_fn(params):
            samples = circuit(params)
            if samples.ndim == 1:
                samples = samples.reshape(-1 if n == 1 else 1, n)

            energies = []
            for sample in samples:
                e = self.qubo.evaluate(sample)
                energies.append(e)

            energies = np.array(energies)
            k = max(1, int(np.ceil(cvar_alpha * len(energies))))
            cvar = float(np.mean(np.sort(energies)[:k]))
            history.append(cvar)
            return cvar

        opt = qml.SPSAOptimizer(maxiter=max_iter)
        params = init_params

        for i in range(max_iter):
            params, cost = opt.step_and_cost(cost_fn, params)
            if i % max(1, max_iter // 5) == 0:
                print(f"    QAOA iter {i:4d}: CVaR={cost:+8.3f}")

        # Final sampling
        final_samples = circuit(params)
        if final_samples.ndim == 1:
            final_samples = final_samples.reshape(-1 if n == 1 else 1, n)

        best_e = float("inf")
        best_bits = None
        for sample in final_samples:
            e = self.qubo.evaluate(sample)
            if e < best_e:
                best_e = e
                best_bits = np.array(sample, dtype=np.int64)

        best_choices = self.qubo.decode_bitstring(best_bits)

        return {
            "best_energy": best_e,
            "best_choices": best_choices,
            "best_bitstring": list(best_bits),
            "energies": history,
        }


class ClassicalFragmentAssembler:
    """
    Classical baselines for fragment assembly:
      - Exhaustive search (small systems)
      - Greedy
      - Simulated annealing
    """

    def __init__(self, qubo: FragmentQUBO, seed: int = 42):
        self.qubo = qubo
        self.rng = np.random.default_rng(seed)

    def solve_exhaustive(self) -> Tuple[float, List[int]]:
        """Exhaustive search (only for small systems)."""
        _, e, choices = self.qubo.ground_state()
        return e, choices

    def solve_greedy(self) -> Tuple[float, List[int]]:
        """Greedy: pick the best conformation for each fragment sequentially."""
        choices = []
        for k in range(self.qubo.n_fragments):
            best_ci = 0
            best_e = float("inf")
            for ci in range(self.qubo.confs_per_frag[k]):
                # Compute energy with this choice
                trial_choices = choices + [ci]
                # Pad with zeros for remaining
                trial_choices += [0] * (self.qubo.n_fragments - len(trial_choices))
                e = self.qubo.evaluate_choices(trial_choices)
                if e < best_e:
                    best_e = e
                    best_ci = ci
            choices.append(best_ci)

        return self.qubo.evaluate_choices(choices), choices

    def solve_sa(
        self,
        t_start: float = 10.0,
        t_end: float = 0.01,
        n_steps: int = 5000,
    ) -> Tuple[float, List[int]]:
        """Simulated Annealing over fragment choices."""
        n_frags = self.qubo.n_fragments

        # Random initial
        current = [self.rng.integers(0, max(1, m)) for m in self.qubo.confs_per_frag]
        current_e = self.qubo.evaluate_choices(current)
        best = list(current)
        best_e = current_e

        for step in range(n_steps):
            t = t_start * (t_end / t_start) ** (step / max(n_steps - 1, 1))

            # Propose: change one fragment's conformation
            k = self.rng.integers(0, n_frags)
            old_ci = current[k]
            new_ci = self.rng.integers(0, max(1, self.qubo.confs_per_frag[k]))

            current[k] = new_ci
            new_e = self.qubo.evaluate_choices(current)

            delta = new_e - current_e
            if delta < 0 or self.rng.random() < np.exp(-delta / max(t, 1e-10)):
                current_e = new_e
                if current_e < best_e:
                    best_e = current_e
                    best = list(current)
            else:
                current[k] = old_ci

        return best_e, best
