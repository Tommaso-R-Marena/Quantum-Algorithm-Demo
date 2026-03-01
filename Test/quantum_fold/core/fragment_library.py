"""
fragment_library.py
Fragment library generation for Quantum Fragment Assembly.

A protein is decomposed into overlapping fragments of k residues.
Each fragment has a discrete set of conformations drawn from
Ramachandran-binned dihedral angles. The assembly problem —
choosing one conformation per fragment to minimise total energy —
is the QUBO that the quantum optimiser solves.

Key concepts:
  - A "fragment" is a contiguous stretch of k residues.
  - Fragments overlap by s residues (default s = k//2).
  - Each fragment conformation is defined by (phi, psi) angles
    for each residue in the fragment.
  - The number of conformations per fragment = n_bins^k (exponential).
    For tractability, we prune to the M best conformations per fragment.

References:
  [1] Simons et al., J. Mol. Biol. 268, 209 (1997) — fragment assembly (Rosetta)
  [2] Robert et al., npj Quantum Inf. 7, 38 (2021)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from .backbone import (
    build_ca_trace,
    ramachandran_bins,
    bin_to_angles,
    kabsch_rmsd,
)
from .force_field import (
    contact_energy,
    clash_energy,
    ramachandran_score,
    CoarseGrainedForceField,
)
from .residue import predict_secondary_structure, RAMA_REGIONS


@dataclass
class FragmentConformation:
    """A single conformation of a fragment."""
    phi: np.ndarray          # dihedral angles
    psi: np.ndarray
    ca_coords: np.ndarray    # Cα coordinates (local frame)
    internal_energy: float   # Ramachandran + local clash score
    bin_indices: List[int]   # Ramachandran bin index per residue


@dataclass
class Fragment:
    """A fragment of the protein with multiple conformations."""
    start_idx: int           # residue index in full sequence
    end_idx: int             # exclusive
    sequence: str            # subsequence
    conformations: List[FragmentConformation] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx

    @property
    def n_conformations(self) -> int:
        return len(self.conformations)


class FragmentLibrary:
    """
    Generates and manages fragment conformations for quantum assembly.

    Parameters
    ----------
    sequence : str
        Full protein sequence.
    fragment_size : int
        Number of residues per fragment (default 5).
    overlap : int
        Overlap between adjacent fragments (default 2).
    n_rama_bins : int
        Number of Ramachandran bins per dimension (default 4).
        Total bins per residue = n_rama_bins^2 (e.g., 16 for n=4).
    max_conformations : int
        Maximum conformations per fragment after pruning (default 16).
    """

    def __init__(
        self,
        sequence: str,
        fragment_size: int = 5,
        overlap: int = 2,
        n_rama_bins: int = 4,
        max_conformations: int = 16,
    ):
        self.sequence = sequence
        self.n_residues = len(sequence)
        self.fragment_size = fragment_size
        self.overlap = overlap
        self.n_rama_bins = n_rama_bins
        self.max_conformations = max_conformations
        self.fragments: List[Fragment] = []
        self.ss_prediction = predict_secondary_structure(sequence)

    def generate(self) -> List[Fragment]:
        """
        Generate the fragment library.

        1. Divide the sequence into overlapping fragments.
        2. For each fragment, generate conformations from Ramachandran bins.
        3. Score each conformation.
        4. Prune to keep only the best M conformations.

        Returns
        -------
        fragments : list of Fragment
        """
        step = self.fragment_size - self.overlap
        if step < 1:
            step = 1

        # Generate fragment positions
        positions = []
        i = 0
        while i + self.fragment_size <= self.n_residues:
            positions.append((i, i + self.fragment_size))
            i += step
        # Handle last fragment
        if positions and positions[-1][1] < self.n_residues:
            last_start = max(0, self.n_residues - self.fragment_size)
            positions.append((last_start, self.n_residues))

        if not positions:
            positions = [(0, self.n_residues)]

        # Generate conformations for each fragment
        self.fragments = []
        total_bins = self.n_rama_bins ** 2  # bins per residue

        for start, end in positions:
            frag_seq = self.sequence[start:end]
            frag_ss = self.ss_prediction[start:end]
            frag = Fragment(start_idx=start, end_idx=end, sequence=frag_seq)

            # Generate SS-guided conformations
            conformations = self._generate_conformations(frag_seq, frag_ss)

            # Score and prune
            scored = []
            for conf in conformations:
                scored.append(conf)

            # Sort by internal energy and keep top M
            scored.sort(key=lambda c: c.internal_energy)
            frag.conformations = scored[:self.max_conformations]

            self.fragments.append(frag)

        return self.fragments

    def _generate_conformations(
        self,
        frag_seq: str,
        frag_ss: str,
    ) -> List[FragmentConformation]:
        """
        Generate conformations for a single fragment.

        Strategy: Use secondary-structure-guided sampling instead of
        exhaustive enumeration (which is intractable for k > 3 with
        many bins).
        """
        k = len(frag_seq)
        conformations = []

        # 1. SS-guided canonical conformations
        ss_angles = {
            "H": (-1.05, -0.79),    # alpha helix: phi=-60, psi=-45
            "E": (-2.25, 2.35),       # beta strand: phi=-129, psi=135
            "C": (-1.22, 2.53),       # coil (PPII-like)
        }

        # Generate the base SS conformation
        base_phi = np.array([ss_angles.get(frag_ss[i], (-1.22, 2.53))[0] for i in range(k)])
        base_psi = np.array([ss_angles.get(frag_ss[i], (-1.22, 2.53))[1] for i in range(k)])

        base_ca = build_ca_trace(base_phi, base_psi)
        base_energy = ramachandran_score(base_phi, base_psi, frag_seq)
        conformations.append(FragmentConformation(
            phi=base_phi, psi=base_psi, ca_coords=base_ca,
            internal_energy=base_energy, bin_indices=[0] * k,
        ))

        # 2. Random perturbations from the base SS conformation
        rng = np.random.default_rng(hash(frag_seq) % (2**31))
        n_random = min(self.max_conformations * 3, 200)

        for _ in range(n_random):
            # Perturb each angle by a small amount
            phi_pert = base_phi + rng.normal(0, 0.3, k)
            psi_pert = base_psi + rng.normal(0, 0.3, k)

            # Clip to [-pi, pi]
            phi_pert = np.clip(phi_pert, -np.pi, np.pi)
            psi_pert = np.clip(psi_pert, -np.pi, np.pi)

            ca = build_ca_trace(phi_pert, psi_pert)
            e = ramachandran_score(phi_pert, psi_pert, frag_seq)
            e += clash_energy(ca) * 5.0  # penalise clashes

            conformations.append(FragmentConformation(
                phi=phi_pert, psi=psi_pert, ca_coords=ca,
                internal_energy=e, bin_indices=[-1] * k,
            ))

        # 3. Canonical Ramachandran bin conformations
        bins = ramachandran_bins(self.n_rama_bins)
        n_bins_total = len(bins)

        # For each residue, try major Ramachandran regions
        key_bins = [0, n_bins_total // 4, n_bins_total // 2, 3 * n_bins_total // 4]
        for bi in key_bins:
            phi_bin = np.array([bins[bi % n_bins_total][0]] * k)
            psi_bin = np.array([bins[bi % n_bins_total][1]] * k)

            ca = build_ca_trace(phi_bin, psi_bin)
            e = ramachandran_score(phi_bin, psi_bin, frag_seq)
            e += clash_energy(ca) * 5.0

            conformations.append(FragmentConformation(
                phi=phi_bin, psi=psi_bin, ca_coords=ca,
                internal_energy=e, bin_indices=[bi] * k,
            ))

        return conformations

    def n_qubits_needed(self) -> int:
        """
        Total qubits needed to encode the fragment assembly problem.
        Each fragment with M conformations needs ceil(log2(M)) qubits.
        """
        total = 0
        for frag in self.fragments:
            m = max(frag.n_conformations, 1)
            total += int(np.ceil(np.log2(max(m, 2))))
        return total

    def build_assembly_energy_matrix(
        self,
        force_field: Optional[CoarseGrainedForceField] = None,
    ) -> Dict:
        """
        Build the energy matrices for the QUBO formulation.
        Optimized to avoid redundant force field evaluations.
        """
        if force_field is None:
            force_field = CoarseGrainedForceField()

        n_frags = len(self.fragments)

        # 1. Internal energies (Ramachandran + local clashes)
        internal = []
        for frag in self.fragments:
            energies = np.array([c.internal_energy for c in frag.conformations])
            internal.append(energies)

        # 2. Pre-calculate non-bonded internal energies for each conformation
        # (dfire, lj, elec, solv) to avoid redundant calculation in pairwise loop.
        internal_nb = []
        for frag in self.fragments:
            nb_energies = []
            for conf in frag.conformations:
                terms = force_field.score_decomposed(conf.ca_coords, frag.sequence)
                e_nb = terms["dfire"] + terms["lj"] + terms["elec"] + terms["solv"]
                nb_energies.append(e_nb)
            internal_nb.append(np.array(nb_energies))

        # 3. Pairwise interaction energies between adjacent fragments
        pairwise = {}
        for fi in range(n_frags - 1):
            fj = fi + 1
            frag_i = self.fragments[fi]
            frag_j = self.fragments[fj]

            m_i = frag_i.n_conformations
            m_j = frag_j.n_conformations

            E_pair = np.zeros((m_i, m_j), dtype=np.float64)

            # Pre-calculate shared info for the fragment pair
            overlap_start = frag_j.start_idx
            overlap_end = frag_i.end_idx
            n_overlap = max(0, overlap_end - overlap_start)

            # Subsequences for combined trace
            if n_overlap > 0:
                seq_i_only = frag_i.sequence[:-n_overlap]
                seq_overlap = frag_i.sequence[-n_overlap:]
                seq_j_only = frag_j.sequence[n_overlap:]
                combined_seq = seq_i_only + seq_overlap + seq_j_only
            else:
                combined_seq = frag_i.sequence + frag_j.sequence

            for ci in range(m_i):
                conf_i = frag_i.conformations[ci]
                e_nb_i = internal_nb[fi][ci]

                for cj in range(m_j):
                    conf_j = frag_j.conformations[cj]
                    e_nb_j = internal_nb[fj][cj]

                    # Evaluate overlap consistency + interaction energy
                    e = self._pairwise_energy_opt(
                        conf_i, e_nb_i,
                        conf_j, e_nb_j,
                        combined_seq, n_overlap, force_field
                    )
                    E_pair[ci, cj] = e

            pairwise[(fi, fj)] = E_pair

        return {
            "internal": internal,
            "pairwise": pairwise,
            "n_fragments": n_frags,
            "conformations_per_fragment": [f.n_conformations for f in self.fragments],
        }

    def _pairwise_energy_opt(
        self,
        conf_i: FragmentConformation, e_nb_i: float,
        conf_j: FragmentConformation, e_nb_j: float,
        combined_seq: str, n_overlap: int,
        force_field: CoarseGrainedForceField,
    ) -> float:
        """Optimized pairwise energy calculation."""
        energy = 0.0

        # 1. Overlap consistency
        if n_overlap > 0:
            ca_i_overlap = conf_i.ca_coords[-n_overlap:]
            ca_j_overlap = conf_j.ca_coords[:n_overlap]
            rmsd, _ = kabsch_rmsd(ca_i_overlap, ca_j_overlap)
            energy += rmsd ** 2 * 10.0  # quadratic penalty

            # Build combined Cα trace (optimized vectorization)
            ca_i_only = conf_i.ca_coords[:-n_overlap]
            ca_overlap = (ca_i_overlap + ca_j_overlap) * 0.5
            ca_j_only = conf_j.ca_coords[n_overlap:]
            combined_ca = np.vstack([ca_i_only, ca_overlap, ca_j_only])
        else:
            combined_ca = np.vstack([conf_i.ca_coords, conf_j.ca_coords])

        # 2. Inter-fragment energy (E_combined - E_i - E_j)
        ff_terms = force_field.score_decomposed(combined_ca, combined_seq)
        e_ij = ff_terms["dfire"] + ff_terms["lj"] + ff_terms["elec"] + ff_terms["solv"]

        inter_energy = e_ij - e_nb_i - e_nb_j
        energy += inter_energy

        return energy

    def assemble(
        self,
        choices: List[int],
    ) -> np.ndarray:
        """
        Assemble a full Cα trace from fragment conformation choices.

        Parameters
        ----------
        choices : list of int
            For each fragment, the index of the chosen conformation.

        Returns
        -------
        ca_coords : np.ndarray, shape (n_residues, 3)
        """
        if len(choices) != len(self.fragments):
            raise ValueError(
                f"Expected {len(self.fragments)} choices, got {len(choices)}"
            )

        # Weighted average in overlap regions
        coords = np.zeros((self.n_residues, 3), dtype=np.float64)
        weights = np.zeros(self.n_residues, dtype=np.float64)

        for fi, (frag, ci) in enumerate(zip(self.fragments, choices)):
            ci = min(ci, frag.n_conformations - 1)
            conf = frag.conformations[ci]
            ca = conf.ca_coords

            for k in range(frag.length):
                res_idx = frag.start_idx + k
                if res_idx < self.n_residues and k < len(ca):
                    coords[res_idx] += ca[k]
                    weights[res_idx] += 1.0

        # Average overlapping positions
        mask = weights > 0
        coords[mask] /= weights[mask, np.newaxis]

        return coords

    def summary(self) -> str:
        """Print a summary of the fragment library."""
        lines = [
            f"Fragment Library for {self.sequence} (N={self.n_residues})",
            f"  Fragment size: {self.fragment_size}, overlap: {self.overlap}",
            f"  Number of fragments: {len(self.fragments)}",
            f"  Qubits needed: {self.n_qubits_needed()}",
            f"  SS prediction: {self.ss_prediction}",
        ]
        for i, frag in enumerate(self.fragments):
            lines.append(
                f"  Fragment {i}: [{frag.start_idx}:{frag.end_idx}] "
                f"{frag.sequence} ({frag.n_conformations} conformations)"
            )
        return "\n".join(lines)
