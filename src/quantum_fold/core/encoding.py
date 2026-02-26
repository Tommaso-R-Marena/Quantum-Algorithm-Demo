"""
encoding.py
Converts between quantum bitstrings, turn sequences, and 3D lattice coordinates.

This module implements the bridge between the quantum circuit measurement outcomes
(bitstrings of 0/1) and physical protein conformations on the 3D cubic lattice.

The encoding uses 2 bits per variable link (relative turns), giving 2·(N−2) qubits
total. Bead 0 is fixed at origin, bead 1 at (1,0,0), and the coordinate frame
propagates through relative turns.

References:
  [1] Robert et al., npj Quantum Inf. 7, 38 (2021)
  [2] Perdomo-Ortiz et al., Sci. Rep. 2, 571 (2012)
"""

from __future__ import annotations

import numpy as np
from typing import List, Sequence, Tuple

from .lattice import CubicLattice, CoordinateFrame


def bitstring_to_coords(
    bitstring: Sequence[int],
    n_beads: int,
    bits_per_link: int = 2,
) -> List[np.ndarray]:
    """
    Convert a measured bitstring to 3D lattice coordinates.

    Parameters
    ----------
    bitstring : sequence of int (0 or 1)
        Raw measurement outcome. Length = bits_per_link × (n_beads − 2).
    n_beads : int
        Total number of beads in the sequence.
    bits_per_link : int
        2 for 4-direction mode, 3 for 5-direction mode.

    Returns
    -------
    coords : list of np.ndarray
        N integer coordinate vectors. Each is an independent copy (no aliasing).
    """
    n_links = n_beads - 2
    expected_len = bits_per_link * n_links
    bs = list(bitstring)

    # Handle length mismatch gracefully (pad or truncate)
    if len(bs) < expected_len:
        bs.extend([0] * (expected_len - len(bs)))
    elif len(bs) > expected_len:
        bs = bs[:expected_len]

    turn_codes = CubicLattice.bitstring_to_turn_codes(bs, bits_per_link)

    n_directions = 4 if bits_per_link == 2 else 5
    coords = CubicLattice.turn_sequence_to_coords(turn_codes, n_directions)

    # Ensure independent copies (no aliasing)
    return [c.copy() for c in coords]


def coords_to_bitstring(
    coords: List[np.ndarray],
    bits_per_link: int = 2,
) -> List[int]:
    """
    Inverse mapping: convert coordinates to a bitstring.

    This reverse-engineers the turn codes from the coordinate sequence,
    then encodes them as bits. Useful for verifying round-trip consistency.

    Parameters
    ----------
    coords : list of np.ndarray
        Lattice coordinates (N beads).
    bits_per_link : int
        2 or 3.

    Returns
    -------
    bitstring : list of int (0 or 1)
    """
    if len(coords) < 3:
        return []

    # Reconstruct directions from coordinate differences
    directions = []
    for i in range(1, len(coords)):
        d = coords[i] - coords[i - 1]
        directions.append(d)

    # Reconstruct turn codes by tracking the coordinate frame
    frame = CoordinateFrame()
    turn_codes = []

    for k in range(1, len(directions)):
        new_dir = directions[k]
        found = False
        n_dirs = 4 if bits_per_link == 2 else 5

        for tc in range(n_dirs):
            test_dir, test_frame = frame.apply_turn(tc, n_dirs)
            if np.array_equal(test_dir, new_dir):
                turn_codes.append(tc)
                frame = test_frame
                found = True
                break

        if not found:
            # Direction not reachable with current encoding — use 0
            turn_codes.append(0)
            _, frame = frame.apply_turn(0, n_dirs)

    # Encode turn codes as bits
    bitstring = []
    for tc in turn_codes:
        for b in range(bits_per_link - 1, -1, -1):
            bitstring.append((tc >> b) & 1)

    return bitstring


def enumerate_all_bitstrings(n_qubits: int) -> np.ndarray:
    """
    Generate all 2^n_qubits bitstrings as a (2^n, n) integer array.
    Useful for exact diagonalisation on small systems.
    """
    n = 2 ** n_qubits
    return np.array(
        [[(i >> (n_qubits - 1 - b)) & 1 for b in range(n_qubits)] for i in range(n)],
        dtype=np.int64,
    )


def compute_classical_cost_vector(
    n_beads: int,
    sequence: str,
    collision_penalty: float = 100.0,
    bits_per_link: int = 2,
    energy_model: str = "HP",
) -> np.ndarray:
    """
    Compute the classical cost (energy) for every possible bitstring.

    This is the diagonal of the cost Hamiltonian in the computational basis.
    For small systems (n_qubits ≤ 20), this enables exact analysis of the
    energy landscape and verification of quantum algorithm results.

    Parameters
    ----------
    n_beads : int
        Number of beads.
    sequence : str
        Residue sequence.
    collision_penalty : float
        Penalty for overlapping beads.
    bits_per_link : int
        Bits per variable link (2 or 3).
    energy_model : str
        Energy model name.

    Returns
    -------
    costs : np.ndarray
        Shape (2^n_qubits,). Cost for each bitstring.
    """
    from .protein import Protein

    protein = Protein(sequence, energy_model=energy_model)
    n_qubits = CubicLattice.n_qubits(n_beads, bits_per_link)
    all_bs = enumerate_all_bitstrings(n_qubits)

    costs = np.empty(len(all_bs), dtype=np.float64)
    for idx, bs in enumerate(all_bs):
        coords = bitstring_to_coords(bs, n_beads, bits_per_link)
        costs[idx] = protein.evaluate_energy(coords, collision_penalty=collision_penalty)

    return costs


def find_ground_state_bitstring(
    n_beads: int,
    sequence: str,
    collision_penalty: float = 100.0,
    bits_per_link: int = 2,
    energy_model: str = "HP",
) -> Tuple[List[int], float, List[np.ndarray]]:
    """
    Exhaustively find the ground-state bitstring (lowest energy conformation).

    Returns
    -------
    best_bitstring : list of int
    best_energy : float
    best_coords : list of np.ndarray
    """
    from .protein import Protein

    protein = Protein(sequence, energy_model=energy_model)
    n_qubits = CubicLattice.n_qubits(n_beads, bits_per_link)
    all_bs = enumerate_all_bitstrings(n_qubits)

    best_e = float("inf")
    best_bs = None
    best_coords = None

    for bs in all_bs:
        coords = bitstring_to_coords(bs, n_beads, bits_per_link)
        e = protein.evaluate_energy(coords, collision_penalty=collision_penalty)
        if e < best_e:
            best_e = e
            best_bs = list(bs)
            best_coords = coords

    return best_bs, best_e, best_coords
