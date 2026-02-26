"""
lattice.py
3D Cubic Lattice with relative-turn encoding for quantum protein folding.

Encoding scheme (Perdomo-Ortiz et al., Sci. Rep. 2012; Babbush et al., 2014):
  Each variable link is encoded with 2 qubits representing a *relative turn*
  from the previous bond direction. This eliminates backbone reversals
  (physically forbidden immediate hairpins) and reduces qubit count to
  2·(N − 2) instead of 3·(N − 2).

  Turn codes (relative to a right-handed local frame {forward, left, up}):
    00 → straight   (continue in current heading)
    01 → turn left   (90° about the up-axis)
    10 → turn right  (−90° about the up-axis)
    11 → turn up     (90° about the left-axis)

  Note: "turn down" is NOT included; this is a deliberate asymmetry that
  still covers all self-avoiding walks (the missing direction is reachable
  by composing two turns). This keeps the encoding at exactly 2 bits with
  NO invalid states.

  For sequences requiring the downward direction, we add an optional
  5-direction mode using 3 bits per link (with one invalid state penalised).

References:
  [1] Perdomo-Ortiz et al., Sci. Rep. 2, 571 (2012)
  [2] Babbush et al., New J. Phys. 16, 033040 (2014)
  [3] Robert et al., npj Quantum Inf. 7, 38 (2021)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Sequence

# ---------------------------------------------------------------------------
# Direction vectors on a 3D cubic lattice (integer unit vectors)
# ---------------------------------------------------------------------------
DIRECTION_VECTORS = {
    "+x": np.array([1, 0, 0], dtype=np.int64),
    "-x": np.array([-1, 0, 0], dtype=np.int64),
    "+y": np.array([0, 1, 0], dtype=np.int64),
    "-y": np.array([0, -1, 0], dtype=np.int64),
    "+z": np.array([0, 0, 1], dtype=np.int64),
    "-z": np.array([0, 0, -1], dtype=np.int64),
}

# Canonical ordering for integer indexing (used by exact solver DFS)
INT_TO_VEC = [
    np.array([1, 0, 0], dtype=np.int64),   # 0: +x
    np.array([-1, 0, 0], dtype=np.int64),  # 1: -x
    np.array([0, 1, 0], dtype=np.int64),   # 2: +y
    np.array([0, -1, 0], dtype=np.int64),  # 3: -y
    np.array([0, 0, 1], dtype=np.int64),   # 4: +z
    np.array([0, 0, -1], dtype=np.int64),  # 5: -z
]


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate frame for relative-turn encoding
# ═══════════════════════════════════════════════════════════════════════════

class CoordinateFrame:
    """
    Right-handed orthonormal frame {forward, left, up} on ℤ³.

    The frame tracks the local orientation of the backbone so that
    relative turns can be converted to absolute lattice directions.
    All vectors are integer unit vectors — rotations are exact (90° only).
    """

    __slots__ = ("forward", "left", "up")

    def __init__(
        self,
        forward: np.ndarray | None = None,
        left: np.ndarray | None = None,
        up: np.ndarray | None = None,
    ):
        self.forward = forward if forward is not None else np.array([1, 0, 0], dtype=np.int64)
        self.left = left if left is not None else np.array([0, 1, 0], dtype=np.int64)
        self.up = up if up is not None else np.array([0, 0, 1], dtype=np.int64)

    def copy(self) -> "CoordinateFrame":
        return CoordinateFrame(self.forward.copy(), self.left.copy(), self.up.copy())

    # --- Relative turn operations (return new frame) ----------------------

    def turn_straight(self) -> "CoordinateFrame":
        """Continue in the current heading — frame unchanged."""
        return self.copy()

    def turn_left(self) -> "CoordinateFrame":
        """90° rotation about the up-axis (forward → left)."""
        new_forward = self.left.copy()
        new_left = -self.forward.copy()
        return CoordinateFrame(new_forward, new_left, self.up.copy())

    def turn_right(self) -> "CoordinateFrame":
        """−90° rotation about the up-axis (forward → −left)."""
        new_forward = -self.left.copy()
        new_left = self.forward.copy()
        return CoordinateFrame(new_forward, new_left, self.up.copy())

    def turn_up(self) -> "CoordinateFrame":
        """90° rotation about the left-axis (forward → up)."""
        new_forward = self.up.copy()
        new_up = -self.forward.copy()
        return CoordinateFrame(new_forward, self.left.copy(), new_up)

    def turn_down(self) -> "CoordinateFrame":
        """−90° rotation about the left-axis (forward → −up). Used in 5-dir mode."""
        new_forward = -self.up.copy()
        new_up = self.forward.copy()
        return CoordinateFrame(new_forward, self.left.copy(), new_up)

    # --- Apply a turn code ------------------------------------------------

    TURN_4 = {0: "straight", 1: "left", 2: "right", 3: "up"}
    TURN_5 = {0: "straight", 1: "left", 2: "right", 3: "up", 4: "down"}

    def apply_turn(self, turn_code: int, n_directions: int = 4) -> Tuple[np.ndarray, "CoordinateFrame"]:
        """
        Apply a relative turn and return (absolute_direction_vector, new_frame).

        Parameters
        ----------
        turn_code : int
            0–3 for 4-direction mode, 0–4 for 5-direction mode.
        n_directions : int
            4 (2-bit encoding) or 5 (3-bit encoding, code 5–7 are invalid).

        Returns
        -------
        direction : np.ndarray   – unit lattice vector of the new bond
        new_frame : CoordinateFrame
        """
        if n_directions == 4:
            label = self.TURN_4.get(turn_code, "straight")
        else:
            label = self.TURN_5.get(turn_code, "straight")

        if label == "straight":
            new_frame = self.turn_straight()
        elif label == "left":
            new_frame = self.turn_left()
        elif label == "right":
            new_frame = self.turn_right()
        elif label == "up":
            new_frame = self.turn_up()
        elif label == "down":
            new_frame = self.turn_down()
        else:
            new_frame = self.turn_straight()

        return new_frame.forward.copy(), new_frame


# ═══════════════════════════════════════════════════════════════════════════
# CubicLattice — main lattice class
# ═══════════════════════════════════════════════════════════════════════════

class CubicLattice:
    """
    3D cubic lattice utilities for protein folding.

    Provides two encoding modes:
      • **absolute** (6 directions, 0–5) — used by the exact DFS solver
      • **relative turn** (4 turns, 2 bits) — used by quantum algorithms

    The first bond is always fixed along +x (bead 0 at origin, bead 1 at [1,0,0])
    to remove translational and one rotational degree of freedom (symmetry quotient
    reduces search by factor ≥ 6).
    """

    N_ABS_DIRS = 6
    N_REL_TURNS_4 = 4   # 2-bit encoding
    N_REL_TURNS_5 = 5   # 3-bit encoding (with 3 invalid states)

    BITS_PER_LINK_2 = 2  # 4-turn mode
    BITS_PER_LINK_3 = 3  # 5-turn mode

    # --- Absolute direction helpers (for exact solver) --------------------

    @staticmethod
    def get_vector_from_int(direction_int: int) -> np.ndarray:
        """Maps an integer 0–5 to an absolute direction vector."""
        if 0 <= direction_int < 6:
            return INT_TO_VEC[direction_int].copy()
        raise ValueError(f"Invalid direction int {direction_int}; must be 0–5")

    @staticmethod
    def get_non_reverse_directions(last_dir: int) -> List[int]:
        """
        Returns the 5 direction ints that are NOT the reverse of `last_dir`.
        Used for pruning the DFS (backbone can never immediately reverse).
        """
        reverse = last_dir ^ 1  # 0↔1, 2↔3, 4↔5
        return [d for d in range(6) if d != reverse]

    # --- Relative-turn helpers (for quantum algorithms) -------------------

    @staticmethod
    def n_qubits(n_beads: int, bits_per_link: int = 2) -> int:
        """Number of qubits needed for the relative-turn encoding."""
        n_links = max(n_beads - 2, 0)
        return bits_per_link * n_links

    @staticmethod
    def turn_sequence_to_coords(
        turn_codes: Sequence[int],
        n_directions: int = 4,
    ) -> List[np.ndarray]:
        """
        Convert a sequence of relative turn codes to 3D coordinates.

        Parameters
        ----------
        turn_codes : sequence of int
            Length N−2. Each element is a turn code (0–3 for 4-dir, 0–4 for 5-dir).
        n_directions : int
            Number of allowed relative turns (4 or 5).

        Returns
        -------
        coords : list of np.ndarray
            N absolute coordinates on the lattice (integer vectors).
        """
        coords = [
            np.array([0, 0, 0], dtype=np.int64),
            np.array([1, 0, 0], dtype=np.int64),
        ]
        frame = CoordinateFrame()  # initial heading = +x

        for tc in turn_codes:
            direction, frame = frame.apply_turn(tc, n_directions)
            coords.append(coords[-1] + direction)

        return coords

    @staticmethod
    def bitstring_to_turn_codes(
        bitstring: Sequence[int],
        bits_per_link: int = 2,
    ) -> List[int]:
        """
        Parse a raw bitstring (0/1 values) into a list of turn codes.

        Parameters
        ----------
        bitstring : sequence of int (0 or 1)
            Length must be bits_per_link × n_links.
        bits_per_link : int
            2 for 4-direction mode, 3 for 5-direction mode.

        Returns
        -------
        turn_codes : list of int
        """
        n_bits = len(bitstring)
        if n_bits % bits_per_link != 0:
            raise ValueError(
                f"Bitstring length {n_bits} not divisible by bits_per_link={bits_per_link}"
            )
        n_links = n_bits // bits_per_link
        codes = []
        for i in range(n_links):
            chunk = bitstring[i * bits_per_link : (i + 1) * bits_per_link]
            val = 0
            for bit in chunk:
                val = (val << 1) | int(bit)
            codes.append(val)
        return codes

    @staticmethod
    def decode_path(move_ints: Sequence[int]) -> List[np.ndarray]:
        """
        Decode a list of absolute direction ints (0–5) into coordinates.

        Bead 0 at origin, bead 1 at (1,0,0) (fixed).
        move_ints[k] gives the direction for the bond from bead k+1 → bead k+2.
        """
        coords = [
            np.array([0, 0, 0], dtype=np.int64),
            np.array([1, 0, 0], dtype=np.int64),
        ]
        current = np.array([1, 0, 0], dtype=np.int64)
        for m in move_ints:
            vec = CubicLattice.get_vector_from_int(m)
            current = current + vec
            coords.append(current.copy())
        return coords

    # --- Validity checks --------------------------------------------------

    @staticmethod
    def is_self_avoiding(coords: List[np.ndarray]) -> bool:
        """Check that no two beads occupy the same lattice site."""
        seen = set()
        for c in coords:
            key = tuple(c)
            if key in seen:
                return False
            seen.add(key)
        return True

    @staticmethod
    def count_contacts(
        coords: List[np.ndarray],
        sequence: str,
        contact_type: str = "HH",
    ) -> int:
        """
        Count the number of topological (non-bonded, distance-1) contacts
        of the specified type.

        Parameters
        ----------
        contact_type : str
            "HH" for hydrophobic–hydrophobic only,
            "all" for any non-bonded contact.
        """
        n = len(coords)
        if n == 0:
            return 0
            
        c = np.array(coords)
        # Compute squared distance matrix: shape (n, n)
        diffs = c[:, None, :] - c[None, :, :]
        d2 = np.sum(diffs ** 2, axis=2)
        
        # Mask for distance exactly 1
        d_mask = (d2 == 1)
        
        # Upper triangular mask for j >= i + 2 (non-bonded)
        tri_mask = np.triu(np.ones((n, n), dtype=bool), k=2)
        valid_contacts = d_mask & tri_mask
        
        if contact_type == "all":
            return int(np.sum(valid_contacts))
        elif contact_type == "HH":
            h_mask = np.array([res == "H" for res in sequence])
            hh_matrix = h_mask[:, None] & h_mask[None, :]
            return int(np.sum(valid_contacts & hh_matrix))
        return 0
