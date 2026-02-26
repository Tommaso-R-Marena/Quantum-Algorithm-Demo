"""
Tests for quantum_fold.core.encoding
"""

import unittest
import numpy as np
from quantum_fold.core.encoding import (
    bitstring_to_coords,
    coords_to_bitstring,
    enumerate_all_bitstrings,
    compute_classical_cost_vector,
    find_ground_state_bitstring,
)
from quantum_fold.core.lattice import CubicLattice


class TestBitstringToCoords(unittest.TestCase):

    def test_correct_number_of_coords(self):
        """N beads with 2-bit encoding → (N-2)*2 bits → N coords."""
        for n in range(3, 8):
            n_links = n - 2
            bits = [0] * (n_links * 2)
            coords = bitstring_to_coords(bits, n, bits_per_link=2)
            self.assertEqual(len(coords), n, f"Failed for n={n}")

    def test_no_aliasing(self):
        """Modifying one coordinate should not affect others."""
        bits = [0, 0, 0, 0]  # 4 beads
        coords = bitstring_to_coords(bits, 4, bits_per_link=2)
        original_0 = coords[0].copy()
        coords[1][0] = 999
        np.testing.assert_array_equal(coords[0], original_0)

    def test_all_straight_gives_line(self):
        """All-zero bitstring (straight turns) → linear chain."""
        bits = [0, 0, 0, 0, 0, 0]  # 3 links, 5 beads
        coords = bitstring_to_coords(bits, 5, bits_per_link=2)
        for i, c in enumerate(coords):
            np.testing.assert_array_equal(c, [i, 0, 0])

    def test_handles_short_bitstring(self):
        """Short bitstring should be padded with zeros."""
        coords = bitstring_to_coords([0], 4, bits_per_link=2)
        self.assertEqual(len(coords), 4)

    def test_handles_long_bitstring(self):
        """Long bitstring should be truncated."""
        coords = bitstring_to_coords([0] * 100, 4, bits_per_link=2)
        self.assertEqual(len(coords), 4)


class TestCoordsTobitstring(unittest.TestCase):

    def test_roundtrip_all_bitstrings(self):
        """Test round-trip for all 2-bit bitstrings with 4 beads."""
        n_beads = 4
        n_qubits = CubicLattice.n_qubits(n_beads, 2)
        all_bs = enumerate_all_bitstrings(n_qubits)

        for bs in all_bs:
            bs_list = list(bs)
            coords = bitstring_to_coords(bs_list, n_beads, bits_per_link=2)
            recovered = coords_to_bitstring(coords, bits_per_link=2)
            self.assertEqual(
                recovered, bs_list,
                f"Round-trip failed for bitstring {bs_list}"
            )


class TestCostVector(unittest.TestCase):

    def test_cost_vector_length(self):
        """Cost vector should have 2^n_qubits entries."""
        n_beads = 4
        n_qubits = CubicLattice.n_qubits(n_beads, 2)
        costs = compute_classical_cost_vector(n_beads, "HPHP")
        self.assertEqual(len(costs), 2 ** n_qubits)

    def test_ground_state_is_minimum(self):
        """find_ground_state_bitstring should match min(cost_vector)."""
        n_beads = 4
        costs = compute_classical_cost_vector(n_beads, "HPHP")
        bs, e, coords = find_ground_state_bitstring(n_beads, "HPHP")
        self.assertAlmostEqual(e, min(costs))


class TestEnumerateAllBitstrings(unittest.TestCase):

    def test_shape(self):
        result = enumerate_all_bitstrings(3)
        self.assertEqual(result.shape, (8, 3))

    def test_all_binary(self):
        result = enumerate_all_bitstrings(4)
        self.assertTrue(np.all((result == 0) | (result == 1)))


if __name__ == "__main__":
    unittest.main()
