"""
Tests for quantum_fold.core.lattice
"""

import unittest
import numpy as np
from quantum_fold.core.lattice import (
    CubicLattice,
    CoordinateFrame,
    INT_TO_VEC,
)


class TestCoordinateFrame(unittest.TestCase):
    """Tests for the relative-turn coordinate frame."""

    def test_initial_frame_is_orthonormal(self):
        frame = CoordinateFrame()
        # Check orthogonality
        self.assertEqual(np.dot(frame.forward, frame.left), 0)
        self.assertEqual(np.dot(frame.forward, frame.up), 0)
        self.assertEqual(np.dot(frame.left, frame.up), 0)
        # Check unit length
        self.assertEqual(np.linalg.norm(frame.forward), 1)
        self.assertEqual(np.linalg.norm(frame.left), 1)
        self.assertEqual(np.linalg.norm(frame.up), 1)

    def test_all_turns_produce_unit_vectors(self):
        """Every turn from every possible frame must produce a unit vector."""
        frame = CoordinateFrame()
        for turn_code in range(4):
            direction, new_frame = frame.apply_turn(turn_code, n_directions=4)
            norm = np.linalg.norm(direction)
            self.assertAlmostEqual(norm, 1.0, places=10,
                                   msg=f"Turn {turn_code} produced non-unit vector {direction}")
            # Check new frame is still orthonormal
            self.assertAlmostEqual(np.dot(new_frame.forward, new_frame.left), 0)
            self.assertAlmostEqual(np.dot(new_frame.forward, new_frame.up), 0)
            self.assertAlmostEqual(np.dot(new_frame.left, new_frame.up), 0)

    def test_no_turn_produces_zero_vector(self):
        """Critical fix: ensure no turn ever produces [0,0,0]."""
        frame = CoordinateFrame()
        for turn_code in range(4):
            direction, _ = frame.apply_turn(turn_code, n_directions=4)
            self.assertFalse(
                np.array_equal(direction, [0, 0, 0]),
                f"Turn {turn_code} produced zero vector!"
            )

    def test_straight_preserves_direction(self):
        frame = CoordinateFrame()
        direction, new_frame = frame.apply_turn(0, n_directions=4)
        np.testing.assert_array_equal(direction, [1, 0, 0])
        np.testing.assert_array_equal(new_frame.forward, frame.forward)

    def test_four_left_turns_return_to_start(self):
        """Four left turns should form a square, returning to original heading."""
        frame = CoordinateFrame()
        for _ in range(4):
            _, frame = frame.apply_turn(1, n_directions=4)  # turn left
        np.testing.assert_array_equal(frame.forward, [1, 0, 0])


class TestCubicLattice(unittest.TestCase):
    """Tests for the main lattice class."""

    def test_get_vector_from_int_all_directions(self):
        """All 6 directions produce valid unit vectors."""
        for d in range(6):
            vec = CubicLattice.get_vector_from_int(d)
            self.assertEqual(np.linalg.norm(vec), 1)

    def test_get_vector_from_int_invalid_raises(self):
        with self.assertRaises(ValueError):
            CubicLattice.get_vector_from_int(6)

    def test_non_reverse_directions(self):
        """get_non_reverse_directions should return exactly 5 directions."""
        for d in range(6):
            non_rev = CubicLattice.get_non_reverse_directions(d)
            self.assertEqual(len(non_rev), 5)
            # Should not contain the reverse
            reverse = d ^ 1
            self.assertNotIn(reverse, non_rev)

    def test_n_qubits(self):
        self.assertEqual(CubicLattice.n_qubits(4, bits_per_link=2), 4)
        self.assertEqual(CubicLattice.n_qubits(3, bits_per_link=2), 2)
        self.assertEqual(CubicLattice.n_qubits(2, bits_per_link=2), 0)

    def test_turn_sequence_to_coords_straight_line(self):
        """All straight turns → beads in a line along +x."""
        coords = CubicLattice.turn_sequence_to_coords([0, 0, 0], n_directions=4)
        self.assertEqual(len(coords), 5)
        for i, c in enumerate(coords):
            np.testing.assert_array_equal(c, [i, 0, 0])

    def test_is_self_avoiding_true(self):
        coords = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])]
        self.assertTrue(CubicLattice.is_self_avoiding(coords))

    def test_is_self_avoiding_false(self):
        coords = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 0])]
        self.assertFalse(CubicLattice.is_self_avoiding(coords))

    def test_decode_path(self):
        """decode_path with no moves → 2 beads."""
        coords = CubicLattice.decode_path([])
        self.assertEqual(len(coords), 2)
        np.testing.assert_array_equal(coords[0], [0, 0, 0])
        np.testing.assert_array_equal(coords[1], [1, 0, 0])

    def test_bitstring_to_turn_codes(self):
        codes = CubicLattice.bitstring_to_turn_codes([0, 1, 1, 0], bits_per_link=2)
        self.assertEqual(codes, [1, 2])  # 01=1, 10=2

    def test_count_contacts(self):
        # L-shaped fold: bead 0 at (0,0,0), bead 1 at (1,0,0), bead 2 at (1,1,0)
        # No contacts possible with only 3 beads
        coords = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])]
        self.assertEqual(CubicLattice.count_contacts(coords, "HHH", "HH"), 0)


class TestRoundTrip(unittest.TestCase):
    """Round-trip tests: bitstring → coords → bitstring."""

    def test_roundtrip_straight(self):
        """Straight-line bitstring should round-trip."""
        from quantum_fold.core.encoding import bitstring_to_coords, coords_to_bitstring

        bits = [0, 0, 0, 0]  # 2 links, both straight
        coords = bitstring_to_coords(bits, n_beads=4, bits_per_link=2)
        self.assertEqual(len(coords), 4)

        recovered = coords_to_bitstring(coords, bits_per_link=2)
        self.assertEqual(recovered, bits)


if __name__ == "__main__":
    unittest.main()
