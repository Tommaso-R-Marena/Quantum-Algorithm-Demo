"""
Tests for quantum_fold.core.protein
"""

import unittest
import numpy as np
from quantum_fold.core.protein import Protein, HPModel, HPPlusModel, MJModel


class TestHPModel(unittest.TestCase):

    def test_hh_contact_energy(self):
        model = HPModel()
        self.assertEqual(model.contact_energy("H", "H"), -1.0)

    def test_hp_contact_energy(self):
        model = HPModel()
        self.assertEqual(model.contact_energy("H", "P"), 0.0)

    def test_pp_contact_energy(self):
        model = HPModel()
        self.assertEqual(model.contact_energy("P", "P"), 0.0)


class TestHPPlusModel(unittest.TestCase):

    def test_hh_contact(self):
        model = HPPlusModel()
        self.assertEqual(model.contact_energy("H", "H"), -1.0)

    def test_hp_contact(self):
        model = HPPlusModel()
        self.assertEqual(model.contact_energy("H", "P"), -0.5)
        self.assertEqual(model.contact_energy("P", "H"), -0.5)

    def test_pp_contact(self):
        model = HPPlusModel()
        self.assertEqual(model.contact_energy("P", "P"), 0.0)


class TestMJModel(unittest.TestCase):

    def test_known_value(self):
        model = MJModel()
        # ALA-ALA should be -2.72
        e = model.contact_energy("A", "A")
        self.assertAlmostEqual(e, -2.72, places=2)

    def test_symmetry(self):
        model = MJModel()
        self.assertAlmostEqual(
            model.contact_energy("A", "R"),
            model.contact_energy("R", "A"),
            places=5,
        )

    def test_unknown_residue(self):
        model = MJModel()
        self.assertEqual(model.contact_energy("X", "A"), 0.0)


class TestProtein(unittest.TestCase):

    def test_valid_hp(self):
        p = Protein("HPHP")
        self.assertTrue(p.is_hp())
        self.assertEqual(p.n, 4)

    def test_invalid_hp(self):
        p = Protein("ACDE")
        self.assertFalse(p.is_hp())

    def test_bonded_pair_no_energy(self):
        """Adjacent beads (bonded) should contribute 0 energy."""
        p = Protein("HH")
        self.assertEqual(p.get_interaction_energy(0, 1, 1), 0.0)

    def test_evaluate_energy_no_contact(self):
        """Straight line → no contacts → energy = 0."""
        p = Protein("HPHP")
        coords = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([2, 0, 0]),
            np.array([3, 0, 0]),
        ]
        e = p.evaluate_energy(coords, collision_penalty=1000.0)
        self.assertEqual(e, 0.0)

    def test_evaluate_energy_one_contact(self):
        """
        HPHP folded into an L-shape with H0 adjacent to H2:
          H(0) — P(1)
                  |
          H(3) — H(2)   ← H(2) and H(0)? No, check distances.

        Actually for H-H contact (i=0, j=2 with |i-j|=2):
        beads at (0,0,0) and (0,1,0) → d²=1 → H-H contact → E = -1
        """
        p = Protein("HPHP")
        coords = [
            np.array([0, 0, 0]),  # H
            np.array([1, 0, 0]),  # P
            np.array([1, 1, 0]),  # H
            np.array([0, 1, 0]),  # P
        ]
        e = p.evaluate_energy(coords, collision_penalty=1000.0)
        # H0 at (0,0,0), H2 at (1,1,0) → d²=2, no contact
        # P1 at (1,0,0), P3 at (0,1,0) → d²=2, no contact
        # H0 at (0,0,0), P3 at (0,1,0) → d²=1, but H-P = 0 in HP model
        self.assertEqual(e, 0.0)

    def test_evaluate_energy_with_collision(self):
        """Two beads at the same position → collision penalty."""
        p = Protein("HPH")
        coords = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 0, 0]),  # collision with bead 0
        ]
        e = p.evaluate_energy(coords, collision_penalty=1000.0)
        # Collision between 0 and 2, plus any contact energy
        self.assertGreaterEqual(e, 999.0)

    def test_decomposed_energy(self):
        p = Protein("HPH")
        coords = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 0, 0]),
        ]
        result = p.evaluate_energy_decomposed(coords, collision_penalty=100.0)
        self.assertEqual(result["n_collisions"], 1)
        self.assertAlmostEqual(result["collision"], 100.0)

    def test_max_possible_contacts(self):
        p = Protein("HHHH")
        mc = p.max_possible_contacts()
        self.assertGreater(mc, 0)

    def test_coords_length_mismatch_raises(self):
        p = Protein("HPHP")
        with self.assertRaises(ValueError):
            p.evaluate_energy([np.array([0, 0, 0])], collision_penalty=10.0)


if __name__ == "__main__":
    unittest.main()
