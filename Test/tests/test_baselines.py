"""
Tests for quantum_fold.algorithms.baselines
"""

import unittest
import numpy as np
from quantum_fold.core.protein import Protein
from quantum_fold.core.lattice import CubicLattice
from quantum_fold.algorithms.baselines import (
    ExactSolver,
    GreedyLocalSearch,
    SimulatedAnnealing,
    GeneticAlgorithm,
    ReplicaExchangeMC,
)


class TestExactSolver(unittest.TestCase):

    def test_hphp_optimal(self):
        """HPHP: H's at positions 0,2 cannot form d=1 contact on
        3D cubic lattice (min non-bonded d² = 2). Optimal E = 0."""
        p = Protein("HPHP")
        solver = ExactSolver(p)
        e, coords = solver.solve()
        self.assertEqual(e, 0.0, f"Expected 0.0, got {e}")
        self.assertTrue(CubicLattice.is_self_avoiding(coords))

    def test_hhpp_optimal(self):
        """HHPP: H0-H1 are bonded (no contact energy), H0-P2 and 
        H0-P3 are HP (energy 0). Optimal E = 0."""
        p = Protein("HHPP")
        solver = ExactSolver(p)
        e, coords = solver.solve()
        self.assertEqual(e, 0.0, f"Expected 0.0, got {e}")

    def test_hhhh_optimal(self):
        """HHHH: one non-bonded H-H contact achievable. Optimal E = -1."""
        p = Protein("HHHH")
        solver = ExactSolver(p)
        e, coords = solver.solve()
        self.assertEqual(e, -1.0, f"Expected -1.0, got {e}")

    def test_hhpphh_optimal(self):
        """HHPPHH should have optimal energy -2."""
        p = Protein("HHPPHH")
        solver = ExactSolver(p)
        e, coords = solver.solve()
        self.assertLessEqual(e, -2.0, f"Expected ≤ -2.0, got {e}")

    def test_all_coords_unique(self):
        """Exact solution must be self-avoiding."""
        p = Protein("HPHPHH")
        solver = ExactSolver(p)
        _, coords = solver.solve()
        self.assertTrue(CubicLattice.is_self_avoiding(coords))

    def test_all_polar_zero_energy(self):
        """All-P sequence → no contacts → E = 0."""
        p = Protein("PPPP")
        solver = ExactSolver(p)
        e, _ = solver.solve()
        self.assertEqual(e, 0.0)


class TestGreedyLocalSearch(unittest.TestCase):

    def test_finds_nonpositive_energy(self):
        """GLS should find energy ≤ 0 for a simple HP sequence."""
        p = Protein("HPHP")
        gls = GreedyLocalSearch(p, n_restarts=50, max_steps=100, seed=42)
        e, coords, _ = gls.solve()
        self.assertLessEqual(e, 0.0)
        if len(coords) > 0:
            self.assertTrue(CubicLattice.is_self_avoiding(coords))


class TestSimulatedAnnealing(unittest.TestCase):

    def test_finds_reasonable_energy(self):
        p = Protein("HPHP")
        sa = SimulatedAnnealing(
            p, t_start=3.0, t_end=0.01,
            n_steps=1000, n_restarts=5, seed=42,
        )
        e, coords, info = sa.solve()
        self.assertLessEqual(e, 0.0)


class TestGeneticAlgorithm(unittest.TestCase):

    def test_finds_reasonable_energy(self):
        p = Protein("HPHP")
        ga = GeneticAlgorithm(
            p, pop_size=30, n_generations=50,
            mutation_rate=0.15, seed=42,
        )
        e, coords, info = ga.solve()
        # GA should at least find a valid fold
        self.assertIsInstance(e, float)
        self.assertGreater(len(coords), 0)


class TestReplicaExchangeMC(unittest.TestCase):

    def test_finds_reasonable_energy(self):
        p = Protein("HPHP")
        remc = ReplicaExchangeMC(
            p, n_replicas=4, t_min=0.1, t_max=5.0,
            n_steps=200, n_exchanges=20, seed=42,
        )
        e, coords, info = remc.solve()
        self.assertLessEqual(e, 0.0)
        self.assertGreater(info["swap_count"], 0)


if __name__ == "__main__":
    unittest.main()
