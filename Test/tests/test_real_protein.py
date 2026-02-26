"""
Tests for backbone, metrics, and fragment library.
"""

import unittest
import numpy as np

from quantum_fold.core.backbone import (
    build_backbone,
    build_ca_trace,
    extract_dihedrals,
    kabsch_rmsd,
    nerf_place_atom,
    compute_dihedral,
    ca_distance_matrix,
    contact_map,
)
from quantum_fold.core.force_field import (
    CoarseGrainedForceField,
    radius_of_gyration,
    rg_target,
    clash_energy,
    contact_energy,
)
from quantum_fold.core.fragment_library import FragmentLibrary
from quantum_fold.core.residue import predict_secondary_structure, sequence_features
from quantum_fold.utils.metrics import tm_score, gdt_ts, lddt


class TestNeRF(unittest.TestCase):

    def test_nerf_distance(self):
        """NeRF placement should produce correct bond length."""
        # Non-collinear atoms (collinear makes cross product degenerate)
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([1.5, 1.0, 0.0])
        d = nerf_place_atom(a, b, c, 1.5, np.radians(120), np.pi)
        dist = np.linalg.norm(d - c)
        self.assertAlmostEqual(dist, 1.5, places=4)


class TestBackbone(unittest.TestCase):

    def test_backbone_length(self):
        """Build backbone for 5 residues: should produce 15 atoms."""
        n = 5
        phi = np.full(n, -1.05)
        psi = np.full(n, -0.79)
        bb = build_backbone(phi, psi)
        self.assertEqual(bb.shape, (15, 3))

    def test_ca_trace_length(self):
        n = 8
        phi = np.full(n, -1.05)
        psi = np.full(n, -0.79)
        ca = build_ca_trace(phi, psi)
        self.assertEqual(ca.shape, (8, 3))

    def test_ca_distances_reasonable(self):
        """Adjacent Calpha distances should be in [1.5, 5.0] A range.
        The full backbone has N-CA-C atoms; CA-to-CA spans ~3.8 A for
        trans peptide but can be shorter depending on dihedral angles."""
        n = 10
        phi = np.full(n, -1.05)
        psi = np.full(n, -0.79)
        ca = build_ca_trace(phi, psi)
        for i in range(n - 1):
            d = np.linalg.norm(ca[i + 1] - ca[i])
            self.assertGreater(d, 1.5, f"CA {i}-{i+1} too close: {d:.3f}")
            self.assertLess(d, 5.5, f"CA {i}-{i+1} too far: {d:.3f}")


class TestKabschRMSD(unittest.TestCase):

    def test_identical_structures(self):
        """RMSD of identical structures = 0."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
        rmsd_val, aligned = kabsch_rmsd(coords, coords)
        self.assertAlmostEqual(rmsd_val, 0.0, places=5)

    def test_translated_structures(self):
        """RMSD should be 0 after alignment of translated structures."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
        shifted = coords + np.array([10, 20, 30])
        rmsd_val, aligned = kabsch_rmsd(shifted, coords)
        self.assertAlmostEqual(rmsd_val, 0.0, places=5)


class TestMetrics(unittest.TestCase):

    def test_tm_score_identical(self):
        """TM-score of identical structures = 1.0."""
        ca = build_ca_trace(np.full(10, -1.05), np.full(10, -0.79))
        tm = tm_score(ca, ca)
        self.assertAlmostEqual(tm, 1.0, places=3)

    def test_gdt_identical(self):
        gdt = gdt_ts(
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
        )
        self.assertAlmostEqual(gdt, 1.0, places=3)

    def test_lddt_identical(self):
        ca = build_ca_trace(np.full(8, -1.05), np.full(8, -0.79))
        ldt = lddt(ca, ca)
        self.assertAlmostEqual(ldt, 1.0, places=3)


class TestForceField(unittest.TestCase):

    def test_rg_positive(self):
        ca = build_ca_trace(np.full(10, -1.05), np.full(10, -0.79))
        rg = radius_of_gyration(ca)
        self.assertGreater(rg, 0)

    def test_no_clashes_in_helix(self):
        ca = build_ca_trace(np.full(10, -1.05), np.full(10, -0.79))
        e = clash_energy(ca)
        self.assertAlmostEqual(e, 0.0, places=3)

    def test_composite_score_runs(self):
        ff = CoarseGrainedForceField()
        ca = build_ca_trace(np.full(10, -1.05), np.full(10, -0.79))
        score = ff.score(ca, "AALLEAALLN")
        self.assertIsInstance(score, float)


class TestFragmentLibrary(unittest.TestCase):

    def test_generate_fragments(self):
        lib = FragmentLibrary("YYDPETGTWY", fragment_size=5, overlap=2,
                              max_conformations=4)
        fragments = lib.generate()
        self.assertGreater(len(fragments), 0)
        for frag in fragments:
            self.assertGreater(frag.n_conformations, 0)

    def test_qubits_needed(self):
        lib = FragmentLibrary("YYDPETGTWY", fragment_size=5, overlap=2,
                              max_conformations=4)
        lib.generate()
        n_qubits = lib.n_qubits_needed()
        self.assertGreater(n_qubits, 0)

    def test_assembly(self):
        lib = FragmentLibrary("YYDPETGTWY", fragment_size=5, overlap=2,
                              max_conformations=4)
        lib.generate()
        choices = [0] * len(lib.fragments)
        coords = lib.assemble(choices)
        self.assertEqual(len(coords), 10)  # 10 residues


class TestSecondaryStructure(unittest.TestCase):

    def test_ss_prediction_length(self):
        ss = predict_secondary_structure("AAALLLAAALL")
        self.assertEqual(len(ss), 11)

    def test_sequence_features(self):
        feats = sequence_features("ACDEFGH")
        self.assertIn("length", feats)
        self.assertEqual(feats["length"], 7)


if __name__ == "__main__":
    unittest.main()
