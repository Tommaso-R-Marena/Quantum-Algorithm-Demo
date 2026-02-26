"""
real_benchmarks.py
Real protein benchmark set for evaluating the QFA pipeline.

Each benchmark has a PDB ID, chain, and sequence. The native Calpha
coordinates are downloaded from the RCSB PDB and used as ground truth
for RMSD/TM-score evaluation.

Benchmark proteins are selected to span a range of sizes, fold types,
and difficulty levels relevant to the quantum fragment assembly approach.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RealBenchmark:
    """A real protein benchmark."""
    name: str
    pdb_id: str
    chain: str
    sequence: str
    fold_class: str
    n_residues: int
    description: str
    difficulty: str  # "easy", "medium", "hard"


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark proteins
# ═══════════════════════════════════════════════════════════════════════════

class BenchmarkRegistry:
    """Registry for available real protein benchmarks."""
    
    def __init__(self):
        self._benchmarks: Dict[str, RealBenchmark] = {}

    def register(self, name: str, pdb_id: str, chain: str, seq: str, fold_class: str, desc: str, difficulty: str) -> None:
        """Register a new benchmark."""
        self._benchmarks[name] = RealBenchmark(
            name=name, pdb_id=pdb_id, chain=chain, sequence=seq,
            fold_class=fold_class, n_residues=len(seq),
            description=desc, difficulty=difficulty,
        )

    def get(self, name: str) -> RealBenchmark:
        """Get a benchmark by name."""
        if name not in self._benchmarks:
            available = ", ".join(sorted(self._benchmarks.keys()))
            raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
        return self._benchmarks[name]

    def list_all(
        self,
        max_length: Optional[int] = None,
        difficulty: Optional[str] = None,
    ) -> List[RealBenchmark]:
        """List available benchmarks with optional filtering."""
        benchmarks = sorted(self._benchmarks.values(), key=lambda b: b.n_residues)
        if max_length is not None:
            benchmarks = [b for b in benchmarks if b.n_residues <= max_length]
        if difficulty is not None:
            benchmarks = [b for b in benchmarks if b.difficulty == difficulty]
        return benchmarks


_registry = BenchmarkRegistry()


# Very small (10–15 residues): tractable for quantum simulation
_registry.register("chignolin", "5AWL", "A",
     "YYDPETGTWY",
     "beta-hairpin",
     "Chignolin (CLN025): 10-residue designed beta-hairpin, fastest-folding protein",
     "easy")

_registry.register("trp_zip", "1LE1", "A",
     "SWTWENGKWTWK",
     "beta-hairpin",
     "Trp-zip2: 12-residue designed beta-hairpin stabilised by tryptophan pairs",
     "easy")

# Small (16–20 residues): challenging for quantum, benchmark for classical
_registry.register("hp35_nle", "2F4K", "A",
     "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
     "3-helix-bundle",
     "HP35 Nle-Nle: villin headpiece subdomain, 35-residue 3-helix bundle",
     "hard")

_registry.register("trp_cage", "1L2Y", "A",
     "NLYIQWLKDGGPSSGRPPPS",
     "alpha+polyproline",
     "Trp-cage (TC5b): 20-residue miniprotein with alpha helix and polyproline",
     "medium")

_registry.register("bba5", "1T8J", "A",
     "EQYTAKQAVAAGFAQKLFIQPGDQE",
     "beta-beta-alpha",
     "BBA5: 23-residue beta-beta-alpha fold",
     "medium")

# Medium (25–35 residues): classical baselines, quantum fragments
_registry.register("ww_domain", "1PIN", "A",
     "KLPPGWEKRMSRSSGRVYYFNHITNASQWERPSG",
     "beta-sheet",
     "WW domain (Pin1): 34-residue all-beta fold with 3 antiparallel strands",
     "hard")

_registry.register("gb1_hairpin", "2OED", "A",
     "GEWTYDDATKTF",
     "beta-hairpin",
     "GB1 hairpin: 12-residue fragment from protein G",
     "easy")

_registry.register("alpha_helix_15", "NONE", "A",
     "AAQAAAAQAAAAQAA",
     "alpha-helix",
     "Designed 15-residue polyalanine alpha helix (no PDB; model structure)",
     "easy")


def get_real_benchmark(name: str) -> RealBenchmark:
    """Get a benchmark by name."""
    return _registry.get(name)


def list_real_benchmarks(
    max_length: Optional[int] = None,
    difficulty: Optional[str] = None,
) -> List[RealBenchmark]:
    """List available benchmarks with optional filtering."""
    return _registry.list_all(max_length=max_length, difficulty=difficulty)


def print_real_benchmark_table():
    """Print a summary table of all real protein benchmarks."""
    benchmarks = list_real_benchmarks()
    print(f"{'Name':<16} {'N':>3} {'PDB':>5} {'Fold Class':<20} {'Difficulty':<8}")
    print("-" * 60)
    for b in benchmarks:
        print(f"{b.name:<16} {b.n_residues:>3} {b.pdb_id:>5} "
              f"{b.fold_class:<20} {b.difficulty:<8}")


def load_benchmark_structure(
    name: str,
    output_dir: str = "structures",
) -> Dict:
    """
    Load a benchmark and its native structure.

    Downloads the PDB file if needed, parses it, and returns both
    the benchmark metadata and the native Calpha coordinates.

    Returns
    -------
    dict with:
      "benchmark": RealBenchmark
      "native_coords": np.ndarray (N, 3) or None
      "native_sequence": str
    """
    bench = get_real_benchmark(name)

    if bench.pdb_id == "NONE":
        # Generate model helix structure
        from ..core.backbone import build_ca_trace
        n = bench.n_residues
        phi = np.full(n, -1.05)  # alpha helix
        psi = np.full(n, -0.79)
        ca = build_ca_trace(phi, psi)
        return {
            "benchmark": bench,
            "native_coords": ca,
            "native_sequence": bench.sequence,
        }

    # Download and parse
    import os
    os.makedirs(output_dir, exist_ok=True)

    from .pdb_io import fetch_pdb, parse_pdb

    try:
        filepath = fetch_pdb(bench.pdb_id, output_dir=output_dir)
        parsed = parse_pdb(filepath, chain=bench.chain)

        return {
            "benchmark": bench,
            "native_coords": parsed["ca_coords"],
            "native_sequence": parsed["sequence"],
        }
    except Exception as e:
        print(f"  Could not load {bench.name}: {e}")
        return {
            "benchmark": bench,
            "native_coords": None,
            "native_sequence": bench.sequence,
        }
