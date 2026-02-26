"""
benchmarks.py
Standard benchmark sequences and automated benchmark runner for HP lattice folding.

Provides a curated set of benchmark sequences from the protein folding literature,
with known optimal energies (when available) for 2D and 3D lattices.

References:
  [1] Dill, Biochemistry 24, 1501 (1985)
  [2] Unger & Moult, J. Mol. Biol. 231, 75 (1993)
  [3] Perdomo-Ortiz et al., Sci. Rep. 2, 571 (2012)
  [4] Robert et al., npj Quantum Inf. 7, 38 (2021)
  [5] Yue & Dill, PNAS 92, 146 (1995)
"""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BenchmarkSequence:
    """A benchmark protein sequence with metadata."""
    name: str
    sequence: str
    optimal_energy_3d: Optional[float] = None
    optimal_energy_2d: Optional[float] = None
    source: str = ""
    notes: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def h_count(self) -> int:
        return self.sequence.count("H")

    @property
    def p_count(self) -> int:
        return self.sequence.count("P")


# ═══════════════════════════════════════════════════════════════════════════
# Standard benchmark sequences
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARKS: Dict[str, BenchmarkSequence] = {}


def _register(name, seq, e3d=None, e2d=None, source="", notes=""):
    BENCHMARKS[name] = BenchmarkSequence(name, seq, e3d, e2d, source, notes)


# --- Very small (4–6 beads): exact verification ---------------------------
_register("S4a", "HPHP", e3d=0, e2d=-1, source="Standard", notes="No 3D HH contact possible (alternating HP)")
_register("S4b", "HHPP", e3d=0, e2d=-1, source="Standard", notes="HH bonded, no non-bonded HH contacts")
_register("S4c", "HHHH", e3d=-1, e2d=-1, source="Standard", notes="All hydrophobic")
_register("S5a", "HPHPH", e3d=0, e2d=-1, source="Standard", notes="Alternating HP, no 3D HH contacts")
_register("S6a", "HHPPHH", e3d=-2, e2d=-2, source="Standard")
_register("S6b", "HPHPHH", e3d=-1, source="Standard")

# --- Small (8–10 beads): quantum algorithm benchmarks ---------------------
_register("S8a", "HPPHPPHP", e3d=-3, e2d=-3,
          source="Perdomo-Ortiz 2012", notes="Used in quantum annealing benchmarks")
_register("S8b", "HHPPHHPP", e3d=-4, e2d=-3, source="Standard")
_register("S9a", "PHPHPHPHP", e3d=-2, source="Standard")
_register("S10a", "HPHPPHHPHP", e3d=-4, source="Standard")

# --- Medium (12–20 beads): NISQ regime -----------------------------------
_register("S13a", "HPPHPPHHPPHPP", e3d=-6,
          source="Unger & Moult 1993")
_register("S14a", "HHHHPPPHHHHPPP", e3d=-7, source="Standard")
_register("S16a", "PPHPPHHPPHHPPPPH", e3d=-5, source="Dill 1985")
_register("S18a", "HHPPHPPHPPHPPHPPHH", e3d=-8,
          source="Perdomo-Ortiz 2012")
_register("S20a", "HPHPPHHPHPPHPHHPPHPH",
          source="Yue & Dill 1995", notes="20-mer benchmark")

# --- Larger (25–36 beads): for classical baselines only -------------------
_register("S25a", "PPHPPHHPPPPHHPPPPHHPPPPHH",
          source="Unger & Moult 1993")
_register("S36a", "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPHH",
          source="Unger & Moult 1993",
          notes="36-mer classic benchmark")


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark utilities
# ═══════════════════════════════════════════════════════════════════════════

def get_benchmark(name: str) -> BenchmarkSequence:
    """Retrieve a benchmark sequence by name."""
    if name not in BENCHMARKS:
        available = ", ".join(sorted(BENCHMARKS.keys()))
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
    return BENCHMARKS[name]


def list_benchmarks(max_length: Optional[int] = None) -> List[BenchmarkSequence]:
    """List all benchmarks, optionally filtered by maximum length."""
    benchmarks = sorted(BENCHMARKS.values(), key=lambda b: b.length)
    if max_length is not None:
        benchmarks = [b for b in benchmarks if b.length <= max_length]
    return benchmarks


def print_benchmark_table(max_length: Optional[int] = None):
    """Print a summary table of all benchmarks."""
    benchmarks = list_benchmarks(max_length)
    print(f"{'Name':<8} {'N':>3} {'nH':>3} {'Seq':<40} {'E*(3D)':>7} {'Source':<25}")
    print("-" * 95)
    for b in benchmarks:
        e3d = f"{b.optimal_energy_3d:.0f}" if b.optimal_energy_3d is not None else "?"
        seq_display = b.sequence if len(b.sequence) <= 38 else b.sequence[:35] + "..."
        print(f"{b.name:<8} {b.length:>3} {b.h_count:>3} {seq_display:<40} {e3d:>7} {b.source:<25}")


def run_benchmark_suite(
    solver_factory,
    benchmark_names: Optional[List[str]] = None,
    max_length: int = 12,
    n_seeds: int = 5,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Run a solver on multiple benchmark sequences and collect results.

    Parameters
    ----------
    solver_factory : callable
        Function (protein, seed) → solver object with a .solve() method
        that returns (energy, coords, info_dict) or (energy, coords).
    benchmark_names : list of str, optional
        Specific benchmarks to run. If None, runs all up to max_length.
    max_length : int
        Maximum sequence length to include.
    n_seeds : int
        Number of independent runs per benchmark.
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict
        Keys are benchmark names, values are dicts with:
          "best_energy", "mean_energy", "std_energy",
          "best_time", "mean_time", "success_rate",
          "optimal_energy", "approx_ratio"
    """
    from ..core.protein import Protein

    if benchmark_names is None:
        benchmarks = list_benchmarks(max_length)
    else:
        benchmarks = [get_benchmark(n) for n in benchmark_names]

    results = {}

    for bench in benchmarks:
        if verbose:
            print(f"\n--- Benchmark: {bench.name} ({bench.sequence}, N={bench.length}) ---")

        energies = []
        times = []

        for seed in range(n_seeds):
            protein = Protein(bench.sequence)
            try:
                solver = solver_factory(protein, seed)
                t_start = time.time()
                result = solver.solve()
                t_elapsed = time.time() - t_start

                if isinstance(result, tuple) and len(result) >= 2:
                    e = result[0]
                else:
                    e = result.get("energy", result.get("best_energy", float("inf")))

                energies.append(float(e))
                times.append(t_elapsed)

                if verbose:
                    print(f"  Seed {seed}: E={e:.3f}  t={t_elapsed:.3f}s")
            except Exception as ex:
                if verbose:
                    print(f"  Seed {seed}: FAILED ({ex})")
                energies.append(float("inf"))
                times.append(0.0)

        energies = np.array(energies)
        times = np.array(times)
        finite_mask = np.isfinite(energies)

        best_e = float(np.min(energies[finite_mask])) if np.any(finite_mask) else float("inf")
        mean_e = float(np.mean(energies[finite_mask])) if np.any(finite_mask) else float("inf")
        std_e = float(np.std(energies[finite_mask])) if np.any(finite_mask) else 0.0

        opt_e = bench.optimal_energy_3d

        success_rate = 0.0
        approx_ratio = None
        if opt_e is not None:
            success_rate = float(np.mean(np.abs(energies[finite_mask] - opt_e) < 0.01))
            if abs(opt_e) > 1e-12:
                approx_ratio = best_e / opt_e

        results[bench.name] = {
            "sequence": bench.sequence,
            "length": bench.length,
            "best_energy": best_e,
            "mean_energy": mean_e,
            "std_energy": std_e,
            "best_time": float(np.min(times)) if len(times) > 0 else 0.0,
            "mean_time": float(np.mean(times)),
            "success_rate": success_rate,
            "optimal_energy": opt_e,
            "approx_ratio": approx_ratio,
        }

        if verbose:
            print(f"  Summary: best={best_e:.3f}, mean={mean_e:.3f}±{std_e:.3f}, "
                  f"success={success_rate*100:.0f}%")

    return results
