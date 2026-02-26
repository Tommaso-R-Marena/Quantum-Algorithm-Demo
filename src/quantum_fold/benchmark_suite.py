"""
benchmark_suite.py
Rigorous benchmarking and ablation study for the QFA pipeline.

Runs the hybrid pipeline on a set of real proteins, computes structural
quality metrics (RMSD, TM-score, GDT-TS, lDDT), performs ablation
studies on each force field component, and generates publication-ready
output tables with confidence intervals.

Usage:
  python -m quantum_fold.benchmark_suite [--quick] [--ablation]
"""

from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    protein: str
    n_residues: int
    method: str
    rmsd: float = float("nan")
    tm_score: float = float("nan")
    gdt_ts: float = float("nan")
    lddt: float = float("nan")
    contact_f1: float = float("nan")
    total_energy: float = float("nan")
    rg: float = float("nan")
    rg_target: float = float("nan")
    n_fragments: int = 0
    n_qubits: int = 0
    time_seconds: float = 0.0
    n_contacts: int = 0


def run_single_benchmark(
    name: str,
    method: str = "sa",
    config: Optional[Dict] = None,
    seed: int = 42,
) -> BenchmarkResult:
    """Run pipeline on a single benchmark protein."""
    from quantum_fold.utils.real_benchmarks import load_benchmark_structure
    from quantum_fold.algorithms.hybrid_pipeline import HybridPipeline
    from quantum_fold.utils.metrics import (
        rmsd as compute_rmsd,
        tm_score as compute_tm,
        gdt_ts as compute_gdt,
        lddt as compute_lddt,
        contact_overlap,
    )
    from quantum_fold.core.force_field import radius_of_gyration, rg_target

    data = load_benchmark_structure(name, output_dir="benchmark_structures")
    bench = data["benchmark"]
    native = data["native_coords"]
    seq = data.get("native_sequence", bench.sequence) or bench.sequence

    cfg = {
        "method": method,
        "fragment_size": 5,
        "overlap": 2,
        "max_conformations": 8,
        "n_rama_bins": 4,
        "use_diffusion": False,
        "n_refine_steps": 50,  # Reduced for speed
        "seed": seed,
        "sa_steps": 3000,      # Reduced for speed
    }
    if config:
        cfg.update(config)

    pipeline = HybridPipeline(
        sequence=seq,
        native_coords=native,
        config=cfg,
    )

    result = pipeline.run()
    pred = result["predicted_coords"]

    br = BenchmarkResult(
        protein=name,
        n_residues=len(seq),
        method=method,
        total_energy=result.get("total_energy", float("nan")),
        rg=result.get("rg", float("nan")),
        rg_target=rg_target(len(seq)),
        n_fragments=result.get("n_fragments", 0),
        n_qubits=result.get("n_qubits", 0),
        time_seconds=result.get("time_total", 0),
        n_contacts=result.get("n_contacts", 0),
    )

    if native is not None and len(native) == len(pred):
        br.rmsd = compute_rmsd(pred, native)
        br.tm_score = compute_tm(pred, native)
        br.gdt_ts = compute_gdt(pred, native)
        br.lddt = compute_lddt(pred, native)
        _, _, f1 = contact_overlap(pred, native)
        br.contact_f1 = f1

    return br


def run_multi_seed_benchmark(
    name: str,
    method: str = "sa",
    n_seeds: int = 5,
    config: Optional[Dict] = None,
) -> List[BenchmarkResult]:
    """Run benchmark with multiple random seeds for statistical robustness."""
    results = []
    for seed in range(42, 42 + n_seeds):
        try:
            cfg = dict(config or {})
            cfg["seed"] = seed
            br = run_single_benchmark(name, method, cfg, seed)
            results.append(br)
            print(f"  Seed {seed}: RMSD={br.rmsd:.3f}, TM={br.tm_score:.3f}")
        except Exception as e:
            print(f"  Seed {seed}: FAILED — {e}")
    return results


def compute_statistics(results: List[BenchmarkResult]) -> Dict[str, str]:
    """Compute mean ± std for key metrics."""
    if not results:
        return {}

    metrics = {}
    for field_name in ["rmsd", "tm_score", "gdt_ts", "lddt", "contact_f1"]:
        values = [getattr(r, field_name) for r in results
                  if not np.isnan(getattr(r, field_name))]
        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            metrics[field_name] = f"{mean:.3f} +/- {std:.3f}"
        else:
            metrics[field_name] = "N/A"

    return metrics


def run_ablation_study(
    name: str,
    seed: int = 42,
) -> Dict[str, BenchmarkResult]:
    """
    Ablation study: disable each force field component one at a time
    and measure the impact on structure quality.

    Returns dict: component_name -> BenchmarkResult
    """
    from quantum_fold.core.force_field import CoarseGrainedForceField

    components = ["dfire", "hbond", "torsion", "lj", "elec", "rg", "solv", "cb_contact"]
    results = {}

    # Full model (baseline)
    print("\n  Ablation: Full model")
    results["full"] = run_single_benchmark(name, "sa", seed=seed)

    # Ablate each component
    for comp in components:
        print(f"\n  Ablation: without {comp}")
        cfg = {f"w_{comp}": 0.0}
        try:
            results[f"no_{comp}"] = run_single_benchmark(
                name, "sa", config=cfg, seed=seed
            )
        except Exception as e:
            print(f"    FAILED: {e}")

    return results


def print_benchmark_table(results: Dict[str, List[BenchmarkResult]]):
    """Print a formatted benchmark results table."""
    print("\n" + "=" * 90)
    print(f"  {'Protein':<15} {'N':>4} {'Method':>8} {'RMSD':>8} "
          f"{'TM':>8} {'GDT-TS':>8} {'lDDT':>8} {'F1':>8} {'Time':>8}")
    print("-" * 90)

    for protein, result_list in results.items():
        stats = compute_statistics(result_list)
        n = result_list[0].n_residues if result_list else 0
        method = result_list[0].method if result_list else ""
        print(f"  {protein:<15} {n:>4} {method:>8} "
              f"{stats.get('rmsd', 'N/A'):>8} "
              f"{stats.get('tm_score', 'N/A'):>8} "
              f"{stats.get('gdt_ts', 'N/A'):>8} "
              f"{stats.get('lddt', 'N/A'):>8} "
              f"{stats.get('contact_f1', 'N/A'):>8} "
              f"{result_list[0].time_seconds:>7.1f}s" if result_list else "")

    print("=" * 90)


def print_ablation_table(ablation: Dict[str, BenchmarkResult]):
    """Print ablation study results."""
    print("\n" + "=" * 80)
    print("  ABLATION STUDY")
    print("=" * 80)
    print(f"  {'Config':<20} {'RMSD':>8} {'TM':>8} {'GDT-TS':>8} "
          f"{'Energy':>10} {'dTM':>8}")
    print("-" * 80)

    baseline_tm = ablation.get("full", BenchmarkResult("", 0, "")).tm_score

    for config, br in ablation.items():
        dtm = br.tm_score - baseline_tm if not np.isnan(br.tm_score) else float("nan")
        rmsd_s = f"{br.rmsd:.3f}" if not np.isnan(br.rmsd) else "N/A"
        tm_s = f"{br.tm_score:.3f}" if not np.isnan(br.tm_score) else "N/A"
        gdt_s = f"{br.gdt_ts:.3f}" if not np.isnan(br.gdt_ts) else "N/A"
        dtm_s = f"{dtm:+.3f}" if not np.isnan(dtm) else "N/A"

        print(f"  {config:<20} {rmsd_s:>8} {tm_s:>8} {gdt_s:>8} "
              f"{br.total_energy:>10.2f} {dtm_s:>8}")

    print("=" * 80)


def main():
    """Run the full benchmark suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Suite")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer seeds, smaller proteins")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study")
    parser.add_argument("--method", type=str, default="sa",
                        choices=["sa", "greedy", "exact"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--proteins", type=str, default=None,
                        help="Comma-separated list of protein names")
    args = parser.parse_args()

    # Force UTF-8
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    os.makedirs("benchmark_results", exist_ok=True)

    # Select proteins
    if args.proteins:
        protein_names = [p.strip() for p in args.proteins.split(",")]
    elif args.quick:
        protein_names = ["chignolin", "trp_zip", "gb1_hairpin"]
    else:
        protein_names = [
            "chignolin", "trp_zip", "gb1_hairpin",
            "alpha_helix_15", "trp_cage",
        ]

    start_time = time.time()
    all_results: Dict[str, List[BenchmarkResult]] = {}

    print("=" * 60)
    print("  QUANTUM FRAGMENT ASSEMBLY BENCHMARK SUITE")
    print("=" * 60)

    # Main benchmarks
    for name in protein_names:
        print(f"\n--- Benchmarking: {name} ---")
        try:
            results = run_multi_seed_benchmark(
                name, args.method, n_seeds=args.seeds
            )
            all_results[name] = results
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # Print results table
    print_benchmark_table(all_results)

    # Ablation study
    if args.ablation and protein_names:
        ablation_target = protein_names[0]
        print(f"\n--- Ablation study on: {ablation_target} ---")
        ablation = run_ablation_study(ablation_target)
        print_ablation_table(ablation)

        # Save ablation
        abl_data = {}
        for k, v in ablation.items():
            abl_data[k] = asdict(v)
        with open("benchmark_results/ablation.json", "w") as f:
            json.dump(abl_data, f, indent=2, default=str)

    # Save all results
    save_data = {}
    for protein, results in all_results.items():
        save_data[protein] = [asdict(r) for r in results]
        save_data[protein + "_stats"] = compute_statistics(results)

    with open("benchmark_results/benchmark_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time:.1f}s")
    print(f"Results saved: benchmark_results/benchmark_results.json")


if __name__ == "__main__":
    main()
