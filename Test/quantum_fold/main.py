"""
main.py
Entry point for Quantum Protein Folding Experiments.

Supports two modes:
  --mode lattice : HP lattice model (original)
  --mode real    : Real protein structure prediction via Quantum Fragment Assembly

Usage:
  # Lattice mode (HP model)
  python -m quantum_fold.main --mode lattice --seq HPHP --algo all

  # Real protein mode
  python -m quantum_fold.main --mode real --seq YYDPETGTWY --method sa
  python -m quantum_fold.main --mode real --benchmark chignolin --method vqe
  python -m quantum_fold.main --mode real --list-benchmarks
"""

from __future__ import annotations

import argparse
import time
import json
import os
import sys
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Protein Folding Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lattice HP model
  python -m quantum_fold.main --mode lattice --seq HPHP --algo all

  # Real protein (SA baseline)
  python -m quantum_fold.main --mode real --seq YYDPETGTWY --method sa

  # Real protein benchmark
  python -m quantum_fold.main --mode real --benchmark chignolin --method sa

  # List available benchmarks
  python -m quantum_fold.main --mode real --list-benchmarks
        """,
    )
    parser.add_argument("--mode", type=str, default="real", choices=["lattice", "real"])
    parser.add_argument("--seq", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--pdb", type=str, default=None, help="PDB ID for native structure")

    # Lattice-mode args
    parser.add_argument("--algo", type=str, default="all")
    parser.add_argument("--model", type=str, default="HP", choices=["HP", "HP+", "MJ"])

    # Real-mode args
    parser.add_argument("--method", type=str, default="sa",
                        choices=["exact", "greedy", "sa", "vqe", "qaoa", "all"])
    parser.add_argument("--fragment-size", type=int, default=5)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--max-conformations", type=int, default=8)
    parser.add_argument("--use-diffusion", action="store_true")

    # Common args
    parser.add_argument("--shots", type=int, default=300)
    parser.add_argument("--iter", type=int, default=50)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--cvar-alpha", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    # Force UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "lattice":
        _run_lattice(args)
    else:
        _run_real(args)


def _run_real(args):
    """Run the real protein pipeline."""
    from quantum_fold.utils.real_benchmarks import (
        print_real_benchmark_table,
        get_real_benchmark,
        load_benchmark_structure,
        list_real_benchmarks,
    )
    from quantum_fold.algorithms.hybrid_pipeline import HybridPipeline
    from quantum_fold.utils.pdb_io import write_ca_pdb, fetch_pdb, parse_pdb
    from quantum_fold.utils.metrics import print_metrics

    if args.list_benchmarks:
        print("\nReal Protein Benchmarks:")
        print_real_benchmark_table()

        # Also list lattice benchmarks
        print("\nHP Lattice Benchmarks:")
        from quantum_fold.utils.benchmarks import print_benchmark_table
        print_benchmark_table()
        return

    # Determine sequence and native
    native_coords = None
    seq = args.seq

    if args.benchmark:
        data = load_benchmark_structure(args.benchmark, output_dir=args.outdir)
        bench = data["benchmark"]
        seq = bench.sequence
        native_coords = data["native_coords"]
        print(f"Benchmark: {bench.name} ({bench.description})")

    elif args.pdb:
        filepath = fetch_pdb(args.pdb, output_dir=args.outdir)
        parsed = parse_pdb(filepath)
        native_coords = parsed["ca_coords"]
        seq = seq or parsed["sequence"]
        print(f"PDB: {args.pdb}, chain {parsed['chain']}, "
              f"N={parsed['n_residues']}")

    if not seq:
        seq = "YYDPETGTWY"  # chignolin default
        print(f"No sequence specified, using default: {seq}")

    # Run pipeline for each method
    methods = [args.method] if args.method != "all" else ["greedy", "sa"]

    all_results = {}
    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")

        config = {
            "method": method,
            "fragment_size": args.fragment_size,
            "overlap": args.overlap,
            "n_rama_bins": 4,
            "max_conformations": args.max_conformations,
            "use_diffusion": args.use_diffusion,
            "shots": args.shots,
            "max_iter": args.iter,
            "depth": args.depth,
            "cvar_alpha": args.cvar_alpha,
            "seed": args.seed,
            "n_refine_steps": 200,
        }

        pipeline = HybridPipeline(
            sequence=seq,
            native_coords=native_coords,
            config=config,
        )

        result = pipeline.run()
        all_results[method] = result

        # Export predicted structure
        pred_coords = result["predicted_coords"]
        filename = f"{args.outdir}/predicted_{method}_{seq[:10]}.pdb"
        write_ca_pdb(pred_coords, seq, filename, remarks=[
            f"Method: {method}",
            f"Energy: {result['total_energy']:.3f}",
        ])
        print(f"  Exported PDB: {filename}")

        # Generate high-res 3D Visualisations
        try:
            from quantum_fold.utils.visualization import plot_pipeline_dashboard, plot_interactive_3d
            plot_pipeline_dashboard(result, output_dir=args.outdir, prefix=f"{method}_{seq[:10]}")
            
            # Export interactive AlphaFold-style 3D plot
            if "confidence_scores" in result:
                plot_interactive_3d(
                    coords=result["predicted_coords"],
                    sequence=result["sequence"],
                    confidence_scores=result["confidence_scores"],
                    title=f"Interactive Structure ({method})",
                    filename=os.path.join(args.outdir, f"{method}_{seq[:10]}_interactive.html")
                )
            
            print(f"  Exported Visualisations: {args.outdir}/{method}_{seq[:10]}_*.png (and .html)")
        except Exception as e:
            print(f"  Warning: Visualisation failed: {e}")

        # Print metrics if native available
        if native_coords is not None and "rmsd" in result:
            print(f"\n  Quality Metrics:")
            print(f"    RMSD:     {result.get('rmsd', 'N/A'):.3f} A")
            print(f"    TM-score: {result.get('tm_score', 'N/A'):.3f}")

    # Comparison table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  METHOD COMPARISON")
        print(f"{'='*70}")
        print(f"{'Method':<10} {'Energy':>10} {'Rg':>8} {'RMSD':>8} {'TM':>8} {'Time':>8}")
        print("-" * 70)
        for method, r in all_results.items():
            rmsd = r.get("rmsd", float("nan"))
            tm = r.get("tm_score", float("nan"))
            print(f"{method:<10} {r['total_energy']:>10.3f} {r['rg']:>8.2f} "
                  f"{rmsd:>8.3f} {tm:>8.3f} {r['time_total']:>8.2f}s")

    # Save results
    results_file = f"{args.outdir}/results_real_{seq[:10]}.json"
    serialisable = {}
    for method, r in all_results.items():
        serialisable[method] = {
            k: v for k, v in r.items()
            if not isinstance(v, np.ndarray) and k != "predicted_coords"
            and k != "raw_coords" and k != "aligned_coords"
        }
    with open(results_file, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    print(f"\nResults saved: {results_file}")


def _run_lattice(args):
    """Run the lattice HP model pipeline (original functionality)."""
    from quantum_fold.core.protein import Protein
    from quantum_fold.core.lattice import CubicLattice
    from quantum_fold.algorithms.baselines import (
        ExactSolver, GreedyLocalSearch, SimulatedAnnealing,
        GeneticAlgorithm, ReplicaExchangeMC,
    )
    from quantum_fold.utils.plotting import plot_fold, plot_convergence
    from quantum_fold.utils.visualization import export_pdb, fold_summary
    from quantum_fold.utils.statistics import summary_table
    from quantum_fold.utils.benchmarks import get_benchmark, print_benchmark_table

    if args.list_benchmarks:
        print_benchmark_table()
        return

    # Determine sequence
    if args.benchmark:
        bench = get_benchmark(args.benchmark)
        seq_str = bench.sequence
    elif args.seq:
        seq_str = args.seq.upper()
    else:
        seq_str = "HHPPHH"

    protein = Protein(seq_str, energy_model=args.model)
    N = protein.n

    print(f"\nLattice mode: {seq_str} (N={N}, model={args.model})")

    algos = [a.strip().lower() for a in args.algo.split(",")]
    if "all" in algos:
        algos = ["exact", "greedy", "sa", "ga", "remc"]

    results = {}
    exact_energy = None

    if "exact" in algos:
        t0 = time.time()
        solver = ExactSolver(protein)
        e, coords = solver.solve()
        dt = time.time() - t0
        exact_energy = e
        print(f"  Exact: E={e}, time={dt:.3f}s")
        export_pdb(coords, seq_str, f"{args.outdir}/exact_{seq_str}.pdb")
        results["Exact"] = {"energy": e, "time": dt, "success_rate": 1.0}

    if "greedy" in algos:
        t0 = time.time()
        gls = GreedyLocalSearch(protein, n_restarts=200, max_steps=500, seed=args.seed)
        e, coords, _ = gls.solve()
        dt = time.time() - t0
        print(f"  Greedy: E={e}, time={dt:.3f}s")
        results["Greedy"] = {"energy": e, "time": dt}

    if "sa" in algos:
        t0 = time.time()
        sa = SimulatedAnnealing(protein, t_start=5.0, t_end=0.01,
                                n_steps=5000, n_restarts=10, seed=args.seed)
        e, coords, _ = sa.solve()
        dt = time.time() - t0
        print(f"  SA: E={e}, time={dt:.3f}s")
        results["SA"] = {"energy": e, "time": dt}

    if "ga" in algos:
        t0 = time.time()
        ga = GeneticAlgorithm(protein, pop_size=100, n_generations=200,
                              mutation_rate=0.1, seed=args.seed)
        e, coords, _ = ga.solve()
        dt = time.time() - t0
        print(f"  GA: E={e}, time={dt:.3f}s")
        results["GA"] = {"energy": e, "time": dt}

    if "remc" in algos:
        t0 = time.time()
        remc = ReplicaExchangeMC(protein, n_replicas=8, t_min=0.1, t_max=10.0,
                                 n_steps=500, n_exchanges=100, seed=args.seed)
        e, coords, info = remc.solve()
        dt = time.time() - t0
        print(f"  REMC: E={e}, time={dt:.3f}s")
        results["REMC"] = {"energy": e, "time": dt}

    if results:
        ref = exact_energy or min(r["energy"] for r in results.values())
        print("\n" + summary_table(results, ref))


if __name__ == "__main__":
    main()
