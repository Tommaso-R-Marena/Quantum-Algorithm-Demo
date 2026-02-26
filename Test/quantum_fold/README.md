# Quantum Protein Folding Framework

A research-grade framework for **real protein structure prediction** using quantum-classical hybrid algorithms. Implements **Quantum Fragment Assembly (QFA)** — a novel approach where protein backbone conformations are decomposed into overlapping fragments, formulated as a QUBO, and solved with variational quantum algorithms (VQE/QAOA).

## Key Innovation

**Quantum Fragment Assembly**: Proteins are decomposed into overlapping fragments. Each fragment has discrete conformations drawn from Ramachandran-binned dihedral angles. The combinatorial assembly problem — selecting optimal fragment conformations — is encoded as a QUBO and solved with VQE/QAOA. A denoising diffusion model provides structural priors over Calpha distance matrices.

## Features

| Category | Features |
|----------|----------|
| **Real Proteins** | PDB download/parsing, 20 amino acids, 8 benchmark proteins |
| **Energy** | 6-term coarse-grained force field (contact, Ramachandran, VdW, Rg, H-bond, solvation) |
| **Backbone** | NeRF coordinate generation, Kabsch superposition, dihedral extraction |
| **Fragments** | SS-guided Ramachandran sampling, overlap scoring, QUBO formulation |
| **Quantum** | CVaR-VQE, QAOA, fragment QUBO encoding (ceil(log2(M)) qubits per fragment) |
| **Classical** | Exact enumeration, greedy, simulated annealing, genetic algorithm, REMC |
| **Diffusion** | DDPM distance matrix sampler, MDS coordinate reconstruction, SS priors |
| **Metrics** | RMSD, TM-score, GDT-TS, lDDT, contact map overlap (P/R/F1) |
| **Lattice** | HP/HP+/MJ models, 2-bit turn encoding, Walsh-Hadamard Hamiltonian |
| **Output** | PDB/XYZ export, publication-quality plots, JSON results, LaTeX tables |

## Quick Start

```bash
pip install pennylane matplotlib numpy

# Real protein structure prediction (SA baseline)
python -m quantum_fold.main --mode real --seq YYDPETGTWY --method sa

# Real protein benchmark with VQE
python -m quantum_fold.main --mode real --benchmark chignolin --method vqe --shots 300

# Compare methods
python -m quantum_fold.main --mode real --seq YYDPETGTWY --method all

# With diffusion prior
python -m quantum_fold.main --mode real --seq YYDPETGTWY --method sa --use-diffusion

# From a PDB structure (downloads native for comparison)
python -m quantum_fold.main --mode real --pdb 5AWL --method sa

# List available benchmarks
python -m quantum_fold.main --list-benchmarks

# Lattice HP model (original)
python -m quantum_fold.main --mode lattice --seq HHPPHH --algo all
```

## Architecture

```
quantum_fold/
├── core/
│   ├── residue.py           # 20 amino acids, Chou-Fasman SS prediction
│   ├── backbone.py          # NeRF coordinates, Kabsch RMSD, dihedrals
│   ├── force_field.py       # 6-term coarse-grained scoring
│   ├── fragment_library.py  # Fragment generation and QUBO matrices
│   ├── lattice.py           # 2-bit turn encoding (HP lattice)
│   ├── protein.py           # HP/HP+/MJ energy models
│   ├── encoding.py          # Bitstring <-> coordinates
│   └── hamiltonian.py       # QUBO/Ising Hamiltonian
├── algorithms/
│   ├── fragment_qopt.py     # Quantum Fragment Assembly (VQE/QAOA)
│   ├── diffusion.py         # DDPM backbone sampler
│   ├── hybrid_pipeline.py   # Full prediction pipeline
│   ├── baselines.py         # Exact, Greedy, SA, GA, REMC
│   ├── vqe_runner.py        # CVaR-VQE (lattice)
│   └── qaoa_runner.py       # QAOA (lattice)
├── utils/
│   ├── pdb_io.py            # PDB download and parsing
│   ├── metrics.py           # RMSD, TM-score, GDT-TS, lDDT
│   ├── real_benchmarks.py   # 8 real protein benchmarks
│   ├── benchmarks.py        # 17 HP lattice benchmarks
│   ├── statistics.py        # Bootstrap CI, Cohen's d, Wilcoxon
│   ├── plotting.py          # Publication-quality plots
│   └── visualization.py     # PDB/XYZ export
├── main.py                  # Experiment orchestrator
└── README.md
```

## Pipeline

```
Sequence → SS Prediction → Fragment Library → QUBO Formulation
                                                     ↓
                              Diffusion Prior → Distance Restraints
                                                     ↓
                                            VQE / QAOA / SA
                                                     ↓
                                         Fragment Assembly
                                                     ↓
                                         Local Refinement
                                                     ↓
                                    Score + RMSD/TM-score Report
```

## Benchmark Proteins

| Name | N | PDB | Fold | Difficulty |
|------|---|-----|------|-----------|
| Chignolin | 10 | 5AWL | Beta-hairpin | Easy |
| Trp-zip2 | 12 | 1LE1 | Beta-hairpin | Easy |
| GB1 hairpin | 12 | 2OED | Beta-hairpin | Easy |
| Alpha helix | 15 | — | Alpha-helix | Easy |
| Trp-cage | 20 | 1L2Y | Alpha+PP | Medium |
| BBA5 | 23 | 1T8J | BBA | Medium |
| WW domain | 34 | 1PIN | Beta-sheet | Hard |
| HP35 | 35 | 2F4K | 3-helix | Hard |

## References

1. Perdomo-Ortiz et al., *Sci. Rep.* **2**, 571 (2012)
2. Robert et al., *npj Quantum Inf.* **7**, 38 (2021)
3. Barkoutsos et al., *Quantum* **4**, 256 (2020) — CVaR-VQE
4. Zhang & Skolnick, *Proteins* **57**, 702 (2004) — TM-score
5. Zhou & Zhou, *Protein Sci.* **11**, 2714 (2002) — DFIRE
6. Parsons et al., *J. Comp. Chem.* **26**, 1063 (2005) — NeRF
7. Ho et al., NeurIPS 2020 — DDPM

## Tests

```bash
python -m pytest tests/ -v     # 71 tests
```
