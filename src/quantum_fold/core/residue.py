"""
residue.py
Amino acid properties and backbone geometry for real protein modelling.

Contains:
  - All 20 standard amino acids with codes, masses, volumes
  - Kyte-Doolittle hydrophobicity scale
  - Van der Waals radii for Cα coarse-graining
  - Ideal backbone geometry (Engh-Huber parameters)
  - Secondary structure propensities (Chou-Fasman)

References:
  [1] Engh & Huber, Acta Cryst. A47, 392 (1991)
  [2] Kyte & Doolittle, J. Mol. Biol. 157, 105 (1982)
  [3] Chou & Fasman, Biochemistry 13, 222 (1974)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional

# ═══════════════════════════════════════════════════════════════════════════
# Amino acid data
# ═══════════════════════════════════════════════════════════════════════════

AA_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AA_3TO1 = {v: k for k, v in AA_1TO3.items()}
STANDARD_AAS = list(AA_1TO3.keys())

# Kyte-Doolittle hydrophobicity (higher = more hydrophobic)
HYDROPHOBICITY = {
    "I":  4.5, "V":  4.2, "L":  3.8, "F":  2.8, "C":  2.5,
    "M":  1.9, "A":  1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "D": -3.5,
    "E": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5,
}

# Van der Waals radii for Cα (Å) — effective radii for coarse-grained model
VDW_RADIUS = {aa: 3.8 for aa in STANDARD_AAS}  # ~3.8 Å uniform for Cα
VDW_RADIUS.update({"G": 3.4, "A": 3.6, "W": 4.2, "F": 4.0, "Y": 4.0})

# Molecular weight (Da)
MOLECULAR_WEIGHT = {
    "G": 57.05, "A": 71.08, "V": 99.13, "L": 113.16, "I": 113.16,
    "P": 97.12, "F": 147.18, "W": 186.21, "M": 131.20, "S": 87.08,
    "T": 101.10, "C": 103.14, "Y": 163.18, "H": 137.14, "D": 115.09,
    "E": 129.12, "N": 114.10, "Q": 128.13, "K": 128.17, "R": 156.19,
}

# Chou-Fasman secondary structure propensities (alpha, beta, turn)
SS_PROPENSITY = {
    "A": (1.42, 0.83, 0.66), "R": (0.98, 0.93, 0.95), "N": (0.67, 0.89, 1.56),
    "D": (1.01, 0.54, 1.46), "C": (0.70, 1.19, 1.19), "Q": (1.11, 1.10, 0.98),
    "E": (1.51, 0.37, 0.74), "G": (0.57, 0.75, 1.56), "H": (1.00, 0.87, 0.95),
    "I": (1.08, 1.60, 0.47), "L": (1.21, 1.30, 0.59), "K": (1.16, 0.74, 1.01),
    "M": (1.45, 1.05, 0.60), "F": (1.13, 1.38, 0.60), "P": (0.57, 0.55, 1.52),
    "S": (0.77, 0.75, 1.43), "T": (0.83, 1.19, 0.96), "W": (1.08, 1.37, 0.96),
    "Y": (0.69, 1.47, 1.14), "V": (1.06, 1.70, 0.50),
}

# ═══════════════════════════════════════════════════════════════════════════
# Backbone geometry (Engh-Huber ideal values)
# ═══════════════════════════════════════════════════════════════════════════

# Bond lengths (Å)
BOND_LENGTH_N_CA = 1.458   # N—Cα
BOND_LENGTH_CA_C = 1.525   # Cα—C
BOND_LENGTH_C_N = 1.329    # C—N (peptide bond)
BOND_LENGTH_CA_CA = 3.80   # Cα—Cα (virtual bond, approximate)

# Bond angles (radians)
BOND_ANGLE_N_CA_C = np.radians(111.2)   # N—Cα—C
BOND_ANGLE_CA_C_N = np.radians(116.2)   # Cα—C—N
BOND_ANGLE_C_N_CA = np.radians(121.7)   # C—N—Cα

# Peptide bond (ω angle) — typically 180° (trans) or 0° (cis for Pro)
OMEGA_TRANS = np.pi
OMEGA_CIS = 0.0

# Ramachandran angle ranges (degrees) for common secondary structures
RAMA_REGIONS = {
    "alpha_R":  (-63.0, -43.0),   # right-handed alpha helix
    "alpha_L":  (57.0, 47.0),     # left-handed alpha helix
    "beta":     (-119.0, 113.0),  # beta strand
    "ppII":     (-75.0, 145.0),   # polyproline II
    "turn_I":   (-60.0, -30.0),   # type I turn (i+1)
    "turn_II":  (-60.0, 120.0),   # type II turn (i+1)
}


def predict_secondary_structure(sequence: str) -> str:
    """
    Simple secondary structure prediction using Chou-Fasman propensities.
    Returns a string of same length: H=helix, E=strand, C=coil.
    """
    n = len(sequence)
    ss = list("C" * n)

    # Sliding window of 6 for helix nucleation
    for i in range(n - 5):
        window = sequence[i:i+6]
        alpha_scores = [SS_PROPENSITY.get(aa, (1.0, 1.0, 1.0))[0] for aa in window]
        if sum(1 for s in alpha_scores if s > 1.0) >= 4:
            for j in range(i, min(i + 6, n)):
                if ss[j] != "E":
                    ss[j] = "H"

    # Sliding window of 5 for strand nucleation
    for i in range(n - 4):
        window = sequence[i:i+5]
        beta_scores = [SS_PROPENSITY.get(aa, (1.0, 1.0, 1.0))[1] for aa in window]
        if sum(1 for s in beta_scores if s > 1.0) >= 3:
            for j in range(i, min(i + 5, n)):
                if ss[j] != "H":
                    ss[j] = "E"

    return "".join(ss)


def sequence_features(sequence: str) -> Dict[str, float]:
    """Compute sequence-level features for a protein."""
    n = len(sequence)
    hydro = [HYDROPHOBICITY.get(aa, 0.0) for aa in sequence]
    return {
        "length": n,
        "mean_hydrophobicity": float(np.mean(hydro)),
        "std_hydrophobicity": float(np.std(hydro)),
        "fraction_hydrophobic": sum(1 for h in hydro if h > 0) / max(n, 1),
        "molecular_weight": sum(MOLECULAR_WEIGHT.get(aa, 110.0) for aa in sequence),
    }
