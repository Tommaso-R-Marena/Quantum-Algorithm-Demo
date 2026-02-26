"""
metrics.py
Structure quality metrics for protein structure prediction.

Implements:
  - RMSD (with Kabsch superposition)
  - TM-score (Zhang & Skolnick, 2004)
  - GDT-TS (Global Distance Test - Total Score)
  - lDDT (local Distance Difference Test)
  - Contact map overlap (precision, recall, F1)

References:
  [1] Zhang & Skolnick, Proteins 57, 702 (2004) — TM-score
  [2] Zemla, Nucleic Acids Res. 31, 3370 (2003) — GDT-TS
  [3] Mariani et al., Bioinformatics 29, 2722 (2013) — lDDT
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from ..core.backbone import kabsch_rmsd, ca_distance_matrix, contact_map


def rmsd(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
) -> float:
    """
    Compute RMSD after optimal superposition (Kabsch).

    Parameters
    ----------
    pred_coords : (N, 3)
    ref_coords : (N, 3)

    Returns
    -------
    rmsd : float (Angstroms)
    """
    val, _ = kabsch_rmsd(pred_coords, ref_coords)
    return val


def tm_score(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
) -> float:
    """
    Compute TM-score (Zhang & Skolnick, 2004).

    TM-score is a length-independent metric in [0, 1]:
      TM = max_alignment { (1/L_ref) * sum_i 1/(1 + (d_i/d0)^2) }

    where d_i is the distance between aligned residues i after
    optimal superposition, and d0 is a length-dependent scale:
      d0 = 1.24 * (L_ref - 15)^(1/3) - 1.8

    Returns
    -------
    tm : float in [0, 1]
      1.0 = identical structures
      > 0.5 generally indicates same fold
      > 0.17 roughly random
    """
    assert pred_coords.shape == ref_coords.shape
    L = len(ref_coords)

    if L < 5:
        return 1.0 if L > 0 else 0.0

    # Length-dependent distance scale
    d0 = 1.24 * max(L - 15, 1) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)  # minimum d0

    # Align structures
    _, aligned = kabsch_rmsd(pred_coords, ref_coords)

    # Compute per-residue distances after alignment
    d_i = np.sqrt(np.sum((aligned - ref_coords) ** 2, axis=1))

    # TM-score
    tm = np.sum(1.0 / (1.0 + (d_i / d0) ** 2)) / L

    return float(tm)


def gdt_ts(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
) -> float:
    """
    Compute GDT-TS (Global Distance Test - Total Score).

    GDT-TS = (GDT_1 + GDT_2 + GDT_4 + GDT_8) / 4

    where GDT_t = fraction of aligned residues within t Angstroms.

    Returns
    -------
    gdt : float in [0, 1]
    """
    assert pred_coords.shape == ref_coords.shape
    L = len(ref_coords)

    if L == 0:
        return 0.0

    # Align
    _, aligned = kabsch_rmsd(pred_coords, ref_coords)
    d_i = np.sqrt(np.sum((aligned - ref_coords) ** 2, axis=1))

    thresholds = [1.0, 2.0, 4.0, 8.0]
    gdt_scores = []
    for t in thresholds:
        frac = np.sum(d_i <= t) / L
        gdt_scores.append(frac)

    return float(np.mean(gdt_scores))


def lddt(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    cutoff: float = 15.0,
    thresholds: Optional[list] = None,
) -> float:
    """
    Compute lDDT (local Distance Difference Test).

    For each pair of residues within cutoff in the reference:
      preserved(i,j) = 1 if |d_pred(i,j) - d_ref(i,j)| < threshold

    lDDT = average over thresholds of the fraction of preserved distances.

    Returns
    -------
    lddt : float in [0, 1]
    """
    if thresholds is None:
        thresholds = [0.5, 1.0, 2.0, 4.0]

    D_pred = ca_distance_matrix(pred_coords)
    D_ref = ca_distance_matrix(ref_coords)
    n = len(ref_coords)

    # Valid mask for reference distances (i < j)
    tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_mask = (D_ref < cutoff) & tri_mask
    total = np.sum(valid_mask)
    
    if total == 0:
        return 1.0

    diffs = np.abs(D_pred - D_ref)
    
    scores = []
    for t in thresholds:
        preserved = np.sum((diffs < t) & valid_mask)
        scores.append(preserved / total)

    return float(np.mean(scores))


def contact_overlap(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    threshold: float = 8.0,
    min_seq_sep: int = 3,
) -> Tuple[float, float, float]:
    """
    Compute contact map overlap metrics.

    Returns (precision, recall, F1).
    """
    pred_cmap = contact_map(pred_coords, threshold, min_seq_sep)
    ref_cmap = contact_map(ref_coords, threshold, min_seq_sep)

    # Use upper triangle only to avoid double counting
    tri_mask = np.triu(np.ones_like(pred_cmap, dtype=bool), k=1)
    
    pred_contacts = pred_cmap & tri_mask
    ref_contacts = ref_cmap & tri_mask

    tp = np.sum(pred_contacts & ref_contacts)
    n_pred = np.sum(pred_contacts)
    n_ref = np.sum(ref_contacts)

    precision = tp / max(n_pred, 1)
    recall = tp / max(n_ref, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return float(precision), float(recall), float(f1)


def per_residue_distance(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
) -> np.ndarray:
    """
    Per-residue Calpha distance after alignment.
    Returns array of shape (N,).
    """
    _, aligned = kabsch_rmsd(pred_coords, ref_coords)
    return np.sqrt(np.sum((aligned - ref_coords) ** 2, axis=1))


def print_metrics(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    name: str = "",
) -> dict:
    """Compute and print all metrics."""
    r = rmsd(pred_coords, ref_coords)
    tm = tm_score(pred_coords, ref_coords)
    gdt = gdt_ts(pred_coords, ref_coords)
    ldt = lddt(pred_coords, ref_coords)
    prec, rec, f1 = contact_overlap(pred_coords, ref_coords)

    print(f"  Metrics for {name or 'prediction'}:")
    print(f"    RMSD:    {r:.3f} A")
    print(f"    TM-score: {tm:.3f}")
    print(f"    GDT-TS:  {gdt:.3f}")
    print(f"    lDDT:    {ldt:.3f}")
    print(f"    Contact P/R/F1: {prec:.3f}/{rec:.3f}/{f1:.3f}")

    return {
        "rmsd": r, "tm_score": tm, "gdt_ts": gdt,
        "lddt": ldt, "contact_precision": prec,
        "contact_recall": rec, "contact_f1": f1,
    }
