"""
backbone.py
Backbone coordinate generation and manipulation for real proteins.

Implements:
  - NeRF (Natural Extension Reference Frame) for building coordinates
    from dihedral angles (phi, psi, omega)
  - Kabsch superposition for RMSD computation
  - Dihedral angle extraction from coordinates
  - Ramachandran angle discretisation

The NeRF algorithm places each atom by specifying:
  bond length d, bond angle theta, dihedral angle phi
relative to the three preceding atoms in the chain.

References:
  [1] Parsons et al., J. Comp. Chem. 26, 1063 (2005)  -- NeRF
  [2] Kabsch, Acta Cryst. A32, 922 (1976)  -- Superposition
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Sequence
from .residue import (
    BOND_LENGTH_N_CA, BOND_LENGTH_CA_C, BOND_LENGTH_C_N,
    BOND_ANGLE_N_CA_C, BOND_ANGLE_CA_C_N, BOND_ANGLE_C_N_CA,
    OMEGA_TRANS, RAMA_REGIONS,
)


# ═══════════════════════════════════════════════════════════════════════════
# NeRF coordinate generation
# ═══════════════════════════════════════════════════════════════════════════

def nerf_place_atom(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_length: float,
    bond_angle: float,
    dihedral: float,
) -> np.ndarray:
    """
    Place atom D given atoms A, B, C using NeRF.

    D is placed at distance `bond_length` from C,
    with angle B-C-D = `bond_angle`,
    and dihedral A-B-C-D = `dihedral`.

    Parameters
    ----------
    a, b, c : np.ndarray, shape (3,)
        Coordinates of three preceding atoms.
    bond_length : float (Angstroms)
    bond_angle : float (radians)
    dihedral : float (radians)

    Returns
    -------
    d : np.ndarray, shape (3,)
    """
    bc = c - b
    bc_norm = bc / (np.linalg.norm(bc) + 1e-12)

    ab = b - a
    n = np.cross(ab, bc_norm)
    n_norm = n / (np.linalg.norm(n) + 1e-12)

    m = np.cross(n_norm, bc_norm)

    # Position in local frame
    dx = -bond_length * np.cos(bond_angle)
    dy = bond_length * np.sin(bond_angle) * np.cos(dihedral)
    dz = bond_length * np.sin(bond_angle) * np.sin(dihedral)

    d = c + dx * bc_norm + dy * m + dz * n_norm
    return d


def build_backbone(
    phi: Sequence[float],
    psi: Sequence[float],
    omega: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Build full backbone (N, CA, C) coordinates from dihedral angles.

    Parameters
    ----------
    phi : array-like, length N
        Phi angles (radians). phi[0] is typically undefined (set to -60°).
    psi : array-like, length N
        Psi angles (radians). psi[-1] is typically undefined (set to -40°).
    omega : array-like, length N, optional
        Omega angles (radians). Default: all trans (pi).

    Returns
    -------
    coords : np.ndarray, shape (3*N, 3)
        Backbone atoms in order: N0, CA0, C0, N1, CA1, C1, ...
    """
    n_res = len(phi)
    if omega is None:
        omega = [OMEGA_TRANS] * n_res

    # Bond lengths and angles for backbone atoms
    # Pattern: ...-C(i-1)-N(i)-CA(i)-C(i)-N(i+1)-...
    # The dihedrals are:
    #   omega(i): CA(i-1)-C(i-1)-N(i)-CA(i)
    #   phi(i):   C(i-1)-N(i)-CA(i)-C(i)
    #   psi(i):   N(i)-CA(i)-C(i)-N(i+1)

    # Initial atoms (arbitrary placement)
    coords = np.zeros((3 * n_res, 3), dtype=np.float64)
    coords[0] = np.array([0.0, 0.0, 0.0])                    # N0
    coords[1] = np.array([BOND_LENGTH_N_CA, 0.0, 0.0])        # CA0
    coords[2] = nerf_place_atom(                                # C0
        np.array([-BOND_LENGTH_C_N, 0.0, 0.0]),
        coords[0], coords[1],
        BOND_LENGTH_CA_C, BOND_ANGLE_N_CA_C, psi[0],
    )

    for i in range(1, n_res):
        base = 3 * i
        prev_n = coords[base - 3]   # N(i-1)
        prev_ca = coords[base - 2]  # CA(i-1)
        prev_c = coords[base - 1]   # C(i-1)

        # N(i): placed from CA(i-1)-C(i-1) with omega dihedral
        coords[base] = nerf_place_atom(
            prev_n, prev_ca, prev_c,
            BOND_LENGTH_C_N, BOND_ANGLE_CA_C_N, omega[i],
        )

        # CA(i): placed from C(i-1)-N(i) with phi dihedral
        coords[base + 1] = nerf_place_atom(
            prev_ca, prev_c, coords[base],
            BOND_LENGTH_N_CA, BOND_ANGLE_C_N_CA, phi[i],
        )

        # C(i): placed from N(i)-CA(i) with psi dihedral
        coords[base + 2] = nerf_place_atom(
            prev_c, coords[base], coords[base + 1],
            BOND_LENGTH_CA_C, BOND_ANGLE_N_CA_C, psi[i],
        )

    return coords


def extract_ca_coords(backbone: np.ndarray) -> np.ndarray:
    """Extract Cα coordinates from full backbone (every 3rd atom starting at index 1)."""
    return backbone[1::3].copy()


def build_ca_trace(
    phi: Sequence[float],
    psi: Sequence[float],
    omega: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Build only the Cα trace from dihedral angles."""
    backbone = build_backbone(phi, psi, omega)
    return extract_ca_coords(backbone)


# ═══════════════════════════════════════════════════════════════════════════
# Dihedral angle extraction
# ═══════════════════════════════════════════════════════════════════════════

def compute_dihedral(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> float:
    """
    Compute the dihedral angle defined by four points.
    Returns angle in radians, range [-pi, pi].
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    m = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))

    x = np.dot(n1, n2)
    y = np.dot(m, n2)

    return float(-np.arctan2(y, x))


def extract_dihedrals(backbone: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract phi, psi, omega angles from backbone coordinates.

    Parameters
    ----------
    backbone : np.ndarray, shape (3*N, 3)

    Returns
    -------
    phi, psi, omega : np.ndarray, each shape (N,)
        First phi and last psi are set to 0 (undefined).
    """
    n_res = len(backbone) // 3
    phi = np.zeros(n_res)
    psi = np.zeros(n_res)
    omega = np.zeros(n_res)

    for i in range(n_res):
        n_i = backbone[3 * i]
        ca_i = backbone[3 * i + 1]
        c_i = backbone[3 * i + 2]

        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0:
            c_prev = backbone[3 * (i - 1) + 2]
            phi[i] = compute_dihedral(c_prev, n_i, ca_i, c_i)

        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        if i < n_res - 1:
            n_next = backbone[3 * (i + 1)]
            psi[i] = compute_dihedral(n_i, ca_i, c_i, n_next)

        # Omega: CA(i-1) - C(i-1) - N(i) - CA(i)
        if i > 0:
            ca_prev = backbone[3 * (i - 1) + 1]
            c_prev = backbone[3 * (i - 1) + 2]
            omega[i] = compute_dihedral(ca_prev, c_prev, n_i, ca_i)

    return phi, psi, omega


# ═══════════════════════════════════════════════════════════════════════════
# Kabsch superposition and RMSD
# ═══════════════════════════════════════════════════════════════════════════

def kabsch_rmsd(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute the RMSD after optimal superposition (Kabsch algorithm).

    Parameters
    ----------
    coords_pred : np.ndarray, shape (N, 3)
    coords_ref  : np.ndarray, shape (N, 3)

    Returns
    -------
    rmsd : float
    aligned_pred : np.ndarray, shape (N, 3)
        Predicted coordinates after alignment.
    """
    assert coords_pred.shape == coords_ref.shape

    # Centre both structures
    centroid_pred = np.mean(coords_pred, axis=0)
    centroid_ref = np.mean(coords_ref, axis=0)
    p = coords_pred - centroid_pred
    q = coords_ref - centroid_ref

    # Cross-covariance matrix
    H = p.T @ q

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and translation
    aligned = (R @ p.T).T + centroid_ref

    # RMSD
    diff = aligned - coords_ref
    rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    return rmsd, aligned


# ═══════════════════════════════════════════════════════════════════════════
# Ramachandran discretisation
# ═══════════════════════════════════════════════════════════════════════════

def ramachandran_bins(n_bins: int = 8) -> List[Tuple[float, float]]:
    """
    Generate evenly-spaced Ramachandran bins.
    Returns list of (phi_center, psi_center) in radians.
    """
    step = 2 * np.pi / n_bins
    bins = []
    for i in range(n_bins):
        phi_center = -np.pi + (i + 0.5) * step
        for j in range(n_bins):
            psi_center = -np.pi + (j + 0.5) * step
            bins.append((phi_center, psi_center))
    return bins


def discretise_angles(
    phi: float, psi: float, n_bins: int = 8
) -> int:
    """Map continuous (phi, psi) to the nearest Ramachandran bin index."""
    bins = ramachandran_bins(n_bins)
    best_idx = 0
    best_dist = float("inf")
    for idx, (pc, qc) in enumerate(bins):
        # Angular distance (periodic)
        dp = min(abs(phi - pc), 2 * np.pi - abs(phi - pc))
        dq = min(abs(psi - qc), 2 * np.pi - abs(psi - qc))
        dist = dp ** 2 + dq ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def bin_to_angles(
    bin_idx: int, n_bins: int = 8
) -> Tuple[float, float]:
    """Convert a Ramachandran bin index to (phi, psi) center angles."""
    bins = ramachandran_bins(n_bins)
    return bins[bin_idx % len(bins)]


# ═══════════════════════════════════════════════════════════════════════════
# Distance matrix utilities
# ═══════════════════════════════════════════════════════════════════════════

def ca_distance_matrix(ca_coords: np.ndarray) -> np.ndarray:
    """Compute the pairwise Cα distance matrix."""
    n = len(ca_coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(ca_coords[i] - ca_coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def contact_map(
    ca_coords: np.ndarray,
    threshold: float = 8.0,
    min_seq_sep: int = 3,
) -> np.ndarray:
    """
    Binary contact map: C[i,j] = 1 if Cα distance < threshold
    and |i-j| >= min_seq_sep.
    """
    D = ca_distance_matrix(ca_coords)
    n = len(ca_coords)
    C = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            if D[i, j] < threshold:
                C[i, j] = 1
                C[j, i] = 1
    return C
