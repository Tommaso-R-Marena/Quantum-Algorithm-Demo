"""
pdb_io.py
PDB file I/O for real protein structures.

Provides:
  - Download PDB files from RCSB by PDB ID
  - Parse ATOM records to extract Calpha coordinates and sequence
  - Write predicted structures as PDB
  - Handle multi-model and multi-chain files
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..core.residue import AA_3TO1, AA_1TO3


def fetch_pdb(
    pdb_id: str,
    output_dir: str = ".",
    force: bool = False,
) -> str:
    """
    Download a PDB file from RCSB.

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier (e.g. "1L2Y").
    output_dir : str
        Directory to save the file.
    force : bool
        Re-download even if file exists.

    Returns
    -------
    filepath : str
    """
    import urllib.request

    pdb_id = pdb_id.upper().strip()
    filepath = os.path.join(output_dir, f"{pdb_id}.pdb")

    if os.path.exists(filepath) and not force:
        print(f"  PDB file exists: {filepath}")
        return filepath

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {url}...")

    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  Saved: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to download PDB {pdb_id}: {e}")

    return filepath


def parse_pdb(
    filepath: str,
    chain: str = "A",
    model: int = 1,
) -> Dict:
    """
    Parse a PDB file and extract Calpha coordinates and sequence.

    Parameters
    ----------
    filepath : str
    chain : str
        Chain ID to extract (default "A").
    model : int
        Model number (for NMR structures with multiple models).

    Returns
    -------
    dict with:
      "ca_coords" : np.ndarray (N, 3)
      "sequence" : str
      "residue_numbers" : list of int
      "chain" : str
      "n_residues" : int
      "resolution" : float or None
      "title" : str
    """
    ca_coords = []
    sequence = []
    residue_numbers = []
    title = ""
    resolution = None
    current_model = 0
    seen_residues = set()

    with open(filepath, "r") as f:
        for line in f:
            # Title
            if line.startswith("TITLE"):
                title += line[10:].strip() + " "

            # Resolution
            if line.startswith("REMARK   2 RESOLUTION"):
                try:
                    res_str = line[26:].strip().split()[0]
                    resolution = float(res_str)
                except (ValueError, IndexError):
                    pass

            # Model tracking (for NMR)
            if line.startswith("MODEL"):
                try:
                    current_model = int(line[6:].strip())
                except ValueError:
                    current_model += 1

            if line.startswith("ENDMDL"):
                if current_model >= model:
                    break

            # Only process requested model
            if current_model > 0 and current_model != model:
                continue

            # ATOM records
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue

                chain_id = line[21].strip()
                if chain_id != chain:
                    continue

                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())

                # Skip alternate conformations
                alt_loc = line[16].strip()
                if alt_loc and alt_loc != "A":
                    continue

                # Skip duplicate residues
                if res_num in seen_residues:
                    continue
                seen_residues.add(res_num)

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                ca_coords.append([x, y, z])
                aa_1 = AA_3TO1.get(res_name, "X")
                sequence.append(aa_1)
                residue_numbers.append(res_num)

    ca_coords = np.array(ca_coords, dtype=np.float64)

    return {
        "ca_coords": ca_coords,
        "sequence": "".join(sequence),
        "residue_numbers": residue_numbers,
        "chain": chain,
        "n_residues": len(ca_coords),
        "resolution": resolution,
        "title": title.strip(),
    }


def write_ca_pdb(
    ca_coords: np.ndarray,
    sequence: str,
    filename: str = "predicted.pdb",
    chain: str = "A",
    remarks: Optional[List[str]] = None,
) -> str:
    """
    Write Calpha coordinates as a PDB file.

    Parameters
    ----------
    ca_coords : np.ndarray (N, 3)
    sequence : str
    filename : str
    chain : str
    remarks : list of str, optional
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, "w") as f:
        f.write(f"REMARK   Generated by quantum_fold\n")
        if remarks:
            for r in remarks:
                f.write(f"REMARK   {r}\n")
        f.write(f"REMARK   Sequence: {sequence[:60]}\n")

        for i in range(len(ca_coords)):
            x, y, z = ca_coords[i]
            aa = sequence[i] if i < len(sequence) else "X"
            res_name = AA_1TO3.get(aa, "UNK")

            f.write(
                f"ATOM  {i+1:5d}  CA  {res_name:3s} {chain}{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00           C  \n"
            )

        # CONECT
        for i in range(len(ca_coords) - 1):
            f.write(f"CONECT{i+1:5d}{i+2:5d}\n")

        f.write("TER\n")
        f.write("END\n")

    return filename
