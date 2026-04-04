from Bio.PDB import DSSP
from Bio import pairwise2
import numpy as np


def get_RSA(model, path):
    dssp_ret = DSSP(model, path)
    AA = [i[1] for i in dssp_ret.property_list]
    RSA = [i[3] for i in dssp_ret.property_list]

    pdb_seq = ''.join(AA)

    return pdb_seq, RSA


def align_rsa_to_reference(pdb_seq, rsa_values, ref_seq, mask_value=np.nan):
    if len(pdb_seq) != len(rsa_values):
        raise ValueError(
            f"Length mismatch: len(pdb_seq)={len(pdb_seq)}, len(rsa_values)={len(rsa_values)}"
        )

    alignments = pairwise2.align.globalms(ref_seq, pdb_seq, 2, -1, -5, -0.5)
    if not alignments:
        raise ValueError("Failed to align pdb_seq to ref_seq")

    best = alignments[0]
    aligned_ref = best.seqA
    aligned_pdb = best.seqB

    projected_rsa = []
    pdb_idx = 0
    matched_positions = 0
    identical_positions = 0

    for aa_ref, aa_pdb in zip(aligned_ref, aligned_pdb):
        if aa_ref != '-' and aa_pdb != '-':
            projected_rsa.append(float(rsa_values[pdb_idx]))
            matched_positions += 1
            if aa_ref == aa_pdb:
                identical_positions += 1
            pdb_idx += 1

        elif aa_ref == '-' and aa_pdb != '-':
            pdb_idx += 1

        elif aa_ref != '-' and aa_pdb == '-':
            projected_rsa.append(mask_value)

    aligned_rsa = np.array(projected_rsa, dtype=float)

    if len(aligned_rsa) != len(ref_seq):
        raise ValueError(
            f"Projected RSA length mismatch: got {len(aligned_rsa)}, expected {len(ref_seq)}"
        )

    coverage = matched_positions / len(ref_seq) if len(ref_seq) > 0 else 0.0
    identity = identical_positions / matched_positions if matched_positions > 0 else 0.0

    meta = {
        "pdb_seq": pdb_seq,
        "aligned_ref": aligned_ref,
        "aligned_pdb": aligned_pdb,
        "coverage": coverage,
        "identity": identity,
        "matched_positions": matched_positions,
        "identical_positions": identical_positions,
    }

    return aligned_rsa, meta


def get_aligned_RSA(model, path, ref_seq, mask_value=np.nan):
    pdb_seq, rsa_values = get_RSA(model, path)
    aligned_rsa, meta = align_rsa_to_reference(
        pdb_seq, rsa_values, ref_seq, mask_value=mask_value
    )
    return aligned_rsa, meta