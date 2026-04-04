import os
import sys
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("../")

from src.datasets.dataprocess import get_Data, read_fasta_as_dict
from src.utils import get_aligned_RSA
from src.models.Model import SAPP_Model
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser

warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")


def compute_window_coverage(rsa_array, site, window):
    """
    rsa_array: np.ndarray with np.nan for uncovered positions
    site: 0-based residue index
    window: half window size (25 => 51mer)
    """
    start = max(0, site - window)
    end = min(len(rsa_array), site + window + 1)

    local_rsa = rsa_array[start:end]
    observed = np.isfinite(local_rsa).sum()
    total = len(local_rsa)

    coverage = observed / total if total > 0 else np.nan
    return coverage


def summarize_alignment_quality(rsa_array, seq_len, site, window, identity=np.nan):
    protein_coverage = np.isfinite(rsa_array).sum() / seq_len if seq_len > 0 else np.nan
    window_coverage = compute_window_coverage(rsa_array, site, window)
    site_covered = bool(np.isfinite(rsa_array[site])) if 0 <= site < len(rsa_array) else False

    return {
        "ProteinCoverage": protein_coverage,
        "WindowCoverage51mer": window_coverage,
        "Identity": identity,
        "SiteCovered": site_covered,
    }


def resolve_model_paths(entries):
    from pathlib import Path
    resolved = []
    print(entries)
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            resolved += sorted([str(p) for p in path.glob("*.pt")])
        elif path.is_file() and str(path).endswith(".pt"):
            resolved.append(str(path))
        else:
            raise ValueError(f"Invalid model path entry: {entry}")
    return resolved


def get_ptm_config(ptm_type, config):
    default = config.get("default_config", {})
    ptm_specific = config.get("ptm_configs", {}).get(ptm_type, {})
    merged = {**default, **ptm_specific}
    return merged


def load_and_group_data(input_file, fasta_file, ptm_to_residue, config):
    grouped_info = defaultdict(list)
    rsa_dict = defaultdict(dict)
    seq_dict = defaultdict(dict)
    meta_info = defaultdict(list)

    protein_dic = read_fasta_as_dict(fasta_file)

    with open(input_file, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            #if len(tokens) < 5:
            #    raise ValueError("Each line must have at least 5 tab-separated fields")
            if len(tokens) == 5:
                protein_id, site, ptm_type, rsa_or_af_path, label = tokens
                label = int(label)
            elif len(tokens) == 4:
                protein_id, site, ptm_type, rsa_or_af_path = tokens
                label = None
            else:
                raise ValueError(
                    "Each line must have either 4 fields "
                    "(protein_id, site, ptm_type, rsa_path) "
                    "or 5 fields "
                    "(protein_id, site, ptm_type, rsa_path, label)"
                )

            site = int(site)

            seq = protein_dic[protein_id]
            assert seq[site] in ptm_to_residue[ptm_type], f"Residue mismatch at site {site} in {protein_id}"

            ptm_config = get_ptm_config(ptm_type, config)
            window = ptm_config.get("window_size", 25)

            identity = np.nan

            try:
                if rsa_or_af_path.endswith('.npy'):
                    rsa_values = np.load(rsa_or_af_path)

                    if len(rsa_values) != len(seq):
                        raise ValueError(
                            f"Loaded .npy RSA length mismatch for {protein_id}: "
                            f"{len(rsa_values)} vs seq length {len(seq)}"
                        )

                elif rsa_or_af_path.endswith('.cif'):
                    structure = MMCIFParser(QUIET=True).get_structure('model', rsa_or_af_path)
                    rsa_values, meta = get_aligned_RSA(
                        structure[0], rsa_or_af_path, seq, mask_value=np.nan
                    )
                    identity = meta.get("identity", np.nan)
                    print(
                        f"{protein_id}: coverage={meta['coverage']:.3f}, "
                        f"identity={meta['identity']:.3f}"
                    )

                elif rsa_or_af_path.endswith('.pdb'):
                    structure = PDBParser(QUIET=True).get_structure('model', rsa_or_af_path)
                    rsa_values, meta = get_aligned_RSA(
                        structure[0], rsa_or_af_path, seq, mask_value=np.nan
                    )
                    identity = meta.get("identity", np.nan)
                    print(
                        f"{protein_id}: coverage={meta['coverage']:.3f}, "
                        f"identity={meta['identity']:.3f}"
                    )

                else:
                    raise ValueError(f"Unsupported RSA source format: {rsa_or_af_path}")

            except Exception as e:
                print(f"Failed to get RSA for {protein_id} from {rsa_or_af_path}: {e}")
                continue

            rsa_values = np.asarray(rsa_values, dtype=float)

            quality_summary = summarize_alignment_quality(
                rsa_values, len(seq), site, window, identity=identity
            )

            # 모델 입력용: NaN -> 0
            rsa_values_for_model = np.nan_to_num(rsa_values, nan=0.0)

            grouped_info[ptm_type].append((protein_id, site, label if label is not None else -1, ptm_type))
            rsa_dict[ptm_type][protein_id] = rsa_values_for_model
            seq_dict[ptm_type][protein_id] = seq

            meta_info[ptm_type].append({
                "ProteinID": protein_id,
                "Site": site,
                "PTMType": ptm_type,
                "Label": label,
                **quality_summary
            })

    return grouped_info, rsa_dict, seq_dict, meta_info


def run_inference(config):
    ptm_to_residue = {
        'SAPPphos': ['S', 'T'], 'SAPP-methylR': ['R'], 'SAPP-phosY': ['Y'],
        'SAPP-sumoK': ['K'], 'SAPP-methylK': ['K'],
        'SAPP-acetylK': ['K'], 'SAPP-ubiquitinK': ['K'],
        'SAPP-CMGC': ['S', 'T'], 'SAPP-CAMK': ['S', 'T'], 'SAPP-CDK': ['S', 'T'],
        'SAPP-AGC': ['S', 'T'], 'SAPP-MAPK': ['S', 'T'], 'SAPP-PKA': ['S', 'T'],
        'SAPP-PKC': ['S', 'T'], 'SAPP-CK2': ['S', 'T']
    }

    input_path = config["input_path"]
    fasta_path = config["fasta_path"]
    output_path = config["output_path"]
    device = config.get("device", "cpu")

    grouped_info, rsa_dict, seq_dict, meta_info = load_and_group_data(
        input_path, fasta_path, ptm_to_residue, config
    )

    all_results = []
    all_model_results = []

    for ptm_type, info_list in grouped_info.items():
        print(f"Running inference for {ptm_type}...")
        ptm_config = get_ptm_config(ptm_type, config)

        window = ptm_config.get("window_size", 25)
        batch_size = ptm_config.get("batch_size", 128)

        test_tensors = get_Data(
            info_list, seq_dict[ptm_type], rsa_dict[ptm_type], window
        )

        loader = DataLoader(
            TensorDataset(*test_tensors),
            batch_size=batch_size,
            shuffle=False
        )

        model = SAPP_Model(
            vocab_size=ptm_config.get("embedding_dim", 22),
            window=window,
            hidden=ptm_config.get("hidden_size", 256),
            n_layers=ptm_config.get("n_layers", 2),
            attn_heads=ptm_config.get("attn_heads", 4),
            feed_forward_dim=ptm_config.get("feed_forward_dim", 758),
            device=device
        ).to(device)

        pred_list = []
        weight_files = resolve_model_paths(ptm_config.get("model_paths"))

        sample_meta_df = pd.DataFrame(meta_info[ptm_type])

        for weight_file in weight_files:
            model.load_state_dict(torch.load(weight_file, map_location=device))
            model.eval()

            preds = []
            for batch in tqdm(loader, desc=f"{ptm_type} - {weight_file}"):
                batch = [b.to(device) for b in batch]
                pred, _ = model(batch[0], batch[1], batch[2], batch[3])
                preds.append(pred.view(-1).detach().cpu().numpy())

            pred_values = np.concatenate(preds)
            pred_list.append(pred_values)

            model_name = os.path.splitext(os.path.basename(weight_file))[0]

            model_result_df = sample_meta_df.copy()
            model_result_df["Pred"] = pred_values
            model_result_df["Model"] = model_name

            all_model_results.append(model_result_df)

        averaged_preds = np.mean(np.stack(pred_list), axis=0)

        result_df = sample_meta_df.copy()
        result_df["Pred"] = averaged_preds

        all_results.append(result_df)

    ensemble_path = output_path + '_ensemble.csv'
    final_df = pd.concat(all_results, ignore_index=True)

    output_cols = [
        "ProteinID", "Site", "PTMType", "Pred",
        "ProteinCoverage", "WindowCoverage51mer", "Identity", "SiteCovered"
    ]

    if "Label" in final_df.columns and final_df["Label"].notna().any():
        output_cols.insert(3, "Label")

    final_df = final_df[output_cols]
    final_df.to_csv(ensemble_path, index=False)
    print(f"Averaged inference results saved to: {ensemble_path}")

    modelwise_output_path = output_path + '_by_model.csv'
    modelwise_df = pd.concat(all_model_results, ignore_index=True)

    model_output_cols = [
        "ProteinID", "Site", "PTMType", "Model", "Pred",
        "ProteinCoverage", "WindowCoverage51mer", "Identity", "SiteCovered"
    ]

    if "Label" in modelwise_df.columns and modelwise_df["Label"].notna().any():
        model_output_cols.insert(3, "Label")

    modelwise_df = modelwise_df[model_output_cols]
    modelwise_df.to_csv(modelwise_output_path, index=False)
    print(f"Model-wise inference results saved to: {modelwise_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on PTM data using SAPP")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)

    run_inference(config)