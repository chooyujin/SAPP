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
from src.utils import get_RSA
from src.models.Model import SAPP_Model
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser

warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")

def load_and_group_data(input_file,fasta_file, ptm_to_residue):
    grouped_info = defaultdict(list)
    rsa_dict = defaultdict(dict)
    seq_dict = defaultdict(dict)
    protein_dic = read_fasta_as_dict(fasta_file)

    with open(input_file, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) < 5:
                raise ValueError("Each line must have at least 5 tab-separated fields")

            protein_id, site, label, ptm_type, rsa_or_af_path = tokens[:5]
            site = int(site)
            label = int(label)
            seq = protein_dic[protein_id]
            assert seq[site] in ptm_to_residue[ptm_type], f"Residue mismatch at site {site} in {protein_id}"
            try:
                if rsa_or_af_path.endswith('.npy'):
                    rsa_values = np.load(rsa_or_af_path)
                elif rsa_or_af_path.endswith('.cif'):
                    structure = MMCIFParser(QUIET=True).get_structure('model', rsa_or_af_path)
                    pdb_seq, rsa_values = get_RSA(structure[0], rsa_or_af_path)
                    assert pdb_seq == seq, f"The protein {protein_id} sequence and structure must be identical"
                elif rsa_or_af_path.endswith('.pdb'):
                    structure = PDBParser(QUIET=True).get_structure('model', rsa_or_af_path)
                    pdb_seq, rsa_values = get_RSA(structure[0], rsa_or_af_path)
                    assert pdb_seq == seq, f"The protein {protein_id} sequence and structure must be identical"
                else:
                    raise ValueError(f"Unsupported RSA source format: {rsa_or_af_path}")
            except Exception as e:
                print(f"Failed to get RSA for {protein_id} from {rsa_or_af_path}: {e}")
                continue

            grouped_info[ptm_type].append((protein_id, site, label, ptm_type))
            rsa_dict[ptm_type][protein_id] = rsa_values
            seq_dict[ptm_type][protein_id] = seq
    return grouped_info, rsa_dict, seq_dict

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


def run_inference(config):
    ptm_to_residue = {
        'SAPPphos': ['S', 'T'], 'SAPP-methylR': ['R'], 'SAPP-phosY': ['Y'],
        'SAPP-sumoK': ['K'], 'SAPP-methylK': ['K'],
        'SAPP-acetylK': ['K'], 'SAPP-ubiquitinK': ['K'],
        'SAPP-CMGC':['S','T'],'SAPP-CAMK':['S','T'],'SAPP-CDK':['S','T'],
        'SAPP-AGC':['S','T'],'SAPP-MAPK':['S','T'],'SAPP-PKA':['S','T'],
        'SAPP-PKC':['S','T'],'SAPP-CK2':['S','T']
       }

    input_path = config["input_path"]
    fasta_path = config["fasta_path"]
    output_path = config["output_path"]
    device = config.get("device", "cpu")
    
    grouped_info, rsa_dict, seq_dict = load_and_group_data(input_path,fasta_path, ptm_to_residue)
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
        label_tensor = test_tensors[4]

        loader = DataLoader(TensorDataset(*test_tensors), batch_size=batch_size, shuffle=False)

        model = SAPP_Model(
            vocab_size=ptm_config.get("embedding_dim", 22),
            window = window, 
            hidden=ptm_config.get("hidden_size", 256),
            n_layers=ptm_config.get("n_layers", 2),
            attn_heads=ptm_config.get("attn_heads", 4),
            feed_forward_dim=ptm_config.get("feed_forward_dim", 758),
            device=device
        ).to(device)

        pred_list = []
        weight_files = resolve_model_paths(ptm_config.get("model_paths"))
        all_model_results = []  

        for weight_file in weight_files:
            model.load_state_dict(torch.load(weight_file))
            model.eval()

            preds = []
            for batch in tqdm(loader, desc=f"{ptm_type} - {weight_file}"):
                batch = [b.to(device) for b in batch]
                pred, _ = model(batch[0], batch[1], batch[2], batch[3])
                preds.append(pred.view(-1).detach().cpu().numpy())
            
            pred_values = np.concatenate(preds)
            pred_list.append(pred_values)
            
            model_name = os.path.splitext(os.path.basename(weight_file))[0]
            model_result_df = pd.DataFrame({
                'ProteinID': [info[0] for info in info_list],
                'Site': [info[1] for info in info_list],
                'Pred': pred_values,
                'Label': label_tensor.cpu().numpy(),
                'PTMType': ptm_type,
                'Model': model_name
            })
            all_model_results.append(model_result_df)

        averaged_preds = np.mean(np.stack(pred_list), axis=0)

        result_df = pd.DataFrame({
            'ProteinID': [info[0] for info in info_list],
            'Site': [info[1] for info in info_list],
            'Pred': averaged_preds,
            'Label': label_tensor.cpu().numpy(),
            'PTMType': ptm_type
        })

        all_results.append(result_df)

    ensemble_path = output_path + '_ensemble.csv'
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(ensemble_path, index=False)
    print(f"Averaged inference results saved to: {ensemble_path}")
    
    modelwise_output_path = output_path + '_by_model.csv'
    modelwise_df = pd.concat(all_model_results, ignore_index=True)
    modelwise_df.to_csv(modelwise_output_path, index=False)
    print(f"Model-wise inference results saved to: {modelwise_output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on PTM data using SAPP")
    parser.add_argument('--config', type=str,required=True, help='Path to config JSON file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)

    run_inference(config)
