import torch
from tqdm import tqdm 
import numpy as np
import os

aa_dict = {'A': 1,'G': 2,'V': 3,'S': 4,'E': 5,'R': 6,'T': 7,
'I': 8,'D': 9,'P': 10,'K': 11,'Q': 12,'N': 13,'F': 14,'Y': 15,
'M': 16,'H': 17,'W': 18,'C': 19,'-': 0,'L': 20}

def read_fasta_as_dict(fasta_path):
    protein_dict = {}
    with open(fasta_path, 'r') as f:
        current_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    protein_dict[current_id] = "".join(seq_lines)
                current_id = line[1:].split()[0]  # >ProteinID description → ID만 추출
                seq_lines = []
            else:
                seq_lines.append(line)
        if current_id is not None:
            protein_dict[current_id] = "".join(seq_lines)
    return protein_dict


def load_sequence_and_rsa_with_npy(csv_path, fasta_path, rsa_dir, target_residue):
    protein_dic = read_fasta_as_dict(fasta_path)
    rsa_dic = {}
    data_info = []

    for pid in protein_dic:
        rsa_path = os.path.join(rsa_dir, f"{pid}.npy")
        if os.path.exists(rsa_path):
            rsa_dic[pid] = np.load(rsa_path)
        else:
            rsa_dic[pid] = [-1.0] * len(protein_dic[pid])

    with open(csv_path) as f:
        for line in f:
            tokens = line.strip().split("\t")
            if len(tokens) < 4:
                continue
            pid, site, res, label = tokens[0], int(tokens[1]), tokens[2], int(tokens[3])
            if res in target_residue:
                data_info.append((pid, site, label))
    return data_info, protein_dic, rsa_dic
    

def torch_binning(data,max_val,steps):
    bin_edges = torch.linspace(0,max_val,steps=steps)
    digitized = torch.bucketize(data.unsqueeze(-1)[:,:,-1],bin_edges)
    one_hot = torch.nn.functional.one_hot(digitized, num_classes=len(bin_edges)+1)
    
    return digitized,one_hot

def get_Data(data_info,protein_dic,rsa_dic,window):
    data_list = []
    rsa_list = []
    mask_list =[]
    rsamask_list = []
    y_list = []

    for li in tqdm(data_info):
        protein = li[0]
        site = li[1]
        label = li[2]
        seq = protein_dic[protein]
        RSA = rsa_dic[protein]
        onehot = np.zeros(window*2+1)
        rsa_feat = np.zeros(window*2+1)
        
        mask = np.zeros(window*2+1)
        rsa_mask = np.zeros(window*2+1)
        start_idx = max(0,site-window)
        end_idx = min(site+window,len(seq)-1)
        j = window - (site-start_idx)
        for i in range(start_idx, end_idx+1):
            onehot[j] = aa_dict[seq[i]]
            rsa_feat[j] = RSA[i]
            if RSA[i] != -1:
                rsa_mask[j] = 1

            if aa_dict[seq[i]] != 0:
                mask[j] = 1
            j += 1
        
        data_list.append(onehot)
        rsa_list.append(rsa_feat)
        mask_list.append(np.expand_dims(mask,-1)*np.expand_dims(mask,-1).T)
        rsamask_list.append(np.expand_dims(rsa_mask,-1)*np.expand_dims(rsa_mask,-1).T)
        y_list.append(label)
    
    data_list = torch.tensor(np.array(data_list), dtype=torch.long)
    y_list = torch.tensor(np.array(y_list), dtype=torch.float32)
    rsa_list = torch.tensor(np.array(rsa_list), dtype=torch.float32)
    rsamask_list = torch.tensor(np.array(rsamask_list), dtype=torch.bool)
    mask_list = torch.tensor(np.array(mask_list), dtype=torch.bool)
    
    _,rsa_list = torch_binning(rsa_list,1, 20)
    rsa_list = torch.tensor(np.array(rsa_list), dtype=torch.float32)

    return data_list,rsa_list,mask_list,rsamask_list,y_list

