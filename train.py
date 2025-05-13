import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
import torch.optim as optim

from src.datasets.dataprocess import get_Data, load_sequence_and_rsa_with_npy
from src.models.Model import SAPP_Model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    preds, labels = [], []
    for batch in tqdm(loader, desc="Training"):
        batch = [b.to(device) for b in batch]
        optimizer.zero_grad()
        pred, _ = model(batch[0], batch[1], batch[2], batch[3])
        loss = criterion(pred.view(-1), batch[4].float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.append(pred.view(-1).detach().cpu())
        labels.append(batch[4].detach().cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc = ((preds > 0.5) == labels).float().mean().item()
    return total_loss / len(loader), acc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = [b.to(device) for b in batch]
            pred, _ = model(batch[0], batch[1], batch[2], batch[3])
            loss = criterion(pred.view(-1), batch[4].float())
            total_loss += loss.item()
            preds.append(pred.view(-1).detach().cpu())
            labels.append(batch[4].detach().cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc = ((preds > 0.5) == labels).float().mean().item()
    return total_loss / len(loader), acc

def run_training(config):
    path_cfg = config["path_config"]
    ptm_cfg = config["ptm_config"]
    model_cfg = config["model_config"]
    train_cfg = config["train_config"]
    device = torch.device(train_cfg["device"])
    
    os.makedirs(path_cfg["weight_save_dir"], exist_ok=True)

    print("--- Data Processing ---")
    data_info, prot_dict, rsa_dict = load_sequence_and_rsa_with_npy(
        path_cfg["train_data_path"], path_cfg["train_fasta_path"], path_cfg["train_rsa_path"], ptm_cfg["target_residue"]
    )
    labels = [x[2] for x in data_info]
    use_kfold = train_cfg.get("use_KFold", False)
    if use_kfold:
        print(f"--- {train_cfg['Folds']}-Fold processing ---")
        kfold = KFold(n_splits=train_cfg["Folds"], shuffle=True,random_state=train_cfg.get("random_seed", 42))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
            print(f"--- Fold {fold+1}/{train_cfg['Folds']} ---")
            train_info = [data_info[i] for i in train_idx]
            val_info = [data_info[i] for i in val_idx]
            train_loader = DataLoader(TensorDataset(*get_Data(train_info, prot_dict, rsa_dict, model_cfg["window"])),
                                      batch_size=train_cfg["train_batch_size"], shuffle=True)
            val_loader = DataLoader(TensorDataset(*get_Data(val_info, prot_dict, rsa_dict, model_cfg["window"])),
                                    batch_size=train_cfg["valid_batch_size"], shuffle=False)
            train_model(train_loader, val_loader, model_cfg, train_cfg, path_cfg, device, fold+1)
    else:
        if "val_data_path" in path_cfg and "val_fasta_path" in path_cfg and "val_rsa_path" in path_cfg:
            print(f"validation data path exists")
            val_info, val_prot_dict, val_rsa_dict = load_sequence_and_rsa_with_npy(
                path_cfg["val_data_path"], path_cfg["val_fasta_path"], path_cfg["val_rsa_path"], ptm_cfg["target_residue"]
            )
        else:
            test_size = train_cfg.get("valid_size", 0.2)
            print(f"validation data path not exists.. train_valid split {test_size}")
            
            train_info, val_info = train_test_split(data_info, test_size=test_size, random_state=train_cfg.get("random_seed",42), stratify=labels)
            val_prot_dict = prot_dict
            val_rsa_dict = rsa_dict
            data_info = train_info

    train_loader = DataLoader(TensorDataset(*get_Data(data_info, prot_dict, rsa_dict, model_cfg["window"])),
                                  batch_size=train_cfg["train_batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(*get_Data(val_info, val_prot_dict, val_rsa_dict, model_cfg["window"])),
                            batch_size=train_cfg["valid_batch_size"], shuffle=False)
    train_model(train_loader, val_loader, model_cfg, train_cfg, path_cfg, device)


def train_model(train_loader, val_loader, model_cfg, train_cfg, path_cfg, device, fold=None):
    model = SAPP_Model(
        vocab_size=model_cfg["embedding_dim"],
        window=model_cfg["window"],
        hidden=model_cfg["hidden"],
        n_layers=model_cfg["n_layers"],
        attn_heads=model_cfg["attn_heads"],
        feed_forward_dim=model_cfg["feed_forward_dim"],
        dropout=train_cfg["dropout"],
        device=device
    ).to(device)

    # Load pretrained weights if provided
    if "pretrained_model_path" in train_cfg:
        print(f"Loading pretrained weights from {train_cfg['pretrained_model_path']}")
        model.load_state_dict(torch.load(train_cfg["pretrained_model_path"], map_location=device), strict=False)

        if train_cfg.get("freeze_backbone", False):
            print("Freezing model backbone parameters...")
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
                    
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["schedular_Tmax"], eta_min=train_cfg["schedular_eatmin"])
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patient_counter = 0
    model_name = f"best_fold_{fold}.pt" if fold else "best.pt"

    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patient_counter = 0
            torch.save(model.state_dict(), os.path.join(path_cfg["weight_save_dir"], model_name))
        else:
            patient_counter += 1
            if patient_counter >= train_cfg["patient_limit"]:
                print("Early stopping triggered.")
                break

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAPP model")
    parser.add_argument("--config", type=str, required=True, help="Path to train_config.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)
    run_training(config)
