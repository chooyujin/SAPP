# SAPP: Structure-Aware PTM Prediction

SAPP is a transformer-based model for post-translational modification (PTM) site prediction.  
It integrates protein sequence information with structural features, particularly residue-level relative solvent accessibility (RSA).

![SAPP_pipeline](https://github.com/user-attachments/assets/6f4d9584-d62e-4d3c-bd39-4b09c45feda8)

---

## Table of Contents

- [Supported PTM and Kinase Types](#supported-ptm-and-kinase-types)
- [Installation](#installation)
- [Inference](#inference)
  - [Required Input Files](#required-input-files-inference)
  - [Feature Table Format](#feature-table-format-inference)
  - [RSA and Structure Alignment](#rsa-and-structure-alignment)
  - [Configuration File](#configuration-file-inference)
  - [Output Format](#output-format)
  - [Command](#command-inference)
- [Training](#training)
  - [Required Input Files](#required-input-files-training)
  - [Feature Table Format](#feature-table-format-training)
  - [Configuration File](#configuration-file-training)
  - [Command](#command-training)
- [SAPP Datasets and Result Files](#sapp-datasets-and-result-files)
- [Citation](#citation)

---

## Supported PTM and Kinase Types

SAPP supports prediction for multiple PTM types and kinase-specific phosphorylation models.

### PTM type prediction

| Model name | PTM type | Target residue |
|-----------|----------|----------------|
| `SAPPphos` | Phosphorylation | S/T |
| `SAPP-phosY` | Phosphorylation | Y |
| `SAPP-sumoK` | Sumoylation | K |
| `SAPP-methylK` | Methylation | K |
| `SAPP-acetylK` | Acetylation | K |
| `SAPP-ubiquitinK` | Ubiquitination | K |
| `SAPP-methylR` | Methylation | R |

### Kinase-specific phosphorylation prediction

| Model name | Target residue |
|-----------|----------------|
| `SAPP-CMGC` | S/T |
| `SAPP-CAMK` | S/T |
| `SAPP-CDK` | S/T |
| `SAPP-AGC` | S/T |
| `SAPP-MAPK` | S/T |
| `SAPP-PKA` | S/T |
| `SAPP-PKC` | S/T |
| `SAPP-CK2` | S/T |

The `PTMType` value in the input file must match one of the model names defined in the configuration file.

---

## Installation

Install the required dependencies using the provided conda environment file:

```bash
conda env create -f environment.yml
conda activate sapp-env
```

To use GPU acceleration, install PyTorch separately according to your CUDA version.

Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Please refer to the [PyTorch official installation guide](https://pytorch.org/get-started/locally/) to select the appropriate version for your system.

---

## Inference

To run inference, prepare:

1. A feature table in `.tsv` format  
2. A FASTA file containing protein sequences  
3. A JSON configuration file  

SAPP can use either precomputed RSA arrays or protein structure files as structural input.

---

### Required Input Files Inference

| Input file | Description | Format |
|-----------|-------------|--------|
| Feature table | Protein, site, PTM type, and RSA/structure path information | `.tsv` |
| FASTA file | Protein sequences corresponding to `ProteinID` values in the feature table | `.fasta` |
| Configuration file | Runtime, model, and checkpoint settings | `.json` |

---

### Feature Table Format Inference

The inference feature table should be a tab-separated file with either **4 columns** or **5 columns**.

#### Format without labels

```text
ProteinID    Site    PTMType    RSA_or_AF_path
```

#### Format with labels

```text
ProteinID    Site    PTMType    RSA_or_AF_path    Label
```

| Column | Required | Description |
|--------|----------|-------------|
| `ProteinID` | Yes | Unique identifier of the protein. This ID must be present in the input FASTA file. |
| `Site` | Yes | 0-based index of the target residue in the protein sequence. |
| `PTMType` | Yes | PTM type or kinase-specific model name. This value must match a key in `ptm_configs` in the configuration file. |
| `RSA_or_AF_path` | Yes | Path to a residue-level RSA `.npy` file or a protein structure file in `.pdb` or `.cif` format. |
| `Label` | Optional | Ground-truth label. Use `1` for modified sites and `0` for unmodified sites. If omitted, inference still runs and the output files will not include the `Label` column. |

Example without labels:

```text
P12345    135    SAPPphos       /path/to/P12345.npy
Q99999    42     SAPP-acetylK   /path/to/AF-Q99999-F1-model_v4.cif
```

Example with labels:

```text
P12345    135    SAPPphos       /path/to/P12345.npy                  1
Q99999    42     SAPP-acetylK   /path/to/AF-Q99999-F1-model_v4.cif   0
```

`Site` must correspond to a valid target residue for the selected `PTMType`.  
For example, `SAPPphos` requires `S` or `T`, while `SAPP-acetylK` requires `K`.

---

### RSA and Structure Alignment

SAPP uses residue-level relative solvent accessibility, or RSA, as a structural feature.  
RSA can be provided in two ways.

#### 1. Precomputed RSA array `.npy`

If `RSA_or_AF_path` points to a `.npy` file:

- The file should contain one RSA value per residue.
- The RSA array length must be identical to the corresponding protein sequence length in the FASTA file.
- If the RSA length does not match the sequence length, the entry is skipped.

#### 2. Protein structure file `.pdb` or `.cif`

If `RSA_or_AF_path` points to a `.pdb` or `.cif` file:

- SAPP computes RSA values from the structure.
- The structure-derived sequence is aligned to the input FASTA sequence.
- This allows SAPP to handle structures with missing or unmatched residues.
- Residues without valid structural coverage are represented as missing values during alignment.
- For model input, missing or uncovered RSA positions are masked.
- Alignment quality metrics are reported in the output files.

This is particularly useful when using AlphaFold structures, because some structures may contain missing residues or may not perfectly match the reference FASTA sequence.

The following quality-control columns are included in the output:

| Column | Description |
|--------|-------------|
| `ProteinCoverage` | Fraction of residues in the full protein sequence with valid RSA values after structure-to-sequence alignment. |
| `WindowCoverage51mer` | Fraction of residues with valid RSA values in the local sequence window around the target site. |
| `Identity` | Sequence identity between the structure-derived sequence and the input FASTA sequence during alignment. For `.npy` RSA inputs, this value is reported as `NaN`. |
| `SiteCovered` | Whether the target residue itself has a valid RSA value after alignment. |

Low `WindowCoverage51mer` or `SiteCovered = False` indicates that the local structural context around the target site is partially or fully missing.

> Note: `WindowCoverage51mer` reflects the default setting `window_size = 25`, which corresponds to a 51-residue local window. If a different `window_size` is used in the configuration file, this value represents the coverage of the corresponding local window.

---

### Configuration File Inference

The inference configuration file is a JSON file that defines input paths, output paths, runtime settings, and model checkpoint paths.

Example:

```json
{
  "input_path": "example_folder/inference_input_file.tsv",
  "fasta_path": "example_folder/inference_input_protein.fasta",
  "output_path": "example_folder/result",
  "device": "cuda:0",
  "default_config": {
    "window_size": 25,
    "batch_size": 128,
    "embedding_dim": 22,
    "hidden_size": 256,
    "n_layers": 2,
    "attn_heads": 4,
    "feed_forward_dim": 758
  },
  "ptm_configs": {
    "SAPPphos": {
      "model_paths": ["weights/SAPPphos/"]
    },
    "SAPP-acetylK": {
      "model_paths": [
        "weights/SAPP-acetylK/model_fold1.pt",
        "weights/SAPP-acetylK/model_fold2.pt"
      ]
    }
  }
}
```

| Key | Description |
|-----|-------------|
| `input_path` | Path to the inference feature table in `.tsv` format. |
| `fasta_path` | Path to the FASTA file containing protein sequences. |
| `output_path` | Output filename prefix. Two files will be generated: `<output_path>_ensemble.csv` and `<output_path>_by_model.csv`. |
| `device` | Device used for inference, such as `"cuda:0"` or `"cpu"`. |
| `default_config` | Default model and inference parameters applied to all PTM types unless overridden. |
| `ptm_configs` | PTM-specific model settings. Each key must match the `PTMType` values in the input `.tsv` file. |

#### `default_config`

| Key | Description |
|-----|-------------|
| `window_size` | Half-window size around the target residue. The default value `25` gives a 51-residue window. |
| `batch_size` | Batch size for inference. |
| `embedding_dim` | Amino acid embedding dimension. |
| `hidden_size` | Hidden layer size of the model. |
| `n_layers` | Number of transformer layers. |
| `attn_heads` | Number of attention heads. |
| `feed_forward_dim` | Feed-forward network dimension. |

#### `ptm_configs`

Each `ptm_configs` entry should include `model_paths`.

`model_paths` can be:

- a list of `.pt` checkpoint files, or
- a list of directories containing `.pt` checkpoint files.

If a directory is provided, all `.pt` files in the directory are loaded and used as an ensemble for that `PTMType`.

Example:

```json
"ptm_configs": {
  "SAPPphos": {
    "model_paths": ["weights/SAPPphos/"]
  },
  "SAPP-CK2": {
    "model_paths": ["weights/SAPP-CK2/model_fold1.pt", "weights/SAPP-CK2/model_fold2.pt"]
  }
}
```

PTM-specific entries can override values from `default_config`.  
For example:

```json
"SAPPphos": {
  "window_size": 25,
  "batch_size": 256,
  "model_paths": ["weights/SAPPphos/"]
}
```

---

### Output Format

Running inference generates two `.csv` files.

#### 1. `<output_path>_ensemble.csv`

This file contains prediction results averaged across all model checkpoints specified for each `PTMType`.

#### 2. `<output_path>_by_model.csv`

This file contains prediction results from each individual model checkpoint.  
It includes an additional `Model` column indicating which checkpoint generated the prediction.

---

#### Columns in `<output_path>_ensemble.csv`

| Column | Description |
|--------|-------------|
| `ProteinID` | Identifier of the protein containing the target residue. |
| `Site` | 0-based index of the target residue in the input protein sequence. |
| `PTMType` | PTM type or kinase-specific model used for inference. |
| `Label` | Optional ground-truth label. This column is included only when labels are provided in the input file. |
| `Pred` | Predicted probability of modification at the given site. If multiple checkpoints are used, this value is averaged across models. |
| `ProteinCoverage` | Fraction of residues in the full protein sequence with valid RSA values after alignment. |
| `WindowCoverage51mer` | Fraction of residues with valid RSA values in the local sequence window around the target site. |
| `Identity` | Sequence identity between the structure-derived sequence and the input FASTA sequence. For `.npy` RSA inputs, this value is `NaN`. |
| `SiteCovered` | Whether the target residue has a valid RSA value after alignment. |

#### Columns in `<output_path>_by_model.csv`

| Column | Description |
|--------|-------------|
| `ProteinID` | Identifier of the protein containing the target residue. |
| `Site` | 0-based index of the target residue in the input protein sequence. |
| `PTMType` | PTM type or kinase-specific model used for inference. |
| `Label` | Optional ground-truth label. This column is included only when labels are provided in the input file. |
| `Model` | Name of the model checkpoint used for prediction. |
| `Pred` | Predicted probability of modification from the individual model checkpoint. |
| `ProteinCoverage` | Fraction of residues in the full protein sequence with valid RSA values after alignment. |
| `WindowCoverage51mer` | Fraction of residues with valid RSA values in the local sequence window around the target site. |
| `Identity` | Sequence identity between the structure-derived sequence and the input FASTA sequence. For `.npy` RSA inputs, this value is `NaN`. |
| `SiteCovered` | Whether the target residue has a valid RSA value after alignment. |

---

### Command Inference

Run inference with:

```bash
python inference.py \
  --config inference_config.json
```

The `example_folder/` includes example files:

- `inference_input_file.tsv`
- `inference_input_protein.fasta`
- `inference_config.json`

These example files can be used to test the inference pipeline.

---

## Training

To train a SAPP model from scratch, prepare a training configuration file and the required input data.

---

### Required Input Files Training

| Input file | Description | Format |
|-----------|-------------|--------|
| Feature table | Protein site-level training data | `.tsv` |
| FASTA file | Protein sequences corresponding to `ProteinID` values | `.fasta` |
| RSA directory | Directory containing RSA `.npy` files named after `ProteinID` | directory |

RSA files should be named using the corresponding `ProteinID`.

Example:

```text
P12345.npy
Q99999.npy
```

---

### Feature Table Format Training

The training feature table should be a tab-separated file.

| Column | Description |
|--------|-------------|
| `ProteinID` | Unique identifier of the protein. |
| `Site` | 0-based index of the target residue. |
| `Residue` | Amino acid at the target site, represented using a one-letter code. |
| `Label` | Ground-truth label. Use `1` for modified sites and `0` for unmodified sites. |

Example:

```text
P12345    135    S    1
P12345    210    T    0
Q99999    42     K    1
```

---

### Configuration File Training

The training configuration file defines data paths, PTM-specific settings, model parameters, and training options.

| Section | Key | Description |
|---------|-----|-------------|
| `path_config` | `train_data_path` | Path to the training `.tsv` file. |
|  | `train_fasta_path` | Path to the corresponding training FASTA file. |
|  | `train_rsa_path` | Directory containing RSA `.npy` files. |
|  | `weight_save_dir` | Directory where trained model weights will be saved. |
| `ptm_config` | `target_residue` | Target residues for the PTM type, such as `["S", "T"]` or `["K"]`. |
| `model_config` | `window` | Half-window size around the target residue. |
|  | `embedding_dim` | Amino acid embedding dimension. |
|  | `hidden` | Hidden layer size. |
|  | `n_layers` | Number of transformer layers. |
|  | `attn_heads` | Number of attention heads. |
|  | `feed_forward_dim` | Feed-forward network dimension. |
| `train_config` | `epochs` | Number of training epochs. |
|  | `use_KFold` | Whether to use K-fold cross-validation. |
|  | `Folds` | Number of folds for cross-validation. |
|  | `dropout` | Dropout rate. |
|  | `train_batch_size` | Batch size for training. |
|  | `valid_batch_size` | Batch size for validation. |
|  | `learning_rate` | Learning rate. |
|  | `weight_decay` | Weight decay for regularization. |
|  | `schedular_Tmax` | Maximum cycle length for the scheduler. |
|  | `schedular_eatmin` | Minimum learning rate for the scheduler. |
|  | `patient_limit` | Early stopping patience. |
|  | `random_seed` | Random seed for reproducibility. |
|  | `device` | Device for training, such as `"cuda:0"` or `"cpu"`. |
|  | `pretrained_model_path` | Optional path to a pretrained checkpoint for initialization. |
|  | `freeze_backbone` | Optional setting. If `true`, freezes the transformer backbone during fine-tuning. |

---

### Command Training

Run training with:

```bash
python train.py \
  --config train_config.json
```

---

## SAPP Datasets and Result Files

SAPP datasets and result files are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15393286.svg)](https://doi.org/10.5281/zenodo.15393286)

Link: https://zenodo.org/records/15393286

---

## Citation

If you use SAPP, please cite:

[Structure Aware PTM Prediction. Yujin Choo, Seungjin Na, and Eunok Paek, Computers in Biology and Medicine, 2025.](https://www.sciencedirect.com/science/article/pii/S0010482525015227)
