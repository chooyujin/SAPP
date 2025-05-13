# SAPP: Structure-Aware PTM Prediction

SAPP is a transformer-based model for post-translational modification (PTM) site prediction, integrating protein sequence and structural features (RSA).

![SAPP_pipeline](https://github.com/user-attachments/assets/6f4d9584-d62e-4d3c-bd39-4b09c45feda8)

---

- [Supported PTM and Kinase Types](#supported-ptm-and-kinase-types)
- [Installation](#installation)
- [Inference](#inference)
  - [Required Input Files (Inference)](#required-input-files-inference)
  - [Configuration File (Inference)](#configuration-file-inference)
  - [Output format](#output-format)
  - [Command](#command)
- [Training](#inference)
  - [Required Input Files (Training)](#required-input-files-training)
  - [Configuration File (Training)](#configuration-file-training)
  - [Command (Training)](#command-training)
---

## Supported PTM and Kinase Types

- SAPP supports 7 PTM types prediction: 
  - SAPPphos: phosphorylation (S/T)
  - SAPP-phosY: phosphorylation (Y)
  - SAPP-sumoK: sumoylation (K)
  - SAPP-methylK: methylation (K)
  - SAPP-acetylK: acetylation (K)
  - SAPP-ubiquitinK: ubiquitination (K)
    
- SAPP supports 8 Kinase-specific types prediction:
  - CMGC, CAMK, CDK, AGC, MAPK, PKA, PKC, CK2

---

## Installation

Install basic dependencies:

```bash
conda env create -f environment.yml
conda activate sapp-env
```

To use GPU acceleration, install PyTorch separately with:

```bash
# Example: for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

See [PyTorch official site](https://pytorch.org/get-started/locally/) to choose the right CUDA version.

---

## Inference

To perform inference, you need:

- **Two input data files**
- **One configuration file**

### Required Input Files (Inference)

| Input File | Description | Format |
|------------|-------------|--------|
| **Feature table** | Tab-separated file with protein/PTM features | `.tsv` |
| **FASTA file** | Protein sequences for each `ProteinID` | `.fasta` |

#### Feature table (tsv format) 

| Column | Description |
|--------|-------------|
| ProteinID | Unique identifier for the protein |
| Site | 0-based index of the modified residue |
| Label | 1 if modified, 0 if not |
| PTMType | Type of PTM (e.g., SAPPphos, SAPP-methylR) |
| RSA_or_AF_path | Path to RSA `.npy` file or AlphaFold `.pdb`/`.cif` file (RSA will be computed if `.npy` is not provided) |

### Configuration File (Inference)
The third required file is a JSON configuration file that defines paths and runtime parameters:

| Key           | Description |
|----------------|-------------|
| `input_path`   | Path to the `.tsv` file containing PTM site information. See [TSV Format](#1-tsv-format) for column details. |
| `fasta_path`   | Path to the `.fasta` file containing the full protein sequences corresponding to the `ProteinID`s in the input file. Used for verifying or extracting context windows, and for matching AlphaFold structure files to their sequences. |
| `output_path`    | Full path including directory and filename prefix for saving prediction results. Two files will be created: `<output_path>_averaged.csv` and `<output_path>_by_model.csv`. |
| `default_config`  | Dictionary of default hyperparameters used for all models unless overridden  |
| `ptm_configs`  | Dictionary of model-specific configurations for each PTM type |

- `ptm_configs`: Dictionary of PTM-specific settings
  - Each key (e.g. `SAPPphos`) **must match the `PTMType` in the input `.tsv` file**
  - The system uses `model_paths` to load checkpoints
  - If a PTM entry overrides config values (like `hidden_size`), they replace the defaults for that model
  - User can add their own PTM model names to this dictionary, as long as they match corresponding `PTMType` values in the input `.tsv` file.

--- 

### Output Format

Running inference generates two `.csv` files:

1. **`*_ensemble.csv`**:  
   Contains the prediction results averaged across all model checkpoints for each PTM type.

2. **`*_by_model.csv`**:  
   Contains prediction results from each individual model checkpoint, with an additional `Model` column to indicate the source model.

#### Common Columns

| Column     | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| `ProteinID` | Identifier of the protein containing the modified residue                 |
| `Site`      | 0-based index of the residue in the sequence                              |
| `Pred`      | Predicted probability (between 0 and 1) of modification at the given site |
| `Label`     | Ground truth label (1 if modified, 0 if not)                              |
| `PTMType`   | Type of PTM (e.g., SAPPphos, SAPP-acetylK)                                |

#### Additional Column in `*_by_model.csv`

| Column  | Description                                      |
|---------|--------------------------------------------------|
| `Model` | Name of the model checkpoint that made the prediction |


---

### Command

```bash
python inference.py \
  --config inference_config.json \
```

The `example_folder/` includes example files:
- `inference_input_file.tsv`
- `inference_input_protein.fasta`
- `inference_config.json`

These example files can be used to test the inference pipeline, which takes about **1 minute** to run on a single GPU (e.g., RTX 6000).

---

## Training

To train a model from scratch, prepare a training configuration file and required input data.

### Required Input Files (Training)

| Input File         | Description                                                      | Format     |
|--------------------|------------------------------------------------------------------|------------|
| **Feature table**   | Tab-separated file with protein/PTM features | `.tsv`     |
| **FASTA file**     | Protein sequences for each `ProteinID`              | `.fasta`   |
| **RSA Directory**  | Folder containing RSA .npy files named after `ProteinID` (e.g., P12345.npy)| directory  |

> These files should be referenced in the `path_config` section of `train_config.json`.

#### Feature table (tsv format) 

| Column | Description |
|--------|-------------|
| ProteinID | Unique identifier for the protein |
| Site | 0-based index of the modified residue |
| Residue | Amino acid (one-letter code) at the modification site |
| Label | 1 if modified, 0 if not |

### Configuration File (Training)

The `train_config.json` contains paths, model settings, PTM-specific options, and training parameters.

| Section          | Key                 | Description                                                                 |
|------------------|---------------------|-----------------------------------------------------------------------------|
| `path_config`    | `train_data_path`   | Path to training `.tsv` file                                                |
|                  | `train_fasta_path`  | Path to corresponding protein `.fasta` file                                 |
|                  | `train_rsa_path`    | Directory with RSA `.npy`                                                   |
|                  | `weight_save_dir`   | Directory to save trained model weights                                     |
| `ptm_config`     | `target_residue`    | List of target residues for the PTM type (e.g., `S`, `T`)                   |
| `model_config`   | `window`            | Sliding window size for input embedding                                     |
|                  | `embedding_dim`     | Size of amino acid embedding vector                                         |
|                  | `hidden`            | Hidden layer size                                                           |
|                  | `n_layers`          | Number of transformer layers                                                |
|                  | `attn_heads`        | Number of attention heads                                                   |
|                  | `feed_forward_dim`  | Dimension of feed-forward network                                           |
| `train_config`   | `epochs`            | Total number of training epochs                                             |
|                  | `use_KFold`         | Whether to use K-Fold cross-validation (`true` / `false`)                   |
|                  | `Folds`             | Number of folds for cross-validation                                        |
|                  | `dropout`           | Dropout rate                                                                |
|                  | `train_batch_size`  | Batch size during training                                                  |
|                  | `valid_batch_size`  | Batch size during validation                                                |
|                  | `learning_rate`     | Learning rate for optimizer                                                 |
|                  | `weight_decay`      | Weight decay (L2 regularization)                                            |
|                  | `schedular_Tmax`    | Scheduler maximum cycle length (cosine annealing)                           |
|                  | `schedular_eatmin`  | Scheduler minimum learning rate                                             |
|                  | `patient_limit`     | Early stopping patience threshold                                           |
|                  | `random_seed`       | Seed for reproducibility                                                    |
|                  | `device`            | Device for training (e.g., `"cuda:0"` or `"cpu"`)                           |
|                  | `pretrained_model_path`            | (optional) Path to a pretrained model checkpoint to initialize weights                           |
|                  | `freeze_backbone`            | (optional) if true, freezes the transformer backbone during fine-tuning                           |

---

### Command

```bash
python train.py \
  --config train_config.json \
```

