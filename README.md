# PLiSAGE: Enhancing Protein-Ligand Interaction Prediction with Multimodal Surface and Geometry Encoding

[![中文版](https://img.shields.io/badge/-%E4%B8%AD%E6%96%87%E7%89%88-blue)](README_ZH.md)

## Introduction

PLiSAGE is a deep learning framework for protein representation learning. It leverages a novel multimodal learning strategy to simultaneously learn features from both the **atomic-level structure graph** and the **3D surface geometry** of proteins. The goal is to generate more comprehensive and robust protein representations for downstream biological tasks, such as predicting Protein-Ligand Interactions (PLI) and Drug-Target Interactions (DTI).

![Model Architecture](image/architecture.jpg)

## Architecture Overview

Our model consists of two parallel encoder branches that process the two modalities of protein information:

1.  **Structure Encoder**: We use a Geometric Vector Perceptron (GVP) to process the atomic-level structure graph of the protein. This model can learn both scalar features (e.g., amino acid type) and vector features (e.g., atomic coordinates, orientations), effectively capturing its covalent bond structure and spatial conformation.

2.  **Surface Encoder**: We have designed a PointNet++ style Transformer model based on the Masked Autoencoder (MAE) concept. This model first samples the 3D protein surface into a point cloud and computes geometric (e.g., curvature) and chemical (e.g., nearby atom types) features for each point. During the pre-training phase, by randomly masking a large portion of the surface point cloud, the model is forced to infer and reconstruct the complete surface from the visible local regions, thereby learning powerful representations of surface geometry and chemical properties.

In downstream tasks, the features extracted by these two encoders are fused through a Transformer-based fusion module and finally combined with the features of the ligand/drug molecule for prediction.

## Installation & Environment Setup

We strongly recommend using [Conda](https://docs.conda.io/en/latest/) to manage the project environment.

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/PLiSAGE.git
cd PLiSAGE
```

**2. Create and Activate the Conda Environment using `environment.yaml`**

We provide a complete environment configuration file to create the required environment with a single command.
```bash
# This will create a new environment named 'plisage'
conda env create -f environment.yaml

# Activate the newly created environment
conda activate plisage
```

**3. Compile Custom CUDA Extensions**

This project relies on several custom CUDA operators, which must be compiled manually after the Conda environment has been created. Please execute the following commands in the project root directory:

```bash
# Compile Chamfer Distance
cd extensions/chamfer_dist/
python setup.py install
cd ../../

# Compile Earth Mover's Distance
cd extensions/emd/
python setup.py install
cd ../../

# Compile PointNet++ and KNN related operators (if needed)
# Please follow the instructions from the source of pointnet2_ops and knn_cuda in your project.
# For example: pip install pointnet2_ops_lib/. (assuming they are local directories)
```

## Data Preparation

Before model training, the raw data needs to be pre-processed into the `.npz` format that the model can read.

**1. Pre-training Data**

*   **Data**: Place the protein PDB files (or `.pdb.gz` files) for pre-training in a directory, e.g., `data/pre_data/`.
*   **Run Script**: Execute the following command. Please replace `--pdb_directory` and `--output_directory` with your actual paths.
    ```bash
    python predata_process.py --pdb_directory ./data/pre_data --output_directory ./processed_data/processed_pre_data
    ```

**2. PLI Downstream Task Data (PDBbind)**

*   **Data**: Unpack the PDBbind dataset and ensure its directory structure matches the script's expectations (e.g., `data/downsteam_data/pdbbind_v2020/1a0q/1a0q_protein.pdb`).
*   **Configuration**: Modify the `pli_processing` section in `configs/downsteam_config.yml` to ensure `raw_data_dir` and `index_file_path` point to your correct PDBbind paths.
*   **Run Script**:
    ```bash
    python downstreamtasks/pli/pli_data_process.py --config configs/downsteam_config.yml
    ```

**3. DTI Downstream Task Data**

*   **Data**: Prepare a CSV file containing protein IDs and drug SMILES strings, along with the corresponding PDB files.
*   **Configuration**: Modify the `dti_processing` section in `configs/downsteam_config.yml` to ensure the paths are correct.
*   **Run Script**:
    ```bash
    python downstreamtasks/dti/dti_data_process.py --config configs/downsteam_config.yml
    ```

## How to Reproduce

All training processes are driven by configuration files for easy adjustment and reproduction.

### 1. Pre-training

*   **Config File**: `configs/pretrain_config.yml`
*   **Launch Command**:
    ```bash
    python pre_train.py --config configs/pretrain_config.yml
    ```
*   **Key Hyperparameters (in `pretrain_config.yml`)**:
    *   `seed`: 100
    *   `data.batch_size`: 64
    *   `training.num_epochs`: 100
    *   `optimizer.lr`: 0.0001
    *   `loss.temperature`: 0.05 (for contrastive loss)
    *   `loss.reconstruction_weight`: 1.0
    *   `loss.contrastive_weight`: 1.0
    *   `model.Point_MAE.transformer_config.mask_ratio`: 0.6

### 2. Fine-tuning on Downstream Tasks

#### Protein-Ligand Interaction (PLI)

*   **Config File**: `configs/downsteam_config.yml` (section `pli_training`)
*   **Launch Command**:
    ```bash
    python downstreamtasks/pli/pli.py --config configs/downsteam_config.yml
    ```
*   **Key Hyperparameters (in `pli_training` section)**:
    *   `seed`: 100
    *   `data.batch_size`: 16
    *   `training.num_epochs`: 100
    *   `training.pretrained_ckpt_path`: './check_point/checkpoint.pth.tar'
    *   `optimizer.lr`: 0.0001
    *   `optimizer.weight_decay`: 0.0005

#### Drug-Target Interaction (DTI)

*   **Config File**: `configs/downsteam_config.yml` (section `dti_training`)
*   **Launch Command**:
    ```bash
    python downstreamtasks/dti/dti.py --config configs/downsteam_config.yml
    ```
*   **Key Hyperparameters (in `dti_training` section)**:
    *   `seed`: 42
    *   `data.batch_size`: 16
    *   `training.num_epochs`: 100
    *   `training.accumulation_steps`: 8
    *   `training.pretrained_ckpt_path`: './check_point/checkpoint.pth.tar'
    *   `optimizer.lr`: 0.0001

## Computational Resource Requirements

To successfully reproduce this project, we recommend the following or higher-spec computational resources:

*   **GPU**: All experiments in this project were conducted on NVIDIA GPUs. Due to the model size and point cloud processing requirements, a GPU with at least **16GB** of VRAM is recommended (e.g., NVIDIA V100, A100, or GeForce RTX 3090/4090).
*   **RAM**: The data preprocessing step, especially when handling large proteins, can be memory-intensive. **64GB** or more of physical RAM is recommended.
*   **Storage**: The raw datasets and the processed `.npz` files will occupy significant disk space. Please ensure at least **100-200GB** of available space, depending on the size of the datasets you use.
*   **Training Time**: On a single NVIDIA A100 GPU, the pre-training process with the example dataset takes several hours. Fine-tuning on downstream tasks is typically completed within 1-2 hours. The full pre-training time will depend on the scale of your dataset.

## Citation

If you use PLiSAGE in your research, please cite our paper:

```bibtex
@article{Wang2025PLiSAGE,
  author  = {Wang, Tianci and Qiao, Guanyu and Wang, Guohua and Li, Yang},
  title   = {{PLiSAGE}: Enhancing Protein-Ligand Interaction Prediction with Multimodal Surface and Geometry Encoding},
  journal = {Bioinformatics},
  year    = {2025},
  doi     = {10.1093/bioinformatics/btaf608},
  note    = {Manuscript Accepted},
  % --- 下面的信息将在最终排版后提供 ---
  % volume  = {XX},
  % number  = {YY},
  % pages   = {ZZZZ--ZZZZ},
}
```
