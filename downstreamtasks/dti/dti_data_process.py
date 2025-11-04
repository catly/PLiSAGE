import os
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import atom3d.util.formats as fo
import logging
import argparse
import yaml
from easydict import EasyDict

from utils.protein_utils import featurize_as_graph
from utils.geometry_processing import atoms_to_points_normals, curvatures
from utils.common_data_processing import load_structure_from_pdb
from utils.ligand_processing import read_ligand_from_smiles, construct_ligand_graph

# Setup logging and suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_protein_graph(protein_path):
    protein_df = fo.bp_to_df(fo.read_pdb(protein_path))
    protein_graph = featurize_as_graph(protein_df)
    return protein_graph

def save_processed_data_as_npz(protein, ligand, atom_coords, atom_types, xyz, normals, curvature, npz_file):
    protein_graph_data = {
        'protein_x': protein.x.cpu().numpy(),
        'protein_edge_index': protein.edge_index.cpu().numpy(),
        'seq': protein.seq.cpu().numpy(),
        'node_s': protein.node_s.cpu().numpy(),
        'node_v': protein.node_v.cpu().numpy(),
        'edge_s': protein.edge_s.cpu().numpy(),
        'edge_v': protein.edge_v.cpu().numpy()
    }
    ligand_graph_data = {
        'ligand_x': ligand.x.cpu().numpy(),
        'ligand_edge_index': ligand.edge_index.cpu().numpy(),
        'edge_attr': ligand.edge_attr.cpu().numpy(),
        'y': ligand.y.cpu().numpy(),
    }
    np.savez(npz_file, **protein_graph_data, **ligand_graph_data,
             atom_coords=atom_coords.cpu().numpy(),
             atom_types=atom_types.cpu().numpy(),
             xyz=xyz.cpu().numpy(),
             normals=normals.cpu().numpy(),
             curvature=curvature.cpu().numpy())
    logging.debug(f"Saved processed data to {npz_file}")

def process_dti_data(csv_path, pdb_dir, output_dir, curvature_scales):
    data_df = pd.read_csv(csv_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing DTI data"):
        pdb_id = row['ID']
        smiles = row['SMILES']
        label = float(row['Label'])

        pdb_file_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_file_path):
            logging.warning(f"PDB file not found for {pdb_id}, skipping...")
            continue

        try:
            _, drug_atom_fea, mol, h_num_lig = read_ligand_from_smiles(smiles)
            drug_graph = construct_ligand_graph(drug_atom_fea, mol, h_num_lig, label)

            protein_graph = load_protein_graph(pdb_file_path)
            atom_coords, atom_types = load_structure_from_pdb(pdb_file_path)
            
            xyz, normals, batch = atoms_to_points_normals(atom_coords, batch=torch.zeros(atom_coords.shape[0], dtype=torch.long))
            curvature = curvatures(xyz, triangles=None, normals=normals, scales=curvature_scales, batch=batch)

            npz_file = os.path.join(output_dir, f"{pdb_id}.npz")
            save_processed_data_as_npz(protein_graph, drug_graph, atom_coords, atom_types, xyz, normals, curvature, npz_file)
        except Exception as e:
            logging.error(f"Failed to process {pdb_id}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data for DTI task.')
    parser.add_argument('--config', type=str, default='configs/downsteam_config.yml', help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    
    cfg = config.dti_processing
    logging.info(f"Starting DTI data processing with config from {args.config}")
    process_dti_data(
        csv_path=cfg.csv_path,
        pdb_dir=cfg.pdb_dir,
        output_dir=cfg.output_dir,
        curvature_scales=cfg.curvature_scales
    )
    logging.info("DTI data processing finished.")