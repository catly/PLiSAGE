import os
import torch
import numpy as np
from Bio.PDB import PDBParser
import atom3d.util.formats as fo
from utils.protein_utils import featurize_as_graph
from utils.geometry_processing import atoms_to_points_normals, curvatures
import logging
from tqdm import tqdm
import argparse

# Setup logging for clear and structured output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Maps chemical elements to a numeric index for one-hot encoding.
_ELE2NUM = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}

def load_protein_graph(protein_path):
    """Loads a protein from a PDB file and converts it into a graph structure."""
    protein_df = fo.bp_to_df(fo.read_pdb(protein_path))
    protein_graph = featurize_as_graph(protein_df)
    return protein_graph

def load_structure(fname):
    """Extracts atom coordinates and types from a PDB file."""
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        # Map element to an index, defaulting to -1 for unknown elements.
        types.append(_ELE2NUM.get(atom.element, -1))

    coords = np.stack(coords)
    # Create a one-hot encoding for the atom types.
    types_array = np.zeros((len(types), len(_ELE2NUM)))
    for i, t in enumerate(types):
        if t != -1:  # Skip unknown types
            types_array[i, t] = 1.0

    atom_coords = torch.tensor(coords)
    atom_types = torch.tensor(types_array)

    return atom_coords, atom_types

def save_protein_as_npz(data, atom_coords, atom_types, xyz, normals, curvature, npz_file):
    """Saves all computed protein features into a single compressed .npz file."""
    protein_graph = {
        'x': data.x.cpu().numpy(),
        'edge_index': data.edge_index.cpu().numpy(),
        'seq': data.seq.cpu().numpy(),
        'node_s': data.node_s.cpu().numpy(),
        'node_v': data.node_v.cpu().numpy(),
        'edge_s': data.edge_s.cpu().numpy(),
        'edge_v': data.edge_v.cpu().numpy()
    }
    # Use np.savez to bundle all feature arrays into one file for efficient loading.
    np.savez(npz_file,
             **protein_graph,
             atom_coords=atom_coords.cpu().numpy(),
             atom_types=atom_types.cpu().numpy(),
             xyz=xyz.cpu().numpy(),
             normals=normals.cpu().numpy(),
             curvature=curvature.cpu().numpy())
    logging.debug(f"Saved protein graph to {npz_file}")

def process_pdbs_to_npz(pdb_dir, output_dir, curvature_scales):
    """
    Main processing loop that iterates over PDB files, extracts features,
    and saves them to NPZ format.
    """
    # Find both .pdb and .pdb.gz files.
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb') or f.endswith('.pdb.gz')]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        try:
            # 1. Extract sequence-based graph features (for GVP model).
            # The underlying `atom3d` library can handle .gz files automatically.
            protein_graph = load_protein_graph(pdb_path)
            
            # 2. Extract raw atomic coordinates and types for surface generation.
            atom_coords, atom_types = load_structure(pdb_path)
            
            # 3. Generate the protein surface point cloud and normals (for Point-MAE model).
            xyz, normals, batch = atoms_to_points_normals(atom_coords, batch=torch.zeros(atom_coords.shape[0], dtype=torch.long))
            
            # 4. Calculate geometric curvatures at multiple scales to capture surface details.
            curvature = curvatures(xyz, triangles=None, normals=normals, scales=curvature_scales, batch=batch)

            # 5. Save all extracted features to a compressed NPZ file for efficient training.
            # Correctly handle both .pdb and .pdb.gz extensions for the output filename.
            if pdb_file.endswith('.pdb.gz'):
                output_filename = pdb_file[:-7] + '.npz'
            else:
                output_filename = pdb_file[:-4] + '.npz'
            npz_file = os.path.join(output_dir, output_filename)
            
            save_protein_as_npz(protein_graph, atom_coords, atom_types, xyz, normals, curvature, npz_file)
        except Exception as e:
            # Log errors for individual files without stopping the whole process.
            logging.error(f"Failed to process {pdb_file}: {e}")

if __name__ == '__main__':
    # --- Script Entry Point ---
    # Parses command-line arguments and runs the preprocessing pipeline.
    
    parser = argparse.ArgumentParser(description='Preprocess PDB files to NPZ format.')
    parser.add_argument('--pdb_directory', type=str, required=True, help='Input directory for PDB files.')
    parser.add_argument('--output_directory', type=str, required=True, help='Output directory for NPZ files.')
    parser.add_argument('--curvature_scales', type=float, nargs='+', default=[1.0, 2.0, 3.0, 5.0, 10.0], help='A list of scales for curvature calculation.')
    args = parser.parse_args()

    logging.info(f"Starting preprocessing...")
    logging.info(f"  Input PDB directory: {args.pdb_directory}")
    logging.info(f"  Output NPZ directory: {args.output_directory}")
    logging.info(f"  Curvature scales: {args.curvature_scales}")

    process_pdbs_to_npz(
        pdb_dir=args.pdb_directory,
        output_dir=args.output_directory,
        curvature_scales=args.curvature_scales
    )
    logging.info("Preprocessing finished.")
