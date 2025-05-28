import os
import warnings
import torch
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
import pandas as pd
from rdkit import Chem
import atom3d.util.formats as fo
from utils.protein_utils import featurize_as_graph
from utils.openbabel_featurizer import Featurizer
from utils.geometry_processing import atoms_to_points_normals, curvatures
from openbabel import pybel
import openbabel
from torch_geometric.data import Data
warnings.filterwarnings('ignore')
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Element to index mapping for atoms
ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def info_3D(a, b, c):
    ab = b - a
    ac = c - a
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = max(cosine_angle, -1.0)
    angle = np.arccos(cosine_angle)
    ab_length = np.linalg.norm(ab)
    ac_length = np.linalg.norm(ac)
    area = 0.5 * ab_length * ac_length * np.sin(angle)
    return np.degrees(angle), area, ac_length


def read_ligand(smiles):

    ligand = pybel.readstring("smi", smiles)
    ligand.make3D()

    featurizer = Featurizer(save_molecule_codes=False)

    ligand_coord, atom_fea, h_num = featurizer.get_features(ligand)

    ligand_center = torch.tensor(ligand_coord).mean(dim=-2, keepdim=True)

    return ligand_coord, atom_fea, ligand, h_num, ligand_center

def bond_fea(bond, atom1, atom2):
    is_aromatic = int(bond.IsAromatic())
    is_in_ring = int(bond.IsInRing())
    dist = atom1.GetDistance(atom2)
    node1_idx, node2_idx = atom1.GetIdx(), atom2.GetIdx()

    neighbours1 = [a for a in openbabel.OBAtomAtomIter(atom1) if a.GetAtomicNum() != 1 and a.GetIdx() != node2_idx]
    neighbours2 = [a for a in openbabel.OBAtomAtomIter(atom2) if a.GetAtomicNum() != 1 and a.GetIdx() != node1_idx]

    if not neighbours1 and not neighbours2:
        return [dist, 0, 0, 0, 0, 0, 0, 0, 0, 0, is_aromatic, is_aromatic]

    angles, areas, distances = [], [], []
    node1_coord = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    node2_coord = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])

    for atom3 in neighbours1:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angle, area, dist3 = info_3D(node1_coord, node2_coord, node3_coord)
        angles.append(angle)
        areas.append(area)
        distances.append(dist3)

    for atom3 in neighbours2:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angle, area, dist3 = info_3D(node2_coord, node1_coord, node3_coord)
        angles.append(angle)
        areas.append(area)
        distances.append(dist3)

    return [dist, max(angles)*0.01, sum(angles)*0.01, np.mean(angles)*0.01, max(areas), sum(areas), np.mean(areas),
            max(distances)*0.1, sum(distances)*0.1, np.mean(distances)*0.1, is_aromatic, is_in_ring]

def edgelist_to_tensor(edge_list):
    edge_index = torch.tensor(list(zip(*edge_list)), dtype=torch.long)
    return edge_index

def Ligand_graph(lig_atoms_fea, ligand, h_num):
    edges, edges_fea = [], []

    for bond in openbabel.OBMolBondIter(ligand.OBMol):
        atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if atom1.GetAtomicNum() == 1 or atom2.GetAtomicNum() == 1:
            continue

        idx_1 = atom1.GetIdx() - h_num[atom1.GetIdx() - 1] - 1
        idx_2 = atom2.GetIdx() - h_num[atom2.GetIdx() - 1] - 1

        edge = [idx_1, idx_2]
        edge_fea = bond_fea(bond, atom1, atom2)
        edges.extend([edge, [idx_2, idx_1]])
        edges_fea.extend([edge_fea, edge_fea])

    edge_attr = torch.tensor(edges_fea, dtype=torch.float32)
    x = torch.tensor(lig_atoms_fea, dtype=torch.float32)
    edge_index = edgelist_to_tensor(edges)
    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
def load_protein_graph(protein_path):
    protein_df = fo.bp_to_df(fo.read_pdb(protein_path))
    protein_graph = featurize_as_graph(protein_df)
    return protein_graph

# Load atom coordinates and atom types from a PDB file
def load_structure(fname):
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())  # Extract atomic coordinates
        types.append(ele2num.get(atom.element, -1))  # Use -1 for unknown elements

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))

    for i, t in enumerate(types):
        if t != -1:  # Skip unknown types
            types_array[i, t] = 1.0

    atom_coords = torch.tensor(coords)
    atom_types = torch.tensor(types_array)

    return atom_coords, atom_types



# Save protein data as an NPZ file
def save_protein_as_npz(protein, atom_coords, atom_types, xyz, normals, curvature, drug, label, npz_file):
    # Prepare protein graph data
    protein_graph = {
        'protein_x': protein.x.cpu().numpy(),
        'protein_edge_index': protein.edge_index.cpu().numpy(),
        'seq': protein.seq.cpu().numpy(),
        'node_s': protein.node_s.cpu().numpy(),
        'node_v': protein.node_v.cpu().numpy(),
        'edge_s': protein.edge_s.cpu().numpy(),
        'edge_v': protein.edge_v.cpu().numpy()
    }
    drug_graph = {
        'ligand_x': drug.x.cpu().numpy(),
        'ligand_edge_index': drug.edge_index.cpu().numpy(),
        'edge_attr': drug.edge_attr.cpu().numpy(),
    }

    # Save all data as an NPZ file, including SMILES and label
    np.savez(npz_file, **protein_graph,**drug_graph,
             atom_coords=atom_coords.cpu().numpy(),
             atom_types=atom_types.cpu().numpy(),
             xyz=xyz.cpu().numpy(),
             normals=normals.cpu().numpy(),
             curvature=curvature.cpu().numpy(),
             label=label)
    print(f"Saved protein graph and additional data to {npz_file}")



# Process raw data and convert it to NPZ format
def process_raw_data(dataset_path, output_dir, csv_file):
    data_df = pd.read_csv(csv_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process files in each folder
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
        id = row['Unnamed: 0']
        smiles = row['SMILES']
        label = row['Label']

        pdb_file_path = os.path.join(dataset_path, f"{id}.pdb")
        if not os.path.exists(pdb_file_path):
            print(f"File not found: {pdb_file_path}. Skipping...")
            continue

            # Load protein graph and atom data
        protein_graph = load_protein_graph(pdb_file_path)
        atom_coords, atom_types = load_structure(pdb_file_path)
        drug_coord, drug_atom_fea, mol, h_num_lig, drug_center = read_ligand(smiles)
        drug_graph = Ligand_graph(drug_atom_fea, mol, h_num_lig)
        label = float(label)

        if atom_coords is not None and atom_types is not None and protein_graph is not None:
                # Compute surface features like normals and curvatures
            xyz, normals, batch = atoms_to_points_normals(atom_coords,
                                                              batch=torch.zeros(atom_coords.shape[0], dtype=torch.long))
            curvature = curvatures(xyz, triangles=None, normals=normals, scales=[1.0, 2.0, 3.0, 5.0, 10.0],
                                       batch=batch)

                # Save processed data as an NPZ file
            npz_file = os.path.join(output_dir, f"{id}.npz")
            save_protein_as_npz(protein_graph, atom_coords, atom_types, xyz, normals, curvature,drug_graph,label, npz_file)
        else:
            print(f"Skipping {id} due to missing data (atom_coords, atom_types, or protein_graph)")


# Main function to start the processing
if __name__ == '__main__':
    raw_data_path = './data/downsteam_data/dti_data/'  # Path to raw PDB files
    output_directory = './data/downsteam_data/dti_data/processed_dti_data/'  # Output path for NPZ files
    csv_file = './data/downsteam_data/dti_data/'
    process_raw_data(raw_data_path, output_directory,csv_file)


