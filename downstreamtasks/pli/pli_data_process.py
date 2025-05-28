import os
import warnings
import torch
import openbabel
import numpy as np
from openbabel import pybel
from torch_geometric.data import Data
from tqdm import tqdm
from Bio.PDB import PDBParser
import atom3d.util.formats as fo
from utils.protein_utils import featurize_as_graph
from utils.openbabel_featurizer import Featurizer
from utils.geometry_processing import atoms_to_points_normals, curvatures

warnings.filterwarnings('ignore')
torch.manual_seed(1)
torch.cuda.manual_seed(1)
device = torch.device('cuda:0')
ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}

# Helper functions

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

def read_ligand(filepath):
    featurizer = Featurizer(save_molecule_codes=False)
    ligand = next(pybel.readfile("mol2", filepath))
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

def Ligand_graph(lig_atoms_fea, ligand, h_num, score):
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
    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=torch.tensor(score))

def GetPDBDict(path):
    with open(path, 'r') as f:
        return {line.split()[0]: float(line.split()[3]) for line in f if "//" in line}


def load_structure(fname):
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num.get(atom.element, -1))  # 默认未知元素类型为-1

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        if t != -1:  # 跳过未知类型
            types_array[i, t] = 1.0

    atom_coords = torch.tensor(coords)
    atom_types = torch.tensor(types_array)

    return atom_coords, atom_types

def load_protein_graph(protein_path):
    protein_df = fo.bp_to_df(fo.read_pdb(protein_path))
    protein_graph = featurize_as_graph(protein_df)
    return protein_graph

def save_protein_as_npz(protein, ligand, atom_coords, atom_types, xyz, normals, curvature, npz_file):
    protein_graph = {
        'protein_x': protein.x.cpu().numpy(),
        'protein_edge_index': protein.edge_index.cpu().numpy(),
        'seq': protein.seq.cpu().numpy(),
        'node_s': protein.node_s.cpu().numpy(),
        'node_v': protein.node_v.cpu().numpy(),
        'edge_s': protein.edge_s.cpu().numpy(),
        'edge_v': protein.edge_v.cpu().numpy()
    }
    ligand_graph = {
        'ligand_x': ligand.x.cpu().numpy(),
        'ligand_edge_index': ligand.edge_index.cpu().numpy(),
        'edge_attr': ligand.edge_attr.cpu().numpy(),
        'y': ligand.y.cpu().numpy(),
    }
    np.savez(npz_file, **protein_graph, **ligand_graph,
             atom_coords=atom_coords.cpu().numpy(),
             atom_types=atom_types.cpu().numpy(),
             xyz=xyz.cpu().numpy(),
             normals=normals.cpu().numpy(),
             curvature=curvature.cpu().numpy())
    print(f"Saved protein graph to {npz_file}")

def process_raw_data(dataset_path, output_dir, protein_list):
    res = GetPDBDict('./data/downsteam_data/pdbbind_v2020/')# Protein list generation: Extract valid sample IDs from the PDBbind index file
# Note: The index file path must correspond to the dataset version (e.g., INDEX_refined_data.2020 for the 2020 version)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in tqdm(protein_list):
        score = res.get(item)
        lig_file_name = os.path.join(dataset_path, item, f'{item}_ligand.mol2')
        pocket_file_name = os.path.join(dataset_path, item, f'{item}_pocket.pdb')
        protein_path = os.path.join(dataset_path, item)

        if not os.path.exists(protein_path):
            continue

        lig_coord, lig_atom_fea, mol, h_num_lig, ligand_center = read_ligand(lig_file_name)
        lig_graph = Ligand_graph(lig_atom_fea, mol, h_num_lig, score)
        protein_graph = load_protein_graph(pocket_file_name)
        atom_coords, atom_types = load_structure(pocket_file_name)
        xyz, normals, batch = atoms_to_points_normals(atom_coords, batch=torch.zeros(atom_coords.shape[0], dtype=torch.long))
        curvature = curvatures(xyz, triangles=None, normals=normals, scales=[1.0, 2.0, 3.0, 5.0, 10.0], batch=batch)

        npz_file = os.path.join(output_dir, f'{item}.npz')
        save_protein_as_npz(protein_graph,lig_graph,  atom_coords, atom_types, xyz, normals, curvature, npz_file)
def GetPDBList(path):
    with open(path, 'r') as f:
        return [line.split()[0] for line in f if "//" in line]

if __name__ == '__main__':
    raw_data_path = './data/downsteam_data/pdbbind_v2020/'# Raw data path: PDBbind 2020 refined dataset (contains protein-ligand complex structures)
    protein_list = GetPDBList('./data/downsteam_data/pdbbind_v2020/')# Protein list generation: Extract valid sample IDs from the PDBbind index file
# Note: The index file path must correspond to the dataset version (e.g., INDEX_refined_data.2020 for the 2020 version)
    output_directory = './data/downsteam_data/processed_pli_data/'
    process_raw_data(raw_data_path, output_directory, protein_list)

