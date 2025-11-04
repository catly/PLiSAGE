import torch
import numpy as np
from openbabel import pybel
from torch_geometric.data import Data

from utils.openbabel_featurizer import Featurizer

def info_3D(a, b, c):
    """Calculates angle, area, and distance from 3 points."""
    ab = b - a
    ac = c - a
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac) + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    ab_length = np.linalg.norm(ab)
    ac_length = np.linalg.norm(ac)
    area = 0.5 * ab_length * ac_length * np.sin(angle)
    return np.degrees(angle), area, ac_length

def read_ligand_from_file(filepath):
    """Reads a ligand from a file (e.g., .mol2) and extracts features."""
    featurizer = Featurizer(save_molecule_codes=False)
    ligand = next(pybel.readfile(filepath.split('.')[-1], filepath))
    ligand_coord, atom_fea, h_num = featurizer.get_features(ligand)
    return ligand_coord, atom_fea, ligand, h_num

def read_ligand_from_smiles(smiles_string):
    """Reads a ligand from a SMILES string, generates 3D coords, and extracts features."""
    ligand = pybel.readstring("smi", smiles_string)
    ligand.make3D()
    featurizer = Featurizer(save_molecule_codes=False)
    ligand_coord, atom_fea, h_num = featurizer.get_features(ligand)
    return ligand_coord, atom_fea, ligand, h_num

def bond_features(bond, atom1, atom2):
    """Calculates a feature vector for a chemical bond."""
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

def construct_ligand_graph(lig_atoms_fea, ligand, h_num, label):
    """Constructs a PyG Data object for a ligand molecule."""
    edges, edges_fea = [], []
    for bond in openbabel.OBMolBondIter(ligand.OBMol):
        atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if atom1.GetAtomicNum() == 1 or atom2.GetAtomicNum() == 1:
            continue
        idx_1 = atom1.GetIdx() - h_num[atom1.GetIdx() - 1] - 1
        idx_2 = atom2.GetIdx() - h_num[atom2.GetIdx() - 1] - 1
        edge = [idx_1, idx_2]
        edge_fea = bond_features(bond, atom1, atom2)
        edges.extend([edge, [idx_2, idx_1]])
        edges_fea.extend([edge_fea, edge_fea])

    edge_attr = torch.tensor(edges_fea, dtype=torch.float32)
    x = torch.tensor(lig_atoms_fea, dtype=torch.float32)
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=torch.tensor(label))
