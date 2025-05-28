import os
import torch
import numpy as np
from Bio.PDB import PDBParser
import atom3d.util.formats as fo
from utils.protein_utils import featurize_as_graph
from utils.geometry_processing import atoms_to_points_normals,curvatures

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}

def load_protein_graph(protein_path):
    protein_df = fo.bp_to_df(fo.read_pdb(protein_path))
    protein_graph = featurize_as_graph(protein_df)
    return protein_graph

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

def save_protein_as_npz(data, atom_coords, atom_types, xyz, normals, curvature, npz_file):
    protein_graph = {
        'x': data.x.cpu().numpy(),
        'edge_index': data.edge_index.cpu().numpy(),
        'seq': data.seq.cpu().numpy(),
        'node_s': data.node_s.cpu().numpy(),
        'node_v': data.node_v.cpu().numpy(),
        'edge_s': data.edge_s.cpu().numpy(),
        'edge_v': data.edge_v.cpu().numpy()
    }
    # 保存为npz文件
    np.savez(npz_file,
             **protein_graph,
             atom_coords=atom_coords.cpu().numpy(),
             atom_types=atom_types.cpu().numpy(),
             xyz=xyz.cpu().numpy(),
             normals=normals.cpu().numpy(),
             curvature=curvature.cpu().numpy())
    print(f"Saved protein graph to {npz_file}")

def process_pdbs_to_npz(pdb_dir, output_dir):
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        protein_graph = load_protein_graph(pdb_path)
        atom_coords, atom_types = load_structure(pdb_path)
        xyz, normals, batch = atoms_to_points_normals(atom_coords, batch=torch.zeros(atom_coords.shape[0], dtype=torch.long))
        curvature = curvatures(xyz, triangles=None, normals=normals, scales=[1.0, 2.0, 3.0, 5.0, 10.0], batch=batch)

        npz_file = os.path.join(output_dir, pdb_file.replace('.pdb', '.npz'))
        save_protein_as_npz(protein_graph, atom_coords, atom_types, xyz, normals, curvature, npz_file)

if __name__ == '__main__':
    pdb_directory = './alphafold_v2'
    output_directory = './processed_pre_data'
    process_pdbs_to_npz(pdb_directory, output_directory)

