import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.ops.knn import knn_points
from torch_geometric.data import Data,Batch

class Protein_search(Dataset):
    def __init__(self, data_root, K, sample_num, seed):
        # dataset parameters

        self.root = os.path.join(data_root)
        self.sample_points_num = sample_num
        self.K = K
        for _, _, files in os.walk(self.root):
            break
        np.random.seed(seed)
        np.random.shuffle(files)
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = os.path.join(self.root, self.files[index])
        data = np.load(file_name)

        protein_x = torch.tensor(data['protein_x'], dtype=torch.float)
        protein_edge_index = torch.tensor(data['protein_edge_index'], dtype=torch.long)
        seq = torch.tensor(data['seq'], dtype=torch.float)
        node_s = torch.tensor(data['node_s'], dtype=torch.float)
        node_v = torch.tensor(data['node_v'], dtype=torch.float)
        edge_s = torch.tensor(data['edge_s'], dtype=torch.float)
        edge_v = torch.tensor(data['edge_v'], dtype=torch.float)
        protein_graph = Data(x=protein_x,edge_index=protein_edge_index,seq=seq,node_s=node_s, node_v=node_v, edge_s=edge_s,edge_v=edge_v)

        drug_x = torch.tensor(data['ligand_x'], dtype=torch.float)
        drug_edge_index = torch.tensor(data['ligand_edge_index'], dtype=torch.long)
        drug_attr = torch.tensor(data['edge_attr'], dtype=torch.float)
        drug_graph = Data(x=drug_x, edge_attr=drug_attr, edge_index=drug_edge_index)

        label = torch.tensor(data['label'],dtype=torch.float)
        xyz = data['xyz']
        normals = data['normals']
        curvature = data['curvature']
        atom_coords= data['atom_coords']
        atom_types = data['atom_types']

        if len(xyz) > self.sample_points_num:
            idx = np.random.choice(len(xyz), self.sample_points_num, replace=False)
            xyz = xyz[idx]
            normals = normals[idx]
            curvature = curvature[idx]
        else:
            idx = np.random.choice(len(xyz), self.sample_points_num, replace=True)
            xyz = xyz[idx]
            normals = normals[idx]
            curvature = curvature[idx]

        xyz = torch.from_numpy(xyz).float()
        normals = torch.from_numpy(normals).float()
        curvature = torch.from_numpy(curvature).float()
        atom_coords = torch.from_numpy(atom_coords).float()
        atom_types = torch.from_numpy(atom_types).float()

        dists, idx, _ = knn_points(xyz.unsqueeze(0), atom_coords.unsqueeze(0), K=self.K)
        dists = dists.squeeze(0)
        idx = idx.squeeze(0)
        atom_type_sel = atom_types[idx]

        return  protein_graph,xyz, curvature, dists, atom_type_sel,drug_graph,label



