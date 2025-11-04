import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.ops.knn import knn_points
from torch_geometric.data import Data

class Protein_search(Dataset):
    def __init__(self, data_root, K, sample_num, seed=None):
        """
        Dataset for DTI task. 
        The 'seed' parameter is kept for API consistency but is not used internally 
        to prevent deterministic behavior in multi-worker loading.
        """
        self.root = os.path.join(data_root)
        self.sample_points_num = sample_num
        self.K = K
        
        self.files = [f for f in os.listdir(self.root) if f.endswith('.npz')]
        # Shuffling is now handled by DataLoader

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = os.path.join(self.root, self.files[index])
        data = np.load(file_name)

        protein_graph = Data(
            x=torch.tensor(data['protein_x'], dtype=torch.float),
            edge_index=torch.tensor(data['protein_edge_index'], dtype=torch.long),
            seq=torch.tensor(data['seq'], dtype=torch.float),
            node_s=torch.tensor(data['node_s'], dtype=torch.float),
            node_v=torch.tensor(data['node_v'], dtype=torch.float),
            edge_s=torch.tensor(data['edge_s'], dtype=torch.float),
            edge_v=torch.tensor(data['edge_v'], dtype=torch.float)
        )

        drug_graph = Data(
            x=torch.tensor(data['ligand_x'], dtype=torch.float),
            edge_attr=torch.tensor(data['edge_attr'], dtype=torch.float),
            edge_index=torch.tensor(data['ligand_edge_index'], dtype=torch.long)
        )

        label = torch.tensor(data['y'], dtype=torch.float)
        xyz = data['xyz']
        curvature = data['curvature']
        atom_coords = data['atom_coords']
        atom_types = data['atom_types']

        if len(xyz) > self.sample_points_num:
            idx = np.random.choice(len(xyz), self.sample_points_num, replace=False)
        else:
            idx = np.random.choice(len(xyz), self.sample_points_num, replace=True)
        
        xyz = xyz[idx]
        curvature = curvature[idx]

        xyz = torch.from_numpy(xyz).float()
        curvature = torch.from_numpy(curvature).float()
        atom_coords = torch.from_numpy(atom_coords).float()
        atom_types = torch.from_numpy(atom_types).float()

        dists, idx, _ = knn_points(xyz.unsqueeze(0), atom_coords.unsqueeze(0), K=self.K)
        dists = dists.squeeze(0)
        idx = idx.squeeze(0)
        atom_type_sel = atom_types[idx]

        return protein_graph, xyz, curvature, dists, atom_type_sel, drug_graph, label