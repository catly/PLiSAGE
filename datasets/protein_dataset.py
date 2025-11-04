import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.ops.knn import knn_points
from torch_geometric.data import Data

class Protein_search(Dataset):
    def __init__(self, data_root, K, sample_num):
        """
        Dataset for loading pre-processed protein data (.npz files).

        Args:
            data_root (str): The root directory where .npz files are stored.
            K (int): The number of nearest atoms to find for each surface point.
            sample_num (int): The number of points to sample from the surface point cloud.
        """
        self.root = os.path.join(data_root)
        self.sample_points_num = sample_num
        self.K = K
        
        # List all .npz files in the data root directory.
        # Shuffling should be handled by the DataLoader, not here.
        self.files = [f for f in os.listdir(self.root) if f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = os.path.join(self.root, self.files[index])
        data = np.load(file_name)

        # Reconstruct the Torch Geometric Data object for the protein graph
        protein_graph = Data(
            x=torch.tensor(data['x'], dtype=torch.float),
            edge_index=torch.tensor(data['edge_index'], dtype=torch.long),
            seq=torch.tensor(data['seq'], dtype=torch.float),
            node_s=torch.tensor(data['node_s'], dtype=torch.float),
            node_v=torch.tensor(data['node_v'], dtype=torch.float),
            edge_s=torch.tensor(data['edge_s'], dtype=torch.float),
            edge_v=torch.tensor(data['edge_v'], dtype=torch.float)
        )

        xyz = data['xyz']
        curvature = data['curvature']
        atom_coords = data['atom_coords']
        atom_types = data['atom_types']

        # Randomly sample points from the surface point cloud.
        # This uses the worker's random state, ensuring different sampling across epochs.
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

        # For each surface point, find the K nearest atoms and their types.
        dists, idx, _ = knn_points(xyz.unsqueeze(0), atom_coords.unsqueeze(0), K=self.K)
        dists = dists.squeeze(0)
        idx = idx.squeeze(0)
        atom_type_sel = atom_types[idx]

        return protein_graph, xyz, curvature, dists, atom_type_sel
