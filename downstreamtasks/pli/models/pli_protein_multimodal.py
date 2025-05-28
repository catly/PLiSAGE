import torch
import torch.nn as nn
import torch.optim as optim
from stru_encoder.models import GVPModel
from sur_encoder.Point_MAE import Point_MAE
from torch_geometric.nn import AttentiveFP
from torch_geometric.utils import to_dense_batch

class PLIPredictor(nn.Module):
    def __init__(self, config, device):
        super(PLIPredictor, self).__init__()
        self.device = device
        self.protein_struct_encoder = GVPModel(node_in_dim=(6, 3), node_h_dim=(128, 32), edge_in_dim=(32, 1),
                                               edge_h_dim=(32, 1), num_layers=3, drop_rate=0.3)
        self.protein_surface_encoder = Point_MAE(config.model.Point_MAE)
        self.ligandgnn = AttentiveFP(in_channels=18, hidden_channels=64, out_channels=16, edge_dim=12, num_timesteps=3,
                                     num_layers=3, dropout=0.3)
        self.protein_sur_mpl = nn.Linear(48, 128)
        self.sur_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(128, 8, 128*2, batch_first=True,dropout=0.3)
                for _ in range(1)
            ]
        )
        self.struc_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(128, 8, 128*2, batch_first=True,dropout=0.3)
                for _ in range(1)
            ]
        )
        self.out = nn.Sequential(
            nn.Linear(256 + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.ReLU()
        )

    def forward(self, protein_graph, xyz, curvature, dists, atom_type_sel, ligand_graph):
        p_stru = self.protein_struct_encoder(
            (protein_graph.node_s, protein_graph.node_v), protein_graph.edge_index,
            (protein_graph.edge_s, protein_graph.edge_v), protein_graph.batch
        )
        p_sur = self.protein_surface_encoder(xyz, curvature, dists, atom_type_sel)
        p_stru, p_stru_mask = to_dense_batch(p_stru, protein_graph.batch)
        p_sur = self.protein_sur_mpl(p_sur)
        for i, layer in enumerate(self.sur_layers):
            p_sur_new = layer(p_sur)

        for i, layer in enumerate(self.struc_layers):
            p_struc_new = layer(p_stru, p_sur_new, tgt_key_padding_mask=~p_stru_mask)

        p_sur_new = p_sur_new.mean(dim=1)
        p_struc_new = p_struc_new.mean(dim=1)
        protein = torch.cat([p_struc_new, p_sur_new], dim = -1)

        ligand = self.ligandgnn(x=ligand_graph.x, edge_index=ligand_graph.edge_index, edge_attr=ligand_graph.edge_attr,
                                batch=ligand_graph.batch)
        emb = torch.cat((protein, ligand), dim=1)
        y_hat = self.out(emb)
        return torch.squeeze(y_hat)

