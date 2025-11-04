import torch
import torch.nn as nn
from gvp.models import GVPModel
from models.Point_MAE_Finetune import PointMAE_Finetune
from torch_geometric.nn import AttentiveFP
from torch_geometric.utils import to_dense_batch

class DTIPredictor(nn.Module):
    def __init__(self, config, device):
        super(DTIPredictor, self).__init__()
        self.device = device
        self.config = config

        # Initialize encoders with parameters from the shared config section
        gvp_config = self.config.model.gvp
        self.protein_struct_encoder = GVPModel(
            node_in_dim=tuple(gvp_config.node_in_dim),
            node_h_dim=tuple(gvp_config.node_h_dim),
            edge_in_dim=tuple(gvp_config.edge_in_dim),
            edge_h_dim=tuple(gvp_config.edge_h_dim),
            num_layers=gvp_config.num_layers,
            drop_rate=gvp_config.drop_rate
        )
        self.protein_surface_encoder = PointMAE_Finetune(self.config.model.Point_MAE)

        # --- Task-specific heads ---
        # Config for this specific task is expected under 'dti_training'
        task_config = self.config.dti_training

        # Ligand GNN
        ligand_gnn_config = task_config.ligand_gnn
        self.ligandgnn = AttentiveFP(
            in_channels=ligand_gnn_config.in_channels,
            hidden_channels=ligand_gnn_config.hidden_channels,
            out_channels=ligand_gnn_config.out_channels,
            edge_dim=ligand_gnn_config.edge_dim,
            num_timesteps=ligand_gnn_config.num_timesteps,
            num_layers=ligand_gnn_config.num_layers,
            dropout=ligand_gnn_config.dropout
        )

        # Protein feature fusion
        fusion_config = task_config.fusion
        self.protein_sur_mpl = nn.Linear(fusion_config.sur_in_dim, fusion_config.sur_out_dim)
        self.sur_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=fusion_config.d_model, nhead=fusion_config.nhead, 
                                           dim_feedforward=fusion_config.d_model*2, batch_first=True, 
                                           dropout=fusion_config.dropout)
                for _ in range(fusion_config.num_sur_layers)
            ]
        )
        self.struc_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(d_model=fusion_config.d_model, nhead=fusion_config.nhead, 
                                           dim_feedforward=fusion_config.d_model*2, batch_first=True, 
                                           dropout=fusion_config.dropout)
                for _ in range(fusion_config.num_struc_layers)
            ]
        )
        
        # Final prediction MLP for classification
        self.out = nn.Sequential(
            nn.Linear(fusion_config.d_model * 2 + ligand_gnn_config.out_channels, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, protein_graph, xyz, curvature, dists, atom_type_sel, ligand_graph):
        # Get protein features
        p_stru = self.protein_struct_encoder(
            (protein_graph.node_s, protein_graph.node_v), protein_graph.edge_index,
            (protein_graph.edge_s, protein_graph.edge_v), protein_graph.batch
        )
        p_sur = self.protein_surface_encoder(xyz, curvature, dists, atom_type_sel)
        
        # Fuse protein features
        p_stru, p_stru_mask = to_dense_batch(p_stru, protein_graph.batch)
        p_sur = self.protein_sur_mpl(p_sur)
        
        for layer in self.sur_layers:
            p_sur_new = layer(p_sur)

        for layer in self.struc_layers:
            p_struc_new = layer(p_stru, p_sur_new, tgt_key_padding_mask=~p_stru_mask)

        p_sur_fused = p_sur_new.mean(dim=1)
        p_struc_fused = p_struc_new.mean(dim=1)
        protein_embedding = torch.cat([p_struc_fused, p_sur_fused], dim=-1)

        # Get ligand features
        ligand_embedding = self.ligandgnn(
            x=ligand_graph.x, 
            edge_index=ligand_graph.edge_index, 
            edge_attr=ligand_graph.edge_attr,
            batch=ligand_graph.batch
        )
        
        # Combine and predict
        combined_embedding = torch.cat((protein_embedding, ligand_embedding), dim=1)
        prediction = self.out(combined_embedding)
        return torch.squeeze(prediction)