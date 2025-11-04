import torch.nn as nn
from gvp.models import GVPModel
from models.Point_MAE import Point_MAE

class PLiSAGE_Encoder(nn.Module):
    """ 
    The main PLiSAGE model for pre-training. 
    It combines a structure-based encoder (GVP) and a surface-based encoder (Point_MAE).
    """
    def __init__(self, config, device):
        super(PLiSAGE_Encoder, self).__init__()
        self.config = config
        self.device = device

        # Initialize the structure encoder with parameters from the config file
        gvp_config = self.config.model.gvp
        self.protein_struct_encoder = GVPModel(
            node_in_dim=tuple(gvp_config.node_in_dim),
            node_h_dim=tuple(gvp_config.node_h_dim),
            edge_in_dim=tuple(gvp_config.edge_in_dim),
            edge_h_dim=tuple(gvp_config.edge_h_dim),
            num_layers=gvp_config.num_layers,
            drop_rate=gvp_config.drop_rate
        )

        # Initialize the surface encoder
        self.protein_surface_encoder = Point_MAE(config.model.Point_MAE)

    def forward(self, protein_graph, xyz, curvature, dists, atom_type_sel):
        """
        Forward pass of the model.
        Returns embeddings from both modalities and data for reconstruction loss.
        """
        # Get the structure embedding from the GVP model
        structure_embedding = self.protein_struct_encoder(
            (protein_graph.node_s, protein_graph.node_v),
            protein_graph.edge_index,
            (protein_graph.edge_s, protein_graph.edge_v),
            protein_graph.batch
        )

        # Get the surface embedding and data for reconstruction from the Point_MAE model
        reconstructed_points, ground_truth_points, surface_embedding = self.protein_surface_encoder(
            xyz, curvature, dists, atom_type_sel
        )

        return structure_embedding, surface_embedding, reconstructed_points, ground_truth_points