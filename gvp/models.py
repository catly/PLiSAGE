import torch.nn as nn
from . import GVP, GVPConvLayer, LayerNorm
import torch_geometric
import torch
from torch_scatter import scatter_mean

class GVPModel(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim, num_layers=3, drop_rate=0.1):
        
        super().__init__()
        self.output_dim = node_h_dim[0]
        self.rbf_dim = edge_in_dim[0]

        self.residue_embdding = nn.Linear(node_in_dim[0], node_in_dim[0], bias=False)
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))
    #
    # def forward(self, h_V, edge_index, h_E, batch=None):
    #
    #     h_V = self.W_v(h_V)
    #     h_E = self.W_e(h_E)
    #
    #     for layer in self.layers:
    #         h_V = layer(h_V, edge_index, h_E)
    #     out = self.W_out(h_V)
    #
    #     return out
            
        
    def forward(self, h_V, edge_index, h_E, batch):

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        node_feature = self.W_out(h_V)
        graph_feature = torch_geometric.nn.global_mean_pool(node_feature, batch)

        return graph_feature


