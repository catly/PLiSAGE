import torch.nn as nn
from gvp.models import GVPModel
from models.Point_MAE import Point_MAE
class PLIPredictor(nn.Module):
    def __init__(self, config, device):
        super(PLIPredictor, self).__init__()

        self.device = device

        self.protein_struct_encoder = GVPModel(node_in_dim=(6, 3),node_h_dim=(128, 32),edge_in_dim=(32, 1),
                                      edge_h_dim=(32, 1),num_layers=3,drop_rate=0.3)

        self.protein_surface_encoder = Point_MAE(config.model.Point_MAE)
        # self.protein_surface_encoder = dMaSIF(config.model.dmasif)

    def forward(self, protein_graph,xyz, curvature, dists,atom_type_sel):

        p_stru = self.protein_struct_encoder(
            (protein_graph.node_s, protein_graph.node_v),protein_graph.edge_index,
            (protein_graph.edge_s, protein_graph.edge_v),protein_graph.batch)

        # p_struc, p_struc_mask = to_dense_batch(p_stru, protein_graph.batch)
        # print(p_stru.shape)

        loss1,p_sur=self.protein_surface_encoder(xyz,curvature,dists,atom_type_sel)
        # p_sur = iterate(self.protein_surface_encoder, protein)

        # Convert the sparse structure representation to dense format

        return  p_stru,p_sur,loss1
