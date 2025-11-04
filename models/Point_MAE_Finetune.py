import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from utils.logger import print_log
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils

# Note: This file contains a simplified version of the Point_MAE model, 
# intended for fine-tuning. It only includes the encoder part.

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data

class Atom_embedding_MP(nn.Module):
    def __init__(self, args):
        super(Atom_embedding_MP, self).__init__()
        self.D = args.atom_dims
        self.k = 16
        self.n_layers = 3
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )
        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, dist, atomtypes):
        num_points = dist.shape[1]
        num_dims = atomtypes.shape[-1]

        point_emb = torch.ones_like(atomtypes[:, :, 0, :])
        for i in range(self.n_layers):
            features = torch.cat([point_emb[:, :, None, :].repeat(1, 1, self.k, 1), atomtypes, dist], dim=-1)

            messages = self.mlp[i](features)
            messages = messages.sum(-2)
            point_emb = point_emb + self.relu(self.norm[i](messages.reshape(-1, self.D))).reshape(-1, num_points, num_dims)

        return point_emb

class AtomNet_MP(nn.Module):
    def __init__(self, args):
        super(AtomNet_MP, self).__init__()
        self.args = args
        self.transform_types = nn.Sequential(
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
        )
        self.embed = Atom_embedding_MP(args)

    def forward(self, curvature, dist, atom_type):
        atomtypes = self.transform_types(atom_type)
        atomtypes = self.embed(dist, atomtypes)
        return atomtypes

class Encoder(nn.Module):
    def __init__(self, indim, encoder_channel):
        super().__init__()
        self.indim = indim
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(indim, indim * 2, 1),
            nn.BatchNorm1d(indim * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(indim * 2, indim * 2, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(indim * 4, indim * 2, 1),
            nn.BatchNorm1d(indim * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(indim * 2, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.indim)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, channels = xyz.shape
        center = fps(xyz[:, :, :3].contiguous(), self.num_group)
        _, idx = self.knn(xyz[:, :, :3].contiguous(), center)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, channels).contiguous()
        neighborhood = torch.cat([neighborhood[:, :, :, :3] - center.unsqueeze(2), neighborhood[:, :, :, 3:]], dim=-1)
        return neighborhood, center

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

class PointMAE_Finetune(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointMAE_Finetune] Initializing', logger='PointMAE_Finetune')
        self.config = config
        self.trans_dim = config.emb_dims * config.dimscale
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.depth = config.depth
        self.num_heads = config.num_heads

        self.atomnet = AtomNet_MP(config)
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(indim=config.in_channels, encoder_channel=self.trans_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(nn.Linear(3, self.trans_dim), nn.GELU(),
                                       nn.Linear(self.trans_dim, self.trans_dim))
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=1.0,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

    def forward(self, pts, curvature, dist, atom_type, **kwargs):
        feats_chem = self.atomnet(curvature, dist.unsqueeze(-1), atom_type)
        feats_geo = curvature
        features = torch.cat([feats_geo, feats_chem], dim=-1)

        neighborhood, center = self.group_divider(torch.cat([pts, features], dim=-1))

        group_input_tokens = self.encoder(neighborhood[:, :, :, 3:])

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)
        return x
