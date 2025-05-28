import os
import torch
import torch.nn.functional as F
import yaml
import torch_geometric
import scipy
import numpy as np
import argparse
from easydict import EasyDict
from sklearn import metrics
from torch.utils.data import DataLoader
from torch_geometric.data import  Batch
from torchvision import transforms
from downstreamtasks.pli.models.pli_protein_multimodal import PLIPredictor
from utils import data_transforms
from downstreamtasks.pli.pli_dataset import Protein_search
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr
from lifelines.utils import concordance_index



# Set seed for reproducibility
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')


def set_gpu(data, device):
    if isinstance(data, tuple):
        return tuple(set_gpu(d, device) for d in data)
    elif isinstance(data, (torch.Tensor, torch_geometric.data.Data)):
        return data.to(device)
    else:
        return data


def collate_fn(batch):
    protein_graphs, xyzs, curvatures, dists, atom_type_sels,ligand_graphs = zip(*batch)
    protein_graph_batch = Batch.from_data_list(protein_graphs)
    ligand_graph_batch = Batch.from_data_list(ligand_graphs)
    xyz_batch = torch.stack(xyzs)
    curvature_batch = torch.stack(curvatures)
    dists_batch = torch.stack(dists)
    atom_type_sel_batch = torch.stack(atom_type_sels)
    return protein_graph_batch, xyz_batch, curvature_batch, dists_batch, atom_type_sel_batch,ligand_graph_batch


train_transforms = transforms.Compose([
    data_transforms.PointcloudScaleAndTranslate()
])


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def metrics_reg(targets,predicts):
    mae = metrics.mean_absolute_error(y_true=targets,y_pred=predicts)
    rmse = metrics.mean_squared_error(y_true=targets,y_pred=predicts,squared=False)
    r = scipy.stats.mstats.pearsonr(targets, predicts)[0]

    x = [ [item] for item in predicts]
    lr = LinearRegression()
    lr.fit(X=x,y=targets)
    y_ = lr.predict(x)
    sd = (((targets - y_) ** 2).sum() / (len(targets) - 1)) ** 0.5

    return [mae,rmse,r,sd]

def get_cindex(Y, P):
    return concordance_index(Y, P)

def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)

    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))
def get_pearson(y,f):
    rp = pearsonr(y,f)[0]
    return rp

def get_spearman(y,f):
    sp = spearmanr(y,f)[0]

    return sp
def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)

    return 1 - (upp / down)

# Prepare for rm2
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def my_train(train_loader, val_loader, test_loader, kf_filepath, model, config, pretrain_path=None):
    print('Start training')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    loss_list = []
    best_mae = float('inf')
    best_rmse = float('inf')
    num_epochs = 100

    if pretrain_path is not None:
        print(f"Loading pretrained weights from {pretrain_path}")
        pretrain_weights = torch.load(pretrain_path)
        model.load_state_dict(pretrain_weights, strict=False)

    for epoch in range(num_epochs):
        model.train()
        loss_epoch = 0
        n = 0
        for protein_graph, xyz, curvature, dists, atom_type_sel ,ligand_garph in train_loader:
            protein_graph = set_gpu(protein_graph, device)
            ligand_garph = set_gpu(ligand_garph, device)
            xyz = xyz.to(device)
            curvature = curvature.to(device)
            dists = dists.to(device)
            atom_type_sel = atom_type_sel.to(device)
            points = train_transforms(xyz)
            optimizer.zero_grad()
            out = model(protein_graph, points, curvature, dists, atom_type_sel,ligand_garph)
            loss = F.mse_loss(out, ligand_garph.y)
            loss_epoch += loss.item()
            print('epoch:', epoch, ' i', n, ' loss:', loss.item())
            loss.backward()
            optimizer.step()
            n += 1

        loss_list.append(loss_epoch / n)
        print('epoch:', epoch, ' loss:', loss_epoch / n)
        val_err = my_val(model, val_loader, device, scheduler)
        val_mae = val_err[0]
        val_rmse = val_err[1]
        if val_rmse < best_rmse and val_mae < best_mae:
            print('********save model*********')
            if not os.path.exists(kf_filepath):
                os.makedirs(kf_filepath)
            torch.save(model.state_dict(), kf_filepath+'best_model.pt')
            best_mae = val_mae
            best_rmse = val_rmse
            f_log = open(file=(kf_filepath+"/metrics_log.txt"), mode="a")
            str_log = 'epoch: '+ str(epoch) + ' val_mae: ' + str(val_mae) + ' val_rmse: ' + str(val_rmse)+ '\n'
            f_log.write(str_log)
            f_log.close()
            my_test(test_loader, kf_filepath, config)
            torch.cuda.empty_cache()


def my_val(model, val_loader, device, scheduler):
    p_affinity = []
    y_affinity = []

    model.eval()
    loss_epoch = 0
    n = 0
    for protein_graph, xyz, curvature, dists, atom_type_sel ,ligand_garph in val_loader:
        with torch.no_grad():
            protein_graph = set_gpu(protein_graph, device)
            ligand_garph = set_gpu(ligand_garph, device)
            xyz = xyz.to(device)
            curvature = curvature.to(device)
            dists = dists.to(device)
            atom_type_sel = atom_type_sel.to(device)
            # points = train_transforms(xyz)
            predict = model(protein_graph, xyz, curvature, dists, atom_type_sel,ligand_garph)
            loss = F.mse_loss(predict, ligand_garph.y)
            loss_epoch += loss.item()
            n += 1
            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(ligand_garph.y.cpu().tolist())

    scheduler.step(loss_epoch / n)
    affinity_err = metrics_reg(targets=y_affinity,predicts=p_affinity)
    return affinity_err


def my_test(test_loader, kf_filepath, config):
    p_affinity = []
    y_affinity = []
    m_state_dict = torch.load(kf_filepath + 'best_model.pt')
    best_model = LBAPredictor(config=config, device=device).to(device)
    best_model.load_state_dict(m_state_dict)
    best_model.eval()

    for i, (protein_graph, xyz, curvature, dists, atom_type_sel, ligand_graph) in enumerate(test_loader, 0):
        with torch.no_grad():
            protein_graph = set_gpu(protein_graph, device)
            ligand_graph = set_gpu(ligand_graph, device)
            xyz = xyz.to(device)
            curvature = curvature.to(device)
            dists = dists.to(device)
            atom_type_sel = atom_type_sel.to(device)
            # points = train_transforms(xyz)
            predict = best_model(protein_graph, xyz, curvature, dists, atom_type_sel, ligand_graph)
            p_affinity.extend(predict.cpu().tolist())
            y_affinity.extend(ligand_graph.y.cpu().tolist())

    affinity_err = metrics_reg(targets=y_affinity, predicts=p_affinity)
    ci = get_cindex(y_affinity, p_affinity)
    rm2 = get_rm2(y_affinity, p_affinity)
    pearson_r = get_pearson(y_affinity, p_affinity)
    spearman_r = get_spearman(y_affinity, p_affinity)
    with open(file=(kf_filepath + "/metrics_log.txt"), mode="a") as f_log:
        str_log = (f'mae: {affinity_err[0]} rmse: {affinity_err[1]} r: {affinity_err[2]} sd: {affinity_err[3]} '
                   f'ci: {ci:.4f} rm2: {rm2:.4f} pearsonr: {pearson_r:.4f} spearmanr: {spearman_r:.4f}\n')
        f_log.write(str_log)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein-Ligand Interaction Prediction')
    parser.add_argument('--ckpt', type=str, help='Path to pretrained checkpoint')

    args = parser.parse_args()

    with open("./configs/downsteam_config.yml", 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    train_set = Protein_search(data_root='./data/downsteam_data/processed_pli_data/train_data', K=16, sample_num=4096)
    val_set = Protein_search(data_root='./data/downsteam_data/processed_pli_data/val_data', K=16, sample_num=4096)
    test_set = Protein_search(data_root='./data/downsteam_data/processed_pli_data/test_data', K=16, sample_num=4096)

    batch_size = 16
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, collate_fn=collate_fn)

    model = PLIPredictor(config=config, device=device).to(device)

    filepath = './pli_out/'
    my_train(train_loader, val_loader, test_loader, filepath, model, config, pretrain_path=args.ckpt)
