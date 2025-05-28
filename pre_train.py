import os
import torch
import warnings
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import torch_geometric
import time
from easydict import EasyDict
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from torchvision import transforms
from models.protein_multimodal import PLIPredictor
from utils.ntxent_loss import NTXentLoss
from utils import data_transforms
from protein_dataset import Protein_search


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
    protein_graphs, xyzs, curvatures, dists, atom_type_sels = zip(*batch)
    protein_graph_batch = Batch.from_data_list(protein_graphs)
    xyz_batch = torch.stack(xyzs)
    curvature_batch = torch.stack(curvatures)
    dists_batch = torch.stack(dists)
    atom_type_sel_batch = torch.stack(atom_type_sels)
    return protein_graph_batch, xyz_batch, curvature_batch, dists_batch, atom_type_sel_batch

train_transforms = transforms.Compose([
    data_transforms.PointcloudScaleAndTranslate()
])


def estimate_time(start_time, current_epoch, total_epochs):
    elapsed_time = time.time() - start_time
    avg_epoch_time = elapsed_time / (current_epoch + 1)
    remaining_time = avg_epoch_time * (total_epochs - current_epoch - 1)
    return time.strftime("%H:%M:%S", time.gmtime(remaining_time))


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        torch.save(state, "best_model.pth")



def load_checkpoint(checkpoint_path, model, optimizer=None):
    # Load the checkpoint from the path
    checkpoint = torch.load(checkpoint_path)

    # Load the model state dict with strict=False to allow for parameter mismatches
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if optimizer:
        # Try loading the optimizer state dict
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
            print(f"Warning: Optimizer state could not be loaded due to mismatch: {e}")
            print("Skipping optimizer state loading, continuing with newly initialized optimizer.")

    # Return the epoch from the checkpoint
    return checkpoint.get('epoch', None)  # Safely return the epoch or None if not found


def gradient_stopping_criterion(model, tolerance=1e-6, patience=5):

    global no_grad_change_count
    if not hasattr(gradient_stopping_criterion, 'prev_grads'):
        gradient_stopping_criterion.prev_grads = [param.grad.clone() for param in model.parameters() if
                                                  param.grad is not None]
        no_grad_change_count = 0
        return False

    is_grad_unchanged = True
    for i, param in enumerate(model.parameters()):
        if param.grad is not None:
            grad_change = torch.norm(gradient_stopping_criterion.prev_grads[i] - param.grad)
            if grad_change > tolerance:
                is_grad_unchanged = False
            gradient_stopping_criterion.prev_grads[i] = param.grad.clone()

    if is_grad_unchanged:
        no_grad_change_count += 1
    else:
        no_grad_change_count = 0

    return no_grad_change_count >= patience

def my_train(train_loader, kf_filepath, model, config, resume=False, checkpoint_path=None):
    print('Start training')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    criterion = NTXentLoss(batch_size, temperature=0.05, device=device)
    loss_list = []
    best_loss = float('inf')
    num_epochs = 100
    start_time = time.time()

    start_epoch = 0
    if resume and checkpoint_path is not None:
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)+1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        loss_epoch = 0
        n = 0
        for i, (protein_graph, xyz, curvature, dists, atom_type_sel) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}', total=len(train_loader))):
            protein_graph = set_gpu(protein_graph, device)
            xyz = xyz.to(device)
            curvature = curvature.to(device)
            dists = dists.to(device)
            atom_type_sel = atom_type_sel.to(device)

            points = train_transforms(xyz)
            optimizer.zero_grad()

            p_struc, p_sur,loss1 = model(protein_graph, points, curvature, dists, atom_type_sel)
            loss2 = criterion(p_struc, p_sur)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch += loss.item()
            n += 1
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{n}/{len(train_loader)}], Loss: {loss.item():.4f}")

            torch.cuda.empty_cache()

        avg_loss = loss_epoch / len(train_loader)
        scheduler.step(avg_loss)
        loss_list.append(avg_loss)
        remaining_time = estimate_time(start_time, epoch, num_epochs)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Estimated Time Remaining: {remaining_time}")

        if not os.path.exists(kf_filepath):
            os.makedirs(kf_filepath)

        is_best = avg_loss < best_loss
        best_loss = min(avg_loss, best_loss)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss
        }, is_best, os.path.join(kf_filepath, f'checkpoint_epoch_{epoch+1}_4096.pth.tar'))
if __name__ == '__main__':
    with open("./configs/pretrain_config.yml", 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    dataset = Protein_search(data_root='./processed_pre_data', K=16, sample_num=4048)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size, collate_fn=collate_fn)

    model = PLIPredictor(config=config, device=device).to(device)

    filepath = './pre_chack/'
    my_train(dataloader, filepath, model, config, resume=True, checkpoint_path='./pre_chack/.pth.tar')




