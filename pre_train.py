import torch
import yaml
import logging
import argparse
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torchvision import transforms

from models.protein_multimodal import PLIPredictor
from utils.ntxent_loss import NTXentLoss
from utils import data_transforms
from datasets.protein_dataset import Protein_search
from utils.trainer import PreTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collate_fn(batch):
    protein_graphs, xyzs, curvatures, dists, atom_type_sels = zip(*batch)
    protein_graph_batch = Batch.from_data_list(protein_graphs)
    xyz_batch = torch.stack(xyzs)
    curvature_batch = torch.stack(curvatures)
    dists_batch = torch.stack(dists)
    atom_type_sel_batch = torch.stack(atom_type_sels)
    return protein_graph_batch, xyz_batch, curvature_batch, dists_batch, atom_type_sel_batch

def main():
    parser = argparse.ArgumentParser(description='Pre-training script for PLiSAGE model.')
    parser.add_argument('--config', type=str, default='configs/pretrain_config.yml', help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    # --- Environment Setup ---
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    device = torch.device(config.device)

    # --- Data Loading ---
    # Note: We are attaching the transforms to the config object for the Trainer to use.
    config.train_transforms = transforms.Compose([
        data_transforms.PointcloudScaleAndTranslate()
    ])
    dataset = Protein_search(
        data_root=config.data.data_root,
        K=config.data.K,
        sample_num=config.data.sample_num
    )
    dataloader = DataLoader(dataset, batch_size=config.data.batch_size, collate_fn=collate_fn, shuffle=True)

    # --- Model, Optimizer, Loss ---
    model = PLIPredictor(config=config, device=device).to(device)
    
    if config.optimizer.name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer.name} not supported.")

    # The scheduler config in YAML is directly unpacked.
    if config.scheduler.name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config.scheduler)
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler.name} not supported.")

    criterion = NTXentLoss(config.data.batch_size, temperature=config.loss.temperature, device=device)

    # --- Instantiate and Run Trainer ---
    trainer = PreTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        train_loader=dataloader,
        config=config
    )
    trainer.train()

if __name__ == '__main__':
    main()