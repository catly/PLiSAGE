import torch
import yaml
import argparse
import logging
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torchvision import transforms

from downstreamtasks.pli.models.pli_protein_multimodal import PLIPredictor
from utils import data_transforms
from downstreamtasks.pli.pli_dataset import Protein_search
from utils.trainer import RegressionTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collate_fn(batch):
    protein_graphs, xyzs, curvatures, dists, atom_type_sels, ligand_graphs = zip(*batch)
    protein_graph_batch = Batch.from_data_list(protein_graphs)
    ligand_graph_batch = Batch.from_data_list(ligand_graphs)
    xyz_batch = torch.stack(xyzs)
    curvature_batch = torch.stack(curvatures)
    dists_batch = torch.stack(dists)
    atom_type_sel_batch = torch.stack(atom_type_sels)
    return protein_graph_batch, xyz_batch, curvature_batch, dists_batch, atom_type_sel_batch, ligand_graph_batch

def main():
    parser = argparse.ArgumentParser(description='Protein-Ligand Interaction Prediction Training.')
    parser.add_argument('--config', type=str, default='configs/downsteam_config.yml', help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    # Select the config section for this specific task
    cfg = config.pli_training

    # --- Environment and Data Setup ---
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    cfg.train_transforms = transforms.Compose([
        data_transforms.PointcloudScaleAndTranslate()
    ])

    logging.info("Loading datasets...")
    train_set = Protein_search(data_root=cfg.data.train_dir, K=cfg.data.K, sample_num=cfg.data.sample_num)
    val_set = Protein_search(data_root=cfg.data.val_dir, K=cfg.data.K, sample_num=cfg.data.sample_num)
    test_set = Protein_search(data_root=cfg.data.test_dir, K=cfg.data.K, sample_num=cfg.data.sample_num)

    train_loader = DataLoader(dataset=train_set, batch_size=cfg.data.batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=cfg.data.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_set, batch_size=cfg.data.batch_size, collate_fn=collate_fn)

    # --- Model, Optimizer, Scheduler Setup ---
    logging.info("Setting up model and optimizer...")
    # The main config object is passed to the model to access shared model parameters
    model = PLIPredictor(config=config, device=device).to(device)

    if cfg.training.pretrained_ckpt_path:
        logging.info(f"Loading pretrained weights from {cfg.training.pretrained_ckpt_path}")
        pretrain_weights = torch.load(cfg.training.pretrained_ckpt_path, map_location=device)
        model.load_state_dict(pretrain_weights.get('state_dict', pretrain_weights), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)

    # --- Instantiate and Run Trainer ---
    trainer = RegressionTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg
    )
    trainer.train()

if __name__ == '__main__':
    main()
