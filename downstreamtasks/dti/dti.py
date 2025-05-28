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
from models.dti_protein_multimodal import PLIPredictor
from utils import data_transforms
from dti_dataset import Protein_search
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from lifelines.utils import concordance_index
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc


# Set seed for reproducibility
seed = 42

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
    protein_graphs, xyzs, curvatures, dists, atom_type_sels,drug_graphs,label = zip(*batch)
    protein_graph_batch = Batch.from_data_list(protein_graphs)
    xyz_batch = torch.stack(xyzs)
    curvature_batch = torch.stack(curvatures)
    dists_batch = torch.stack(dists)
    atom_type_sel_batch = torch.stack(atom_type_sels)
    drug_graphs_batch = Batch.from_data_list(drug_graphs)
    label_batch = torch.stack(label)
    return protein_graph_batch, xyz_batch, curvature_batch, dists_batch, atom_type_sel_batch,drug_graphs_batch,label_batch


train_transforms = transforms.Compose([
    data_transforms.PointcloudScaleAndTranslate()
])


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])





def my_train(train_loader, val_loader, test_loader, kf_filepath, model, config, pretrain_path=None):
    print('Start training')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    max_auc = 0
    num_epochs = 100
    accumulation_steps = 8

    if pretrain_path is not None:
        print(f"Loading pretrained weights from {pretrain_path}")
        pretrain_weights = torch.load(pretrain_path)
        model.load_state_dict(pretrain_weights, strict=False)

    for epoch in range(num_epochs):
        model.train()
        loss_epoch = 0
        count = 0.0
        optimizer.zero_grad()

        for batch_idx, (protein_graph, xyz, curvature, dists, atom_type_sel, drug_graph, label) in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')):
            protein_graph = set_gpu(protein_graph, device)
            xyz = xyz.to(device)
            curvature = curvature.to(device)
            dists = dists.to(device)
            atom_type_sel = atom_type_sel.to(device)
            drug_graph = set_gpu(drug_graph, device)
            label = label.to(device)
            points = train_transforms(xyz)

            out = model(protein_graph, points, curvature, dists, atom_type_sel, drug_graph)
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(out, label)
            loss = loss / accumulation_steps

            loss.backward()
            loss_epoch += loss.item() * accumulation_steps
            count += 1


            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f'Epoch: {epoch}, Batch: {batch_idx + 1}, Loss: {loss.item() * accumulation_steps:.4f}')

            torch.cuda.empty_cache()

        print(f'Epoch: {epoch}, Average Loss: {loss_epoch / count:.4f}')

        auc, auprc, f1, logits, val_loss = my_val(model, val_loader, device, scheduler)
        if auc > max_auc:
            print('********save model*********')
            if not os.path.exists(kf_filepath):
                os.makedirs(kf_filepath)
            torch.save(model.state_dict(), os.path.join(kf_filepath, 'best_model.pt'))
            max_auc = auc

            with open(os.path.join(kf_filepath, "metrics_log.txt"), "a") as f_log:
                str_log = f'epoch: {epoch + 1} val_auc: {auc:.4f} val_auprc: {auprc:.4f} val_F1: {f1:.4f}\n'
                f_log.write(str_log)

            results = my_test(test_loader, kf_filepath, config)
            with open(os.path.join(kf_filepath, "metrics_log.txt"), "a") as f_log:
                str_log = (
                    f'test_auc: {results["roc_auc"]}\n'
                    f'test_auprc: {results["average_precision"]}\n'
                    f'test_F1: {results["f1_score"]}\n'
                    f'test_loss: {results["loss"]}\n'
                    f'confusion_matrix: {results["confusion_matrix"].tolist()}\n'
                    f'accuracy: {results["accuracy"]}\n'
                    f'recall: {results["recall"]}\n'
                    f'precision: {results["precision"]}\n'
                    f'sensitivity: {results["sensitivity"]}\n'
                    f'specificity: {results["specificity"]}\n'
                    f'optimal_threshold: {results["optimal_threshold"]}\n'
                )
                f_log.write(str_log)


def my_val(model, val_loader, device, scheduler):
    y_pred = []
    y_label = []

    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for protein_graph, xyz, curvature, dists, atom_type_sel ,drug_graph, label  in val_loader:
        with torch.no_grad():
            protein_graph = set_gpu(protein_graph, device)
            xyz = xyz.to(device)
            curvature = curvature.to(device)
            dists = dists.to(device)
            atom_type_sel = atom_type_sel.to(device)
            drug_graph = set_gpu(drug_graph,device)
            label = Variable(label).cuda()
            # points = train_transforms(xyz)
            predict = model(protein_graph,xyz, curvature, dists, atom_type_sel,drug_graph)
            loss_fct = torch.nn.BCELoss()

            label = Variable(label).cuda()

            loss = loss_fct(predict, label)
            loss_accumulate += loss
            count += 1

            logits = predict.detach().cpu().numpy()

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            torch.cuda.empty_cache()

    loss = loss_accumulate / count

    scheduler.step(loss)

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
        #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                                  outputs), y_pred, loss.item()


def my_test(test_loader, kf_filepath, config):
    y_pred = []
    y_label = []
    m_state_dict = torch.load(kf_filepath + '/best_model.pt')
    best_model = LBAPredictor(config=config, device=device).to(device)
    best_model.load_state_dict(m_state_dict)
    best_model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for protein_graph, xyz, curvature, dists, atom_type_sel,drug_graph, label in test_loader:
        with torch.no_grad():
            protein_graph = set_gpu(protein_graph, device)
            xyz = xyz.to(device)
            curvature = curvature.to(device)
            dists = dists.to(device)
            atom_type_sel = atom_type_sel.to(device)
            drug_graph = set_gpu(drug_graph,device)
            label = Variable(label).cuda()
            # points = train_transforms(xyz)
            predict = model(protein_graph, xyz, curvature, dists, atom_type_sel, drug_graph)
            loss_fct = torch.nn.BCELoss()

            label = Variable(label).cuda()

            loss = loss_fct(predict, label)
            loss_accumulate += loss
            count += 1

            logits = predict.detach().cpu().numpy()

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            torch.cuda.empty_cache()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return {
        'roc_auc': roc_auc_score(y_label, y_pred),
        'average_precision': average_precision_score(y_label, y_pred),
        'f1_score': f1_score(y_label, outputs),
        'predictions': y_pred,
        'loss': loss.item(),
        'confusion_matrix': cm1,
        'accuracy': accuracy1,
        'recall': recall_score(y_label, y_pred_s),
        'precision': precision_score(y_label, y_pred_s),
        'sensitivity': sensitivity1,
        'specificity': specificity1,
        'optimal_threshold': thred_optim
    }





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein-Ligand Interaction Prediction')
    parser.add_argument('--ckpt', type=str, help='Path to pretrained checkpoint')

    args = parser.parse_args()
    with open("./configs/downsteam_config.yml", 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    train_set = Protein_search(data_root='./data/downsteam_data/processed_data/train', K=16, sample_num=4096,seed=seed)
    val_set = Protein_search(data_root='./data/downsteam_data/processed_data/val', K=16, sample_num=4096,seed=seed)
    test_set = Protein_search(data_root='./data/downsteam_data/processed_data/test', K=16, sample_num=4096,seed=seed)
    batch_size = 16
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, collate_fn=collate_fn)

    model = PLIPredictor(config=config, device=device).to(device)

    filepath = './dti_out/'
    my_train(train_loader,val_loader,test_loader, filepath, model, config, pretrain_path=args.ckpt)

