#!/usr/bin/python3

# @Time     : 11/7/21
# @Author   : Julie Wang
# @FileName : main.py

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models.trainer import load_from_checkpoint, model_train, model_test, plot_prediction
from torch_geometric.loader import DataLoader
from data_loader.dataloader import TrafficDataset, get_splits, distance_to_weight
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index, to_torch_coo_tensor, dense_to_sparse
)
import torch_geometric
from scipy.sparse import coo_matrix

def main():
    """
    Main function to train and test a model.
    """
    # Constant config to use througout
    config = {
        'BATCH_SIZE': 50,
        'N_PRED': 9,
        'N_HIST': 12,
        'N_DAY_SLOT': 288,
        'N_DAYS': 44,
        'N_NODE': 228,
    }

    gatconfig = {
        'BATCH_SIZE': 50,
        'EPOCHS': 1,
        'WEIGHT_DECAY': 5e-5,
        'INITIAL_LR': 3e-4,
        'DROPOUT': 0.2,
        'USE_GAT_WEIGHTS': True,
        'N_NODE': 228,
        'N_PRED': 9,
        'N_HIST': 12,
        'N_DAY_SLOT': 288,
        'N_DAYS': 44,
        'CHECKPOINT_DIR': './runs',
        'type_model': "GAT"
    }

    tagconfig = {
        'BATCH_SIZE': 50,
        'EPOCHS': 1,
        'k': 3,
        'bias': True,
        'normalize': True,
        'WEIGHT_DECAY': 5e-5,
        'INITIAL_LR': 3e-4,
        'CHECKPOINT_DIR': './runs',
        'N_PRED': 9,
        'N_HIST': 12,
        'DROPOUT': 0.2,
        'N_DAY_SLOT': 288,
        'N_DAYS': 44,
        'N_NODE': 228,
        'type_model': "TAGCN"
    }

    gcnconfig = {
        'BATCH_SIZE': 50,
        'EPOCHS': 20,
        'bias': True,
        'normalize': True,
        'improved': False,
        'cached': False,
        'add_self_loops': True,
        'WEIGHT_DECAY': 5e-5,
        'INITIAL_LR': 3e-4,
        'CHECKPOINT_DIR': './runs',
        'N_PRED': 9,
        'N_HIST': 12,
        'DROPOUT': 0.2,
        'N_DAY_SLOT': 288,
        'N_DAYS': 44,
        'N_NODE': 228,
        'type_model': "GCN"
    }

    # Number of possible windows in a day
    config['N_SLOT'] = config['N_DAY_SLOT'] - (config['N_PRED']  +config['N_HIST']) + 1
    gatconfig['N_SLOT'] = gatconfig['N_DAY_SLOT'] - (gatconfig['N_PRED']  +gatconfig['N_HIST']) + 1
    tagconfig['N_SLOT'] = tagconfig['N_DAY_SLOT'] - (tagconfig['N_PRED']  +tagconfig['N_HIST']) + 1
    gcnconfig['N_SLOT'] = gcnconfig['N_DAY_SLOT'] - (gcnconfig['N_PRED']  +gcnconfig['N_HIST']) + 1

    # Load the weight matrix
    distances = pd.read_csv('./dataset/PeMSD7_W_228.csv', header=None).values
    W = distance_to_weight(distances, gat_version=gatconfig['USE_GAT_WEIGHTS'])
    # Load the dataset
    # np.savetxt('W.csv', W, delimiter=',')
    dist = np.multiply(W, distances)
    # np.savetxt('dist1.csv', dist, delimiter=',')

    norm = np.linalg.norm(dist)
    dist = dist / norm
    dataset = TrafficDataset(config, dist)
    dataset1 = TrafficDataset(config, W)

    # np.savetxt('dist.csv', dist, delimiter=',')
    # np.savetxt('dis.csv', distances, delimiter=',')

    # total of 44 days in the dataset, use 34 for training, 5 for val, 5 for test
    train, val, test = get_splits(dataset, config['N_SLOT'], (34, 5, 5))
    train1, val1, test1 = get_splits(dataset1, config['N_SLOT'], (34, 5, 5))

    train_dataloader = DataLoader(train, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_dataloader = DataLoader(val, batch_size=config['BATCH_SIZE'], shuffle=True)
    test_dataloader = DataLoader(test, batch_size=config['BATCH_SIZE'], shuffle=False)

    train_dataloader1 = DataLoader(train1, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_dataloader1 = DataLoader(val1, batch_size=config['BATCH_SIZE'], shuffle=True)
    test_dataloader1 = DataLoader(test1, batch_size=config['BATCH_SIZE'], shuffle=False)
    # Get gpu if you can
    device = 'cpu'
    print(f"Using {device}")
    # # dist = TrafficDataset(config, dist)
    # # dist = coo_matrix(dist)
    # # print(type(dist))
    # dist = torch.tensor(dist)
    #
    # print(dist)
    #
    # edge, att = dense_to_sparse(dist)
    #
    # print(edge)
    # # print(is_torch_sparse_tensor(edge))
    #
    # edge = to_torch_coo_tensor(edge)
    #
    # print(edge)
    #
    # edge_index, _ = to_edge_index(edge)
    #
    # print(edge_index)
    # print(att)

    # Configure and train model
    config['N_NODE'] = dataset.n_node
    # gatmodel = model_train(train_dataloader, val_dataloader, gatconfig, device)
    # tagmodel = model_train(train_dataloader, val_dataloader, tagconfig, device)
    # gcnmodel = model_train(train_dataloader, val_dataloader, gcnconfig, device)
    # gcnmodel1 = model_train(train_dataloader1, val_dataloader1, gcnconfig, device)

    # Or, load from a saved checkpoint
    # gatmodel = load_from_checkpoint('./runs/model_04-08-145001_GAT_epoch_1.pt', gatconfig)
    # tagmodel = load_from_checkpoint('./runs/model_04-08-144653_TAGCN_epoch_1.pt', tagconfig)
    # gcnmodel = load_from_checkpoint('./runs/model_06-18-155255_GCN_epoch_20.pt', gcnconfig)
    gcnmodel1 = load_from_checkpoint('./runs/model_06-18-160640_GCN_epoch_20.pt', gcnconfig)
    gcnmodel2 = load_from_checkpoint('./runs/model_06-19-111607_GCN_epoch_20.pt', gcnconfig)

    # Test Model
    # gat_pred, y_truth, t = model_test(gatmodel, test_dataloader, device, gatconfig)
    # tag_pred, y_truth, t = model_test(tagmodel, test_dataloader, device, tagconfig)
    # gcn_pred, y_truth, t = model_test(gcnmodel, test_dataloader, device, gcnconfig)
    gcn_pred1, y_truth, t = model_test(gcnmodel1, test_dataloader1, device, gcnconfig)
    gcn_pred2, y_truth, t = model_test(gcnmodel2, test_dataloader1, device, gcnconfig)


    # plt.plot(t, gat_pred, label="GAT")
    # plt.plot(t, tag_pred, label="TAGCN")
    # plt.plot(t, gcn_pred, label="GCNseifi")
    plt.plot(t, gcn_pred1, label="GCN")
    plt.plot(t, gcn_pred2, label="GCNseifi normalized")

    plt.plot(t, y_truth, label='truth')
    plt.title('Predictions of traffic over time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
