import torch
import torch.optim as optim
import re
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import argparse
import sys
sys.path.append('../src/FedTree/GNN/')
import os
from read_tree import read_data
from model import GCN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--gnn_model_path', type=str, default='gnn_model.pt', help='the file path to the saved GNN model')
    parser.add_argument('--tree_model_path', type=str, default='../../../tree.txt', help='the file path to the saved tree models')
    parser.add_argument('--n_classes', type=int, default=256, help='the number of classes (bin ids)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    print("in python training")
    args = get_args()
    graph_datasets = read_data(args.tree_model_path)
    print("graph datasets:", graph_datasets)
    print("dataset[0].y:", graph_datasets[0].y)
    # for graph_data in graph_datasets:
        # T.NormalizeFeatures()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device
    print("n_features:", graph_datasets[0].num_features)
    net = GCN(graph_datasets[0].num_features, args.n_classes).to(device)
    if os.path.isfile(args.gnn_model_path):
        net.load_state_dict(torch.load(args.gnn_model_path))
    net.train()
    # data = data.to(device)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                    weight_decay=args.reg)
    #criterion = F.nll_loss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(args.epochs):
        epoch_loss_collector = []
        for data in graph_datasets:
            data = data.to(device)
            optimizer.zero_grad()

            out = net(data)
            # loss = F.nll_loss(out, data.y)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        # logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    torch.save(net.state_dict(), args.gnn_model_path)




