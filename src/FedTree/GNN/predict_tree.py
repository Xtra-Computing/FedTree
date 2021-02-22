import torch
from model import GCN
from read_tree import read_data
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--gnn_model_path', type=str, default='model', help='the file path to the saved GNN model')
    parser.add_argument('--tree_model_path', type=str, default='../../../tree.txt', help='the file path to the saved tree models')
    parser.add_argument('--label_file_path', type=str, default='../../../predict_labels.txt', help='the file path to save predicted labels')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()

    net = GCN()
    net.load_state_dict(torch.load(gnn_model_path))

    graph_datasets = read_data(tree_model_path)

    device = args.device
    net = GCN().to(device)
    # data = data.to(device)

    was_training = False
    if net.training:
        net.eval()
        was_training = True

    pred_labels = []

    with torch.no_grad():
        for data in graph_datasets:
            # print("x:",x)
            # print("target:",target)

            pred_label = net(data).max(dim=1)[1]
            pred_labels.append(pred_label)


    # write predict labels to file.
    f = open(args.label_file_path, "a")
    for labels in pred_labels:
        for label in labels:
            f.write(str(label))
    f.close()


    if was_training:
        net.train()
