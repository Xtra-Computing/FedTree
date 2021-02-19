import torch
import re
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.nn.functional as F
import argparse
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
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')

    args = parser.parse_args()
    return args


tree_file = open("tree.txt", "r")

# graphs = []
edges_all_graphs = []
edges = []
x = []
y = []
nodes_features = {}
nodes_labels = {}
level = {}

for tree_node in tree_file:
    if tree_node == "" or tree_node == "\n":
        continue
    if ("Party" in tree_node) or ("Tree" in tree_node):
        if edges != []:
            edges_all_graphs.append(edges)
            features = []
            for nid in range(len(nodes_features)):
                features.append(nodes_features[nid])
            x.append(features)
            y.append([nodes_labels[i] for i in range(len(nodes_labels))])

        values = re.findall(r"[-+]?\d*\.\d+|\d+", tree_node)
        #print("values:", values)
        if "Party" in tree_node:
            pid = values[0]
        else:
            tid = values[0]
        edges = []
        level = {}
        nodes_features = {}
        nodes_labels = {}
        continue
    #     use comma to partition the string. then process each string.
    features = tree_node.split(',')
    #print("features:", features)
    for feature in features:
        #print("feature:", feature)
        values = re.findall(r"[-+]?\d*\.\d+|\d+", feature)
        #print("values:", values)
        value = float(values[0])
        if feature.find("nid:") == 0:
            nid = int(value)
        elif feature.find("l:") == 0:
            l = value
        elif feature.find("v:") == 0:
            v = value
        elif feature.find("p:") == 0:
            p = value
        elif feature.find("lch:") == 0:
            lch = value
        elif feature.find("rch:") == 0:
            rch = value
        elif feature.find("split_feature_id:") == 0:
            sp_f_id = value
        elif feature.find("f:") == 0:
            f = value
        elif feature.find("split_bin_id:") == 0:
            sp_bin_id = value
        elif feature.find("gain:") == 0:
            gain = value
        elif feature.find("r:") == 0:
            r = value
        elif feature.find("w:") == 0:
            w = value
        elif feature.find("g/h:") == 0:
            g = values[0]
            h = values[1]
        elif feature.find("left_nid:") == 0:
            left_nid = int(value)
        elif feature.find("right_nid:") == 0:
            right_nid = int(value)
    if v == 1:
        if nid not in level.keys():
            level[nid] = 0
            level[left_nid] = 1
            level[right_nid] = 1
        if left_nid not in level.keys():
            level[left_nid] = level[nid] + 1
        if right_nid not in level.keys():
            level[right_nid] = level[nid] + 1
        #level[left_nid] += 1
        #level[right_nid] += 1
        edges.append([nid, left_nid])
        edges.append([nid, right_nid])
    if (l == 1) or (v == 0):
        for edge in edges:
            if nid in edge:
                edges.remove(edge)
    node_feature = [sp_f_id, gain, r, w, g, h, level]
    nodes_features[nid] = node_feature
    nodes_labels[nid] = sp_bin_id

edges_all_graphs.append(edges)
features = []
for nid in range(len(nodes_features)):
    features.append(nodes_features[nid])
x.append(features)
y.append([nodes_labels[i] for i in range(len(nodes_labels))])

#print("edges_all_graphs:", edges_all_graphs)
#print("x:", x)
#print("y:", y)
print("len edges_all_graphs:", len(edges_all_graphs))
print("len x:", len(x))
print("len y:", len(y))

graph_datasets=[]
for graph_id in range(len(edges_all_graphs)):
    edge_index = torch.tensor(edges_all_graphs[graph_id], dtype=torch.long).t().contiguous()
    x = torch.tensor(x[graph_id], dtype=torch.float)
    y = torch.tensor(y[graph_id], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    graph_datasets.append(data)

# for graph_data in graph_datasets:
    # T.NormalizeFeatures()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = args.device
net = GCN.to(device)
# data = data.to(device)
if args.optimizer == 'sgd':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                weight_decay=args.reg)
criterion = F.nll_loss().to(device)

for epoch in range(args.epochs):
    epoch_loss_collector = []
    for data in graph_datasets:
        data = data.to(device)

        optimizer.zero_grad()

        out = net(data)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()
        epoch_loss_collector.append(loss.item())

    epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
    logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))





