import torch
import re
from bisect import bisect
from torch_geometric.data import Data


def read_data(file_path):

    tree_file = open(file_path, "r")

    # graphs = []
    edges_all_graphs = []
    edges = []
    x = []
    y = []
    nodes_features = {}
    nodes_labels = {}
    level = {}
    max_y = 0
    #n_remove = 0
    leaf_nodes = []
    leaf_nodes_graphs = []
    for tree_node in tree_file:
        if tree_node == "" or tree_node == "\n":
            continue
        if ("Party" in tree_node) or ("Tree" in tree_node):
            # n_remove = 0
            if edges != []:
                edges_all_graphs.append(edges)
                features = []
                for nid in nodes_features.keys():
                    features.append(nodes_features[nid])
                x.append(features)
                y.append([nodes_labels[i] for i in nodes_labels.keys()])

            values = re.findall(r"[-+]?\d*\.\d+|\d+", tree_node)
            #print("values:", values)
            if "Party" in tree_node:
                pid = values[0]
            else:
                tid = values[0]
            if leaf_nodes != []:
                leaf_nodes_graphs.append(leaf_nodes)
            edges = []
            level = {}
            nodes_features = {}
            nodes_labels = {}
            leaf_nodes = []
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
                #nid -= n_remove
                # print("nid:", nid)
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
                g = float(values[0])
                h = float(values[1])
            elif feature.find("left_nid:") == 0:
                left_nid = int(value)
                #left_nid -= n_remove
                # print("left_nid:", left_nid)
            elif feature.find("right_nid:") == 0:
                right_nid = int(value)
                #right_nid -= n_remove
                # print("right_nid:", right_nid)
        # if nid == 12:
        #     print("nid 12 l:", l)
        if (v == 1) and (l != 1):
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
            leaf_nodes.append(nid)
            #n_remove += 1
            # print("nid:", nid)
            # print("n_remove:", n_remove)
            for edge in edges:
                if nid in edge:
                    edges.remove(edge)
        if (v == 1) and (l != 1):
            node_feature = [sp_f_id, gain, r, w, g, h, level[nid]]
            nodes_features[nid] = node_feature
            nodes_labels[nid] = int(sp_bin_id)
        if sp_bin_id > max_y:
            max_y = int(sp_bin_id)
    leaf_nodes_graphs.append(leaf_nodes)
    # print("leaf nodes graphs:", leaf_nodes_graphs)

    edges_all_graphs.append(edges)
    print("edges before reorder:", edges_all_graphs)
    for i in range(len(edges_all_graphs)):
        edges = edges_all_graphs[i]
        leaf_nodes = leaf_nodes_graphs[i]
        # print("leaf_nodes:", leaf_nodes)
        for edge in edges:
            for idx in range(len(edge)):
                nid = edge[idx]
                # print("edge nid:", nid)
                pos = bisect(leaf_nodes, nid)
                # print("pos:", pos)
                edge[idx] -= pos
    print("edges after reorder:", edges_all_graphs)

    features = []
    for nid in nodes_features.keys():
        features.append(nodes_features[nid])

    x.append(features)
    y.append([nodes_labels[i] for i in nodes_labels.keys()])

    #print("edges_all_graphs:", edges_all_graphs)
    #print("x:", x)
    #print("y:", y)
    # print("len edges_all_graphs:", len(edges_all_graphs))
    # print("len x:", len(x))
    # print("len y:", len(y))
    print("max_y:", max_y)
    #print("x:", x)
    graph_datasets = []
    for graph_id in range(len(edges_all_graphs)):
        edge_index = torch.tensor(edges_all_graphs[graph_id], dtype=torch.long).t().contiguous()
        #print("x[graph_id]:", x[graph_id])
        x_tensor = torch.tensor(x[graph_id], dtype=torch.float)
        y_tensor = torch.tensor(y[graph_id], dtype=torch.long)
        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
        #data.num_classes = max_y+1
        graph_datasets.append(data)
        #print("data:", data)
    # print("graph datasets 0:", graph_datasets[0])
    # print("graph datasets 1:", graph_datasets[1])
    return graph_datasets
