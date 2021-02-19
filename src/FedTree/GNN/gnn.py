import torch
import re
from torch_geometric.data import Data


tree_file = open("tree.txt", "r")

# graphs = []
edges_all_graphs = {}
edges = []
x = []
y = []
nodes_features = {}
nodes_labels = {}
level = {}

for tree_node in tree_file:
    if "Party" or "Tree" in tree_node:
        if edges != []:
            edges_all_graphs.append(edges)
            features = []
            for nid in range(len(nodes_features)):
                features.append(nodes_features[nid])
            x.append(features)
            y.append([nodes_labels[i] for i in range(len(nodes_labels))])

        values = re.findall(r"[-+]?\d*\.\d+|\d+", tree_node)
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
    for feature in features:
        values = re.findall(r"[-+]?\d*\.\d+|\d+", feature)
        value = values[0]
        if feature.find("nid:") == 0:
            nid = value
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
            left_nid = value
        elif feature.find("right_nid:") == 0:
            right_nid = value
    if v == 1:
        if nid not in level.keys():
            level[nid] = 0
            level[left_nid] = 0
            level[right_nid] = 0
        level[left_nid] += 1
        level[right_nid] += 1
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


    # [int(s) for s in str.split() if s.isdigit()]
    # tree_node.partition("nid:")[2]


