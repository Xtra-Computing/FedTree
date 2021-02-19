import torch
import re
from torch_geometric.data import Data


tree_file = open("tree.txt", "r")

graphs = []
edges_all_graphs = []
edges = []
x = []
level = {}
for tree_node in tree_file:
    if "Party" or "Tree" in tree_node:
        edges = []
        x = []
        level = {}
        continue
    #     use comma to partition the string. then process each string.
    features = tree_node.split(',')
    for feature in features:
        values = re.findall(r"[-+]?\d*\.\d+|\d+", feature)
        value = values[0]
        if "nid:" in feature:
            nid = value
        elif "l:" in feature:
            l = value
        elif "v:" in feature:
            v = value
        elif "p:" in feature:
            p = value
        elif "lch:" in feature:
            lch = value
        elif "rch:" in feature:
            rch = value
        elif "split_feature_id:" in feature:
            sp_f_id = value
        elif "f:" in feature:
            f = value
        elif "split_bin_id:" in feature:
            sp_bin_id = value
        elif "gain:" in feature:
            gain = value
        elif "r:" in feature:
            r = value
        elif "w:" in feature:
            w = value
        elif "g/h:" in feature:
            g = values[0]
            h = values[1]
    if v == 1:
        if nid not in level.keys():
            level[nid] = 0
            level[lch] = 0
            level[rch] = 0
        level[lch] += 1
        level[rch] += 1
        edges.append([nid, lch])
        edges.append([nid, rch])
    if (l == 1) or (v == 0):
        for edge in edges:
            if nid in edge:
                edges.remove(edge)
    node_feature = [sp_f_id, gain, r, w, g, h, level]
    node_label = [sp_bin_id]


    # [int(s) for s in str.split() if s.isdigit()]
    # tree_node.partition("nid:")[2]


