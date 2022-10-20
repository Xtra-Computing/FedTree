import argparse
import numpy as np
import copy
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="File to partition")
    parser.add_argument("-p", "--n_parties", type=int, help="Number of partitions")
    parser.add_argument("-m", "--mode", type=str, help="horizontal or vertical")
    parser.add_argument("--iid", type=int, default=0, help="IID partition or not")
    parser.add_argument("--beta", type=float, default=0.5, help="parameter for Dir distribution")
    parser.add_argument("--output", type=str, help="output file path")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--test_file", type=int, default=0, help="with test file or not")
    parser.add_argument("--fate", type=int, default=1, help="generate FATE format or not")
    args = parser.parse_args()
    return args


def hori_partition(args):
    total_features = 0
    size = 0
    label_set = set()
    with open(args.file, 'r') as fin:
        line = fin.readline()
        while line:
            line = line[:-1]
            if line and line[-1] == "\r":
                line = line[:-1]
            line = line.rstrip(' ')
            ele = line.split(' ')
            label_set.add(int(ele[0]))
            total_features = max(total_features, int(ele[-1].split(':')[0]))
            size += 1
            line = fin.readline()
    label_dict = {}
    if len(label_set) == 2: # classification
        tmp_label_list = sorted(label_set)
        label_dict = {str(item): str(i) for i, item in enumerate(tmp_label_list)}
    else:   # regression
        label_dict = {str(item): str(item) for item in label_set}
    np.random.seed(args.seed)
    if args.iid:
        idxs = np.random.permutation(size)
        batch_idxs = np.array_split(idxs, args.n_parties)
        for i in range(len(batch_idxs)):
            batch_idxs[i] = np.sort(batch_idxs[i]).tolist()
    else:
        min_req_size = 100
        current_min_size = 0
        while current_min_size < min_req_size:
            idxs = np.random.permutation(size)
            proportions = np.random.dirichlet(np.repeat(args.beta, args.n_parties))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            batch_idxs = np.split(idxs, proportions)
            for i in range(len(batch_idxs)):
                batch_idxs[i] = np.sort(batch_idxs[i]).tolist()
            current_min_size = min([len(item) for item in batch_idxs])

    line2pid = []
    for i in range(size):
        for j in range(args.n_parties):
            if i in batch_idxs[j]:
                line2pid.append(j)
                break

    with open(args.file, 'r') as fin:
        fouts = [open(args.output+args.dataset+"_h_{}_{}".format(args.n_parties, i), 'w') for i in range(args.n_parties)]
        line = fin.readline()
        glob_idx = 0
        while line:
            line = line[:-1]
            if line and line[-1] == "\r":
                line = line[:-1]
            line = line.rstrip(' ')
            fouts[line2pid[glob_idx]].write(line + "\n")
                    # p_off[i] += 1
            glob_idx += 1
            line = fin.readline()
        for item in fouts:
            item.close()
    if args.fate:
        with open(args.file, 'r') as fin:
            fouts = [open(args.output + args.dataset + "_FATE_h_{}_{}".format(args.n_parties, i), 'w') for i in
                     range(args.n_parties)]
            glob_idx = 0
            line = fin.readline()
            while line:
                line = line[:-1]
                if line and line[-1] == "\r":
                    line = line[:-1]
                line = line.rstrip(' ')
                ele = line.split(' ')
                f_ele = [str(glob_idx), label_dict[str(int(ele[0]))]]
                f_ele += ["{}:{}".format(-1 + int(item.split(':')[0]), item.split(':')[1]) for item in
                          ele[1:]]

                fouts[line2pid[glob_idx]].write(','.join(f_ele) + '\n')

                glob_idx += 1
                line = fin.readline()
            for item in fouts:
                item.close()

def verti_partition(args):
    np.random.seed(args.seed)
    num_features = 0
    with open(args.file, 'r') as f:
        line = f.readline()
        while line :
            line = line[:-1]
            if line and line[-1] == "\r":
                line = line[:-1]
            line = line.rstrip(' ')
            l = line.split(' ')
            num_features = max(num_features, int(l[-1].split(':')[0]))
            line = f.readline()

    print(num_features)
    # features = [i+1 for i in range(num_features)]
    # random.shuffle(features)
    # BLOCK = num_features // args.num_parties
    # offset = [i*BLOCK for i in range(args.num_parties)]+[num_features]

    # id2party = {}
    # id2local = {}
    # for i in range(args.num_parties):
    #     features[offset[i]: offset[i+1]] = sorted(features[offset[i]: offset[i+1]])
    #     for j in range(offset[i], offset[i+1]):
    #         id2party[features[j]] = i
    #         id2local[features[j]] = j-offset[i]
    # print(features)

    if args.iid:
        idxs = np.random.permutation([ i+1 for i in range(num_features)])
        batch_idxs = np.array_split(idxs, args.n_parties)
        for i in range(len(batch_idxs)):
            batch_idxs[i] = np.sort(batch_idxs[i]).tolist()
    else:
        min_req_size = 1
        current_min_size = 0
        while current_min_size < min_req_size:
            idxs = np.random.permutation([ i+1 for i in range(num_features)])
            proportions = np.random.dirichlet(np.repeat(args.beta, args.n_parties))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            batch_idxs = np.split(idxs, proportions)
            for i in range(len(batch_idxs)):
                batch_idxs[i] = np.sort(batch_idxs[i]).tolist()
            current_min_size = min([len(item) for item in batch_idxs])

    id2party = {}
    id2local = {}
    offset = [0 for _ in range(args.n_parties)]
    for i in range(1, num_features+1):
        for j in range(args.n_parties):
            if i in batch_idxs[j]:
                id2party[i] = j
                id2local[i] = offset[j]
                offset[j] += 1
                break
    with open(args.file, 'r') as f:
        outfs = [open(args.output+args.dataset+"_v_"+str(args.n_parties)+"_"+str(i), 'w') for i in range(args.n_parties)]
        line = f.readline()

        while line:
            line = line[:-1]
            if line and line[-1] == "\r":
                line = line[:-1]
            line = line.rstrip(' ')
            l = line.split(' ')
            outls = [[] for i in range(args.n_parties)]
            for i in range(args.n_parties):
                outls[i].append(l[0])

            for item in l[1:]:
                k, v = item.split(":")
                outls[id2party[int(k)]].append("{}:{}".format(id2local[int(k)]+1, v))

            for i in range(args.n_parties):
                outfs[i].write(" ".join(outls[i])+'\n')

            line = f.readline()

        for item in outfs:
            item.close()


    if args.test_file:
        with open(args.test_file, 'r') as f:
            with open(args.output+args.dataset+"_v_"+str(args.n_parties)+"_test", 'w') as out_tfs:
                line = f.readline()
                while line:
                    line = line[:-1]
                    if line and line[-1] == "\r":
                        line = line[:-1]
                    line = line.rstrip(' ')
                    l = line.split(' ')
                    outls = [[] for i in range(args.n_parties)]

                    for item in l[1:]:
                        k, v = item.split(":")
                        outls[id2party[int(k)]].append("{}:{}".format(id2local[int(k)]+1+offset[id2party[int(k)]], v))


                    tmpl = [l[0]]
                    for i in range(args.n_parties):
                        tmpl.extend(outls[i])
                    out_tfs.write(" ".join(tmpl)+'\n')
                    line = f.readline()

    if args.fate:
        with open(args.file, 'r') as f:
            outfs = [open(args.output+args.dataset+"_FATE_v_"+str(args.n_parties)+"_"+str(i), 'w') for i in range(args.n_parties)]
            line = f.readline()
            instances = 0
            while line:
                line = line[:-1]
                if line and line[-1] == "\r":
                    line = line[:-1]
                line = line.rstrip(' ')
                l = line.split(' ')
                outls = [[] for i in range(args.n_parties)]
                for i in range(args.n_parties):
                    outls[i].append(str(instances))
                    outls[i].append(l[0])

                for item in l[1:]:
                    k, v = item.split(":")
                    outls[id2party[int(k)]].append("{}:{}".format(id2local[int(k)], v))

                for i in range(args.n_parties):
                    outfs[i].write(",".join(outls[i])+'\n')


                line = f.readline()
                instances += 1

            for item in outfs:
                item.close()


if __name__ == '__main__':
    args = get_args()
    if args.mode == "hori":
        hori_partition(args)
    elif args.mode == "verti":
        verti_partition(args)