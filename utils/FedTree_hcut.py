# cut dataset for FedTree horizontal distributed
import argparse
import random
import os
import copy

def main(args):
    total_features = 0
    size = 0
    label_set = set()
    with open(args.input_file, 'r') as fin:
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
    
    l = [i for i in range(size)]
    random.seed(42)
    random.shuffle(l)
    BLOCK = size // args.num_parties
    offset = [i*BLOCK+min(i, size % args.num_parties) for i in range(args.num_parties)]+[size]
    
    for i in range(args.num_parties):
        l[offset[i]:offset[i+1]] = sorted(l[offset[i]:offset[i+1]])
    
    with open(args.input_file, 'r') as fin:
        fouts = [open(args.prefix+"_h_{}_{}".format(args.num_parties, i), 'w') for i in range(args.num_parties)]
        line = fin.readline()
        glob_idx = 0
        p_off = copy.deepcopy(offset)
        while line:
            line = line[:-1]
            if line and line[-1] == "\r":
                line = line[:-1]
            line = line.rstrip(' ')
            
            for i in range(args.num_parties):
                if p_off[i] < offset[i+1] and l[p_off[i]] == glob_idx:
                    fouts[i].write(line+"\n")
                    p_off[i] += 1
            glob_idx += 1
            line = fin.readline()
        for item in fouts:
            item.close()
    if not args.fate:
        return
    
    with open(args.input_file, 'r') as fin:
        fouts = [open(args.prefix+"_FATE_h_{}_{}".format(args.num_parties, i), 'w') for i in range(args.num_parties)]
        glob_idx = 0
        p_off = copy.deepcopy(offset)
        line = fin.readline()
        while line:
            line = line[:-1]
            if line and line[-1] == "\r":
                line = line[:-1]
            line = line.rstrip(' ')
            for i in range(args.num_parties):
                if p_off[i] < offset[i+1] and l[p_off[i]] == glob_idx:
                    
                    ele = line.split(' ')
                    f_ele = [str(glob_idx), label_dict[str(int(ele[0]))]]
                    f_ele += ["{}:{}".format(-1+int(item.split(':')[0]), item.split(':')[1]) for item in ele[1:]]
                    
                    fouts[i].write(','.join(f_ele)+'\n')
                    p_off[i] += 1

            glob_idx += 1
            line = fin.readline()
        for item in fouts:
            item.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_parties",
        type=int
    )
    parser.add_argument(
        "-f",
        "--input_file",
        type=str
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str
    )
    parser.add_argument(
        "-fate",
        action='store_true'
    )
    
    args = parser.parse_args()
    
    main(args)
