# cut dataset for FedTree vertical distributed and FATE hetero boost

import argparse
import random

def main(args):
    random.seed(42)
    num_features = 0
    with open(args.input_file, 'r') as f:
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
    features = [i+1 for i in range(num_features)]
    random.shuffle(features)
    BLOCK = num_features // args.num_parties
    offset = [i*BLOCK for i in range(args.num_parties)]+[num_features]
    
    id2party = {}
    id2local = {}
    for i in range(args.num_parties):
        features[offset[i]: offset[i+1]] = sorted(features[offset[i]: offset[i+1]])
        for j in range(offset[i], offset[i+1]):
            id2party[features[j]] = i
            id2local[features[j]] = j-offset[i]
    print(features)
    
    with open(args.input_file, 'r') as f:
        outfs = [open(args.prefix+"_v_"+str(args.num_parties)+"_"+str(i), 'w') for i in range(args.num_parties)]
        line = f.readline()
       
        while line:
            line = line[:-1]
            if line and line[-1] == "\r":
                line = line[:-1]
            line = line.rstrip(' ')
            l = line.split(' ')
            outls = [[] for i in range(args.num_parties)]
            for i in range(args.num_parties):
                outls[i].append(l[0])
            
            for item in l[1:]:
                k, v = item.split(":")
                outls[id2party[int(k)]].append("{}:{}".format(id2local[int(k)]+1, v))
            
            for i in range(args.num_parties):
                outfs[i].write(" ".join(outls[i])+'\n')
            
            
            line = f.readline()
        
        for item in outfs:
            item.close()
    
    
    if args.test_file:
        with open(args.test_file, 'r') as f:
            with open(args.prefix+"_v_"+str(args.num_parties)+"_test", 'w') as out_tfs:
                line = f.readline()
                while line:
                    line = line[:-1]
                    if line and line[-1] == "\r":
                        line = line[:-1]
                    line = line.rstrip(' ')
                    l = line.split(' ')
                    outls = [[] for i in range(args.num_parties)]
                    
                    for item in l[1:]:
                        k, v = item.split(":")
                        outls[id2party[int(k)]].append("{}:{}".format(id2local[int(k)]+1+offset[id2party[int(k)]], v))
                    
                    
                    tmpl = [l[0]]
                    for i in range(args.num_parties):
                        tmpl.extend(outls[i])
                    out_tfs.write(" ".join(tmpl)+'\n')
                    line = f.readline()

    if args.fate:
        with open(args.input_file, 'r') as f:
            outfs = [open(args.prefix+"_FATE_v_"+str(args.num_parties)+"_"+str(i), 'w') for i in range(args.num_parties)]
            line = f.readline()
            instances = 0
            while line:
                line = line[:-1]
                if line and line[-1] == "\r":
                    line = line[:-1]
                line = line.rstrip(' ')
                l = line.split(' ')
                outls = [[] for i in range(args.num_parties)]
                for i in range(args.num_parties):
                    outls[i].append(str(instances))
                    outls[i].append(l[0])
                
                for item in l[1:]:
                    k, v = item.split(":")
                    outls[id2party[int(k)]].append("{}:{}".format(id2local[int(k)], v))
                
                for i in range(args.num_parties):
                    outfs[i].write(",".join(outls[i])+'\n')
                
                
                line = f.readline()
                instances += 1
            
            for item in outfs:
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
        "-t",
        "--test_file",
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
