from fedtree.distributed import DistributedFLClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import argparse
import os

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 50051


def trainer(pid: int, num_parties: int):
    x, y = load_digits(return_X_y=True)
    clf = DistributedFLClassifier(
        pid=pid, server_addr=SERVER_HOST, server_port=SERVER_PORT, n_trees=2,
        mode="horizontal", n_parties=num_parties, num_class=10, objective="multi:softmax")
    clf.fit(x, y)
    if pid == 0:
        y_pred = clf.predict(x)
        accuracy = accuracy_score(y, y_pred)
        print("accuracy:", accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-parties", type=int, default=2)
    parser.add_argument("--pid", type=str, default="")
    args = parser.parse_args()
    num_parties = args.num_parties

    if args.pid:  # single machine
        trainer(int(args.pid), num_parties)
        exit(0)

    for pid in range(num_parties):  # multiple machines simulation
        if os.fork() == 0:
            # party
            trainer(pid, num_parties)
            exit(0)

    # server
    trainer(-1, num_parties)
