import torch
from model import GCN

net = GCN()
net.load_state_dict(torch.load(path))

graph_datasets = read_data("tree_predict.txt")

device = args.device
net = GCN().to(device)
# data = data.to(device)

was_training = False
if model.training:
    model.eval()
    was_training = True

with torch.no_grad():
    for data in graph_datasets:
        # print("x:",x)
        # print("target:",target)

        pred_label = net(data).max(dim=1)[1]


logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

if was_training:
    model.train()
