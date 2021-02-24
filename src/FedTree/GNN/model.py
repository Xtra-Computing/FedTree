import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv  # noqa


class GCN(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super(GCN, self).__init__()
        # todo: try normalize in GCNConv
        self.conv1 = GCNConv(n_features, 16, cached=False)
        self.conv2 = GCNConv(16, n_classes, cached=False)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        #print("x in model:", x)
        #print("edge_index in model:", edge_index)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
