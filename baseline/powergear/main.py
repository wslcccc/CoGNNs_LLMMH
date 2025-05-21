from baseline_dataset.POWERGEAR_HELPER.Conv import HECConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LSTM
from torch_geometric.nn import global_mean_pool, global_add_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HECConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dim, overall_dim=0, use_overall: bool = False,
                 batch_norm: bool = False, drop_out=0.5, pool_aggr="add", overall_dim_large=128, relations=4,
                 aggregate="add", simple_JK="last"):
        super(HECConvNet, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(HECConv(in_channels, hidden_channels, dim, num_relation=relations, aggr=aggregate))
        if use_overall:
            self.fc1 = Linear(hidden_channels+overall_dim_large, hidden_channels).cuda()
            self.fc2 = Linear(hidden_channels, 1).cuda()
            self.large_overall = Linear(overall_dim, overall_dim_large)
        else:
            self.fc1 = Linear(hidden_channels, hidden_channels//2).cuda()
            self.fc2 = Linear(hidden_channels//2, 1).cuda()
        if simple_JK == "cat":
            self.cat_fc = Linear(num_layers*hidden_channels, hidden_channels)
        if simple_JK == 'lstm':
            assert in_channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(
                hidden_channels, (num_layers * hidden_channels) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * ((num_layers * hidden_channels) // 2), 1)

        self.use_overall = use_overall
        self.drop_out = drop_out
        self.pool_aggr = pool_aggr
        self.overall_dim_large = overall_dim_large
        self.JK = simple_JK
        print("relations:", relations)

    def forward(self, data):
        x, edge_index, edge_attr, batch, overall_attr, edge_type = data.x, data.edge_index, data.edge_attr, \
            data.node_batch, data.overall, data.edge_type
        h_list = [x]
        for i, conv in enumerate(self.convs):
            h = conv(h_list[i], edge_index, edge_weight=edge_attr, edge_type=edge_type)
            if i != self.num_layers - 1:
                h = h.relu()
                h = F.dropout(h, p=self.drop_out, training=self.training)
            h_list.append(h)
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)
        elif self.JK == 'cat':
            node_representation = torch.cat(h_list[1:], dim=1)
            node_representation = self.cat_fc(node_representation)
        elif self.JK == 'max':
            node_representation = torch.stack(h_list[1:], dim=-1).max(dim=-1)[0]
        elif self.JK == 'lstm':
            node_representation = torch.stack(h_list[1:], dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(node_representation)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            node_representation = (node_representation * alpha.unsqueeze(-1)).sum(dim=1)
        if self.pool_aggr == "add":
            node_representation = global_add_pool(node_representation, batch)
        elif self.pool_aggr == "mean":
            node_representation = global_mean_pool(node_representation, batch)
        if self.use_overall:
            overall_attr = self.large_overall(overall_attr.view(node_representation.size(0), -1))
            overall_attr = overall_attr.relu()
            node_representation = torch.cat([node_representation, overall_attr], dim=-1)
        node_representation = self.fc1(node_representation)
        # x = self.bn1(x)
        node_representation = F.relu(node_representation)
        node_representation = F.dropout(node_representation, p=self.drop_out, training=self.training)
        node_representation = self.fc2(node_representation)
        x_return = torch.squeeze(node_representation)
        return x_return

