from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv, GCNConv, GATConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn.pool import global_add_pool, global_max_pool, SAGPooling
from src.config import FLAGS
from collections import OrderedDict
from torch import nn
from src.utils import MLP, _get_y_with_target
jknFlag = 0

class HierNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, conv_type, drop_out=0.0, pool_ratio=0.5):
        super(HierNet, self).__init__()
        self.task = FLAGS.task
        if self.task == 'regression':
            self.loss_fucntion = torch.nn.MSELoss()
        self.drop_out = drop_out
        self.pool_ratio = pool_ratio
        self.target = FLAGS.target
        if 'regression' in self.task:
            _target_list = self.target
            if not isinstance(FLAGS.target, list):
                _target_list = [self.target]
            self.target_list = [t for t in _target_list]
        else:
            self.target_list = ['perf']
        if conv_type == 'gcn':
            conv = GCNConv
        elif conv_type == 'gat':
            conv = GATConv
        elif conv_type == 'sage':
            conv = SAGEConv
        else:
            conv = GCNConv

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.convs.append(conv(in_channels, hidden_channels))
            else:
                self.convs.append(conv(hidden_channels, hidden_channels))
            self.pools.append(SAGPooling(hidden_channels, self.pool_ratio))
        if jknFlag:
            self.jkn = JumpingKnowledge('lstm', channels=hidden_channels, num_layers=2)

        self.global_pool = global_add_pool
        self.D = FLAGS.D
        d = self.D
        out_dim = FLAGS.out_dim
        self.channels = [d // 2, d // 4, d // 8]
        self.MLPs = nn.ModuleDict()
        for target in self.target_list:
            self.MLPs[target] = MLP(d, out_dim, activation_type=FLAGS.activation,
                                    hidden_channels=self.channels,
                                    num_hidden_lyr=len(self.channels))
        self.first_MLP = Linear(256, d)

    def forward(self, data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch
        h_list = []

        for step in range(len(self.convs)):
            x = self.convs[step](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)
            x, edge_index, _, batch, _, _ = self.pools[step](x, edge_index, None, batch, None)
            h = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
            h_list.append(h)

        if jknFlag:
            x = self.jkn(h_list)
        x = h_list[0] + h_list[1] + h_list[2]
        x = self.first_MLP(x)
        out_dict = OrderedDict()
        total_loss = 0
        out_embed = x
        loss_dict = {}

        for target_name in self.target_list:
            # for target_name in target_list:
            out = self.MLPs[target_name](out_embed)
            y = _get_y_with_target(data, target_name)
            if self.task == 'regression':
                target = y.view((len(y), FLAGS.out_dim))
                if FLAGS.loss == 'RMSE':
                    loss = torch.sqrt(self.loss_fucntion(out, target))
                elif FLAGS.loss == 'MSE':
                    loss = self.loss_fucntion(out, target)
                else:
                    raise NotImplementedError()
            else:
                target = y.view((len(y)))
                loss = self.loss_fucntion(out, target)
            out_dict[target_name] = out
            total_loss += loss
            loss_dict[target_name] = loss
        return out_dict, total_loss, loss_dict


