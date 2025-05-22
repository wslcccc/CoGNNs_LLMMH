import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.loader import DataLoader
import torch
from src.config import FLAGS
from src.utils import MLP, _get_y_with_target
from torch import nn
from collections import OrderedDict
import tqdm

class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=None, num_layers=2, drop_out=0.1):
        super(GCNNet, self).__init__()
        self.task = FLAGS.task
        if self.task == 'regression':
            self.loss_fucntion = torch.nn.MSELoss()
        self.drop_out = drop_out
        self.target = FLAGS.target
        if 'regression' in self.task:
            _target_list = self.target
            if not isinstance(FLAGS.target, list):
                _target_list = [self.target]
            self.target_list = [t for t in _target_list]
        else:
            self.target_list = ['perf']
        if hidden_channels is None:
            hidden_channels = [64, 128]
        self.drop_out = drop_out
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels[i]))
            else:
                self.convs.append(GCNConv(hidden_channels[i - 1], hidden_channels[i]))

        self.global_pool = global_mean_pool
        self.D = FLAGS.D
        d = self.D
        out_dim = FLAGS.out_dim
        self.channels = [d // 2, d // 4, d // 8]
        self.MLPs = nn.ModuleDict()
        for target in self.target_list:
            self.MLPs[target] = MLP(d, out_dim, activation_type=FLAGS.activation,
                                    hidden_channels=self.channels,
                                    num_hidden_lyr=len(self.channels))
        self.first_MLP = Linear(128, d)

    def forward(self, data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch

        for idx in range(len(self.convs)):
            x = self.convs[idx](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.0, training=self.training)

        x = self.global_pool(x, batch)
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
