import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn.conv import PNAConv
from torch_geometric.nn.pool import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation, Set2Set
from torch.nn import BatchNorm1d
# from torch_geometric.nn.norm import BatchNorm
import torch
from src.config import FLAGS
from src.utils import MLP, _get_y_with_target
from torch import nn
from collections import OrderedDict

class PNANet(torch.nn.Module):
    def __init__(self, in_dim, deg, num_layer=2, emb_dim=128, edge_dim=2, drop_ratio=0.5, JK="last",
                 residual=False, graph_pooling="sum"):
        super(PNANet, self).__init__()
        self.task = FLAGS.task
        if self.task == 'regression':
            self.loss_fucntion = torch.nn.MSELoss()
        self.target = FLAGS.target
        if 'regression' in self.task:
            _target_list = self.target
            if not isinstance(FLAGS.target, list):
                _target_list = [self.target]
            self.target_list = [t for t in _target_list]
        else:
            self.target_list = ['perf']
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.num_layer = num_layer
        self.residual = residual  # add residual connection or not
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.graph_pooling = graph_pooling
        self.deg = deg

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        # self.batch_norms = ModuleList()
        for idx in range(num_layer):
            if idx == 0:
                conv = PNAConv(in_channels=in_dim, out_channels=emb_dim,
                               aggregators=aggregators, scalers=scalers, deg=deg,
                               edge_dim=edge_dim, towers=5, pre_layers=1, post_layers=1,
                               divide_input=False)
            else:
                conv = PNAConv(in_channels=emb_dim, out_channels=emb_dim,
                               aggregators=aggregators, scalers=scalers, deg=deg,
                               edge_dim=edge_dim, towers=5, pre_layers=1, post_layers=1,
                               divide_input=False)
            self.convs.append(conv)
            # self.batch_norms.append(BatchNorm(emb_dim))

        # pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = AttentionalAggregation(
                gate_nn=Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), ReLU(),
                                   Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = ModuleList()
        # self.graph_norm = ModuleList()

        self.D = FLAGS.D
        d = self.D
        out_dim = FLAGS.out_dim
        self.channels = [d // 2, d // 4, d // 8]
        self.MLPs = nn.ModuleDict()
        for target in self.target_list:
            self.MLPs[target] = MLP(d, out_dim, activation_type=FLAGS.activation,
                                    hidden_channels=self.channels,
                                    num_hidden_lyr=len(self.channels))
        self.first_MLP = Linear(emb_dim, d)

    def forward(self, data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch

        h_list = [x.to(torch.float32)]

        for layer in range(self.num_layer):

            h = self.convs[layer](x=h_list[layer], edge_index=edge_index, edge_attr=edge_attr)
            # h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # different implementations of JK-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                if layer > 0:
                    node_representation += h_list[layer]
        else:
            node_representation = h_list[-1]

        h_graph = self.pool(node_representation, data.batch)
        out = self.first_MLP(h_graph)
        out_dict = OrderedDict()
        total_loss = 0
        out_embed = out
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

