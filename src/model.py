import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, Dropout, LayerNorm, Identity
import torch.nn.functional as F
from typing import Tuple
import numpy as np
import torch.nn as nn
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs
from CoGNN.action_gumbel_layer import TempSoftPlus, ActionNet
from config import FLAGS
from utils import MLP, _get_y_with_target
from collections import OrderedDict, defaultdict
from nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU

class Net(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActionNetArgs):
        super(Net, self).__init__()
        self.task = FLAGS.task
        self.target = FLAGS.target
        self.D = FLAGS.D
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        self.first_MLP_env_attr = MLP(7, env_args.env_dim, activation_type=FLAGS.activation)
        self.first_MLP_act_attr = MLP(7, action_args.hidden_dim, activation_type=FLAGS.activation)
        self.first_MLP_node = MLP(153, env_args.env_dim, activation_type=FLAGS.activation)
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args.env_dim)
        self.temp = gumbel_args.temp

        self.num_layers = env_args.num_layers
        self.env_net = env_args.load_net()
        layer_norm_cls = LayerNorm if env_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(env_args.env_dim)
        self.skip = env_args.skip
        self.dropout = Dropout(p=env_args.dropout)
        self.drop_ratio = env_args.dropout
        self.act = env_args.act_type.get()
        self.in_act_net = ActionNet(action_args=action_args)
        self.out_act_net = ActionNet(action_args=action_args)

        self.gate_nn = nn.Sequential(nn.Linear(self.D, self.D), ReLU(), Linear(self.D, 1))
        self.glob = MyGlobalAttention(self.gate_nn, None)

        if self.task == 'regression':
            self.loss_fucntion = torch.nn.MSELoss()
        else:
            self.loss_fucntion = torch.nn.CrossEntropyLoss()

        self.MLPs = nn.ModuleDict()
        if 'regression' in self.task:
            _target_list = self.target
            if not isinstance(FLAGS.target, list):
                _target_list = [self.target]
            self.target_list = [t for t in _target_list]
        else:
            self.target_list = ['perf']
        d = self.D
        if d > 64:
            hidden_channels = [d // 2, d // 4, d // 8, d // 16, d // 32]
        else:
            hidden_channels = [d // 2, d // 4, d // 8]
        for target in self.target_list:
            self.MLPs[target] = MLP(d, FLAGS.out_dim, activation_type=FLAGS.activation,
                                    hidden_channels=hidden_channels,
                                    num_hidden_lyr=len(hidden_channels))

    def forward(self, data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch
        if hasattr(data, 'kernel'):
            gname = data.kernel[0]
        env_edge_attr = self.first_MLP_env_attr(edge_attr)
        act_edge_attr = self.first_MLP_act_attr(edge_attr)
        x = self.first_MLP_node(x)
        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)

            # action
            in_logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_attr,
                                        act_edge_attr=act_edge_attr)
            out_logits = self.out_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_attr,
                                          act_edge_attr=act_edge_attr)
            temp = self.temp_model(x=x, edge_index=edge_index,
                                   edge_attr=env_edge_attr) if self.learn_temp else self.temp
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            edge_weight = self.create_edge_weight(edge_index=edge_index,
                                                  keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])

            # environment
            out = self.env_net[0 + gnn_idx](x=x, edge_index=edge_index, edge_weight=edge_weight,
                                            edge_attr=env_edge_attr)
            out = self.dropout(out)
            out = self.act(out)
            if self.skip:
                x = x + out
            else:
                x = out
        x = self.hidden_layer_norm(x)
        x = self.env_net[-1](x)  # decoder
        out, node_att_scores = self.glob(x, batch)

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

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob