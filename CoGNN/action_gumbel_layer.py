
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, ModuleList
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj, OptTensor
from CoGNN.model_parse import GumbelArgs, ActionNetArgs
import tqdm

class ActionNet(nn.Module):
    def __init__(self, action_args: ActionNetArgs):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        self.num_layers = action_args.num_layers
        self.net = action_args.load_net()
        self.dropout = nn.Dropout(action_args.dropout)
        self.act = action_args.act_type.get()

    def forward(self, x: Tensor, edge_index: Adj, env_edge_attr: OptTensor, act_edge_attr: OptTensor) -> Tensor:
        edge_attrs = [env_edge_attr] + (self.num_layers - 1) * [act_edge_attr]
        for idx, (edge_attr, layer) in enumerate(zip(edge_attrs[:-1], self.net[:-1])):
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.dropout(x)
            x = self.act(x)
        x = self.net[-1](x=x, edge_index=edge_index, edge_attr=edge_attrs[-1])
        return x

class TempSoftPlus(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_dim: int):
        super(TempSoftPlus, self).__init__()
        model_list =\
            gumbel_args.temp_model_type.get_component_list(in_dim=env_dim, hidden_dim=env_dim, out_dim=1, num_layers=1,
                                                           bias=False, edges_required=False,
                                                           gin_mlp_func=gumbel_args.gin_mlp_func)
        self.linear_model = ModuleList(model_list)
        self.softplus = nn.Softplus(beta=1)
        self.tau0 = gumbel_args.tau0

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor):
        x = self.linear_model[0](x=x, edge_index=edge_index,edge_attr = edge_attr)
        x = self.softplus(x) + self.tau0
        temp = x.pow_(-1)
        return temp.masked_fill_(temp == float('inf'), 0.)