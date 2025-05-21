from enum import Enum, auto
from torch.nn import Linear, ModuleList, Module, Dropout, ReLU, GELU, Sequential
from torch import Tensor
from typing import NamedTuple, Any, Callable
import torch.nn.functional as F
from torch_geometric.nn.glob import global_mean_pool, global_add_pool
from CoGNN.layers import ModelType


class ActivationType(Enum):
    """
        an object for the different activation types
    """
    RELU = auto()
    GELU = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ActivationType[s]
        except KeyError:
            raise ValueError()

    def get(self):
        if self is ActivationType.RELU:
            return F.relu
        elif self is ActivationType.GELU:
            return F.gelu
        else:
            raise ValueError(f'ActivationType {self.name} not supported')

    def nn(self) -> Module:
        if self is ActivationType.RELU:
            return ReLU()
        elif self is ActivationType.GELU:
            return GELU()
        else:
            raise ValueError(f'ActivationType {self.name} not supported')


class GumbelArgs(NamedTuple):
    learn_temp: bool
    temp_model_type: ModelType
    tau0: float
    temp: float
    gin_mlp_func: Callable


class EnvArgs(NamedTuple):
    model_type: ModelType
    num_layers: int
    env_dim: int

    layer_norm: bool
    skip: bool
    batch_norm: bool
    dropout: float
    act_type: ActivationType
    dec_num_layers: int

    in_dim: int
    out_dim: int
    gin_mlp_func: Callable

    def load_net(self) -> ModuleList:
        component_list = \
            self.model_type.get_component_list(in_dim=self.env_dim, hidden_dim=self.env_dim, out_dim=self.env_dim,
                                               num_layers=self.num_layers, bias=True, edges_required=True,
                                               gin_mlp_func=self.gin_mlp_func)

        if self.dec_num_layers > 1:
            mlp_list = (self.dec_num_layers - 1) * [Linear(self.env_dim, self.env_dim),
                                                    Dropout(self.dropout), self.act_type.nn()]
            mlp_list = mlp_list + [Linear(self.env_dim, self.out_dim)]
            dec_list = [Sequential(*mlp_list)]
        else:
            dec_list = [Linear(self.env_dim, self.out_dim)]

        return ModuleList(component_list + dec_list)


class ActionNetArgs(NamedTuple):
    model_type: ModelType
    num_layers: int
    hidden_dim: int

    dropout: float
    act_type: ActivationType

    env_dim: int
    gin_mlp_func: Callable

    def load_net(self) -> ModuleList:
        net = self.model_type.get_component_list(in_dim=self.env_dim, hidden_dim=self.hidden_dim, out_dim=2,
                                                 num_layers=self.num_layers, bias=True, edges_required=False,
                                                 gin_mlp_func=self.gin_mlp_func)
        return ModuleList(net)


class BatchIdentity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return x

