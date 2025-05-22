from torch_geometric.loader import DataLoader
from CoGNN.layers import ModelType
from dataset_utils import *
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv, GCNConv, GATConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn.pool import global_add_pool, global_max_pool, SAGPooling
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error
from src.config import FLAGS
from collections import OrderedDict
from torch import nn
from src.utils import MLP, _get_y_with_target
from collections import OrderedDict, defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix
from scipy.stats import rankdata, kendalltau
import os
import sys
import pandas as pd
from tqdm import tqdm
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
from src.config import FLAGS
from src.utils import MLP, _get_y_with_target
from collections import OrderedDict, defaultdict
from src.nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU
from typing import Callable
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
jknFlag = 0

def gin_mlp_func() -> Callable:
    def mlp_func(in_channels: int, out_channels: int, bias: bool):
        return Sequential(Linear(in_channels, out_channels, bias=bias),
                ReLU(), Linear(out_channels, out_channels, bias=bias))
    return mlp_func

out_dim = FLAGS.out_dim
gin_mlp_func = gin_mlp_func()

gumbel_args = GumbelArgs(learn_temp=FLAGS.learn_temp, temp_model_type=FLAGS.temp_model_type, tau0=FLAGS.tau0,
                                 temp=FLAGS.temp, gin_mlp_func=gin_mlp_func)
env_args = \
EnvArgs(model_type=ModelType.SUM_GNN, num_layers=FLAGS.env_num_layers, env_dim=FLAGS.env_dim,
        layer_norm=FLAGS.layer_norm, skip=FLAGS.skip, batch_norm=FLAGS.batch_norm, dropout=FLAGS.dropout,
        in_dim=FLAGS.num_features , out_dim=FLAGS.D, dec_num_layers=FLAGS.dec_num_layers, gin_mlp_func=gin_mlp_func,
        act_type=ActivationType.RELU)
action_args = \
        ActionNetArgs(model_type=ModelType.MEAN_GNN, num_layers=FLAGS.act_num_layers,
        hidden_dim=FLAGS.act_dim, dropout=FLAGS.dropout, act_type=ActivationType.RELU,
        gin_mlp_func=gin_mlp_func, env_dim=FLAGS.env_dim)

def _report_rmse_etc(points_dict, label, print_result=True):
    data = defaultdict(list)
    tot_mape, tot_rmse, tot_mse, tot_mae, tot_max_err, tot_tau, tot_std = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    num_data = None
    try:
        for target_name, d in points_dict.items():
            # true_li = d['true']
            # pred_li = d['pred']
            true_li = [data for data,_ in d['pred']]
            pred_li = [data for _,data in d['pred']]
            num_data = len(true_li)
            mape = mean_absolute_percentage_error(true_li, pred_li)
            rmse = mean_squared_error(true_li, pred_li)
            mse = mean_squared_error(true_li, pred_li)
            mae = mean_absolute_error(true_li, pred_li)
            max_err = max_error(true_li, pred_li)
            true_rank = rankdata(true_li)
            pred_rank = rankdata(pred_li)
            tau = kendalltau(true_rank, pred_rank)[0]
            data['target'].append(target_name)
            data['mape'].append(mape)
            data['rmse'].append(rmse)
            data['mse'].append(mse)
            data['mae'].append(mae)
            data['max_err'].append(max_err)
            data['tau'].append(tau)

            # data['rmse'].append(f'{rmse:.4f}')
            # data['mse'].append(f'{mse:.4f}')
            # data['tau'].append(f'{tau: .4f}')
            tot_mape += mape
            tot_rmse += rmse
            tot_mse += mse
            tot_mae += mae
            tot_max_err += max_err
            tot_tau += tau

            pred_std = d.get('pred_std')
            if pred_std is not None:
                assert type(pred_std) is np.ndarray, f'{type(pred_std)}'
                pred_std = np.mean(pred_std)
                data['pred_std'].append(pred_std)
                tot_std += pred_std
        data['target'].append('tot/avg')
        data['mape'].append(tot_mape)
        data['rmse'].append(tot_rmse)
        data['mse'].append(tot_mse)
        data['mae'].append(tot_mae)
        data['max_err'].append(tot_max_err)
        data['tau'].append(tot_tau / len(points_dict))
        if 'pred_std' in data:
            data['pred_std'].append(tot_std / len(points_dict))
    except ValueError as v:
        data = defaultdict(list)

    # data['rmse'].append(f'{tot_rmse:.4f}')
    # data['mse'].append(f'{tot_mse:.4f}')
    # data['tau'].append(f'{tot_tau / len(points_dict):.4f}')
    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    return df


class Net(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActionNetArgs):
        super(Net, self).__init__()
        self.loss_fucntion = torch.nn.MSELoss()
        self.target = FLAGS.target_1
        _target_list = self.target
        self.target_list = [t for t in _target_list]
        self.D = FLAGS.D
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        self.first_MLP_env_attr = MLP(2, env_args.env_dim, activation_type=FLAGS.activation)
        self.first_MLP_act_attr = MLP(2, action_args.hidden_dim, activation_type=FLAGS.activation)
        self.first_MLP_node = MLP(15, env_args.env_dim, activation_type=FLAGS.activation)
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
        self.loss_fucntion = torch.nn.MSELoss()

        self.MLPs = nn.ModuleDict()
        self.target_list = [t for t in _target_list]
        self.D = FLAGS.D
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
        x = x.to(torch.float32)
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

        for num, target_name in enumerate(self.target_list):
            # for target_name in target_list:
            out = self.MLPs[target_name](out_embed)
            y = getattr(data, 'y')[:, num]
            target = y.view((len(y), FLAGS.out_dim))
            if FLAGS.loss == 'RMSE':
                loss = torch.sqrt(self.loss_fucntion(out, target))
            elif FLAGS.loss == 'MSE':
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

def train(epoch, model, train_loader, optimizer):
    model.train()

    total_loss = 0
    correct = 0
    i = 0
    target_list = FLAGS.target_1
    loss_dict = {}
    for t in target_list:
        loss_dict[t] = 0.0
    for data in tqdm(train_loader, position=0, total=len(train_loader), file=sys.stdout):
        data = data.to(FLAGS.device)
        optimizer.zero_grad()
        out, loss, loss_dict_ = model.to(FLAGS.device)(data)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        for t in target_list:
            loss_dict[t] += loss_dict_[t].item()
        optimizer.step()
        i += 1
    return total_loss / len(train_loader.dataset), {key: v / len(train_loader) for key, v in loss_dict.items()}

def inference_loss_function(pred, true):
    return (pred - true) ** 2

def test(loader, tvt, model):
    model.eval()

    inference_loss = 0
    correct, total = 0, 0
    loss_dict = {}
    i = 0
    points_dict = OrderedDict()
    _target_list = FLAGS.target_1
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target_1]

    for t in _target_list:
        loss_dict[t] = 0.0
    for target_name in _target_list:
        points_dict[target_name] = {'true': [], 'pred': []}
    for data in tqdm(loader, position=0, total=len(loader), file=sys.stdout):
        data = data.to(FLAGS.device)
        out_dict, loss, loss_dict_ = model.to(FLAGS.device)(data)

        if FLAGS.task == 'regression':
            total += loss.item()
            for t in _target_list:
                loss_dict[t] += loss_dict_[t].item()
        else:
            loss, pred = torch.max(out_dict[FLAGS.target[0]], 1)
            labels = _get_y_with_target(data, FLAGS.target[0])
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        for num, target_name in enumerate(_target_list):
            out = out_dict[target_name]
            for i in range(len(out)):
                out_value = out[i].item()
                if FLAGS.encode_log and target_name == 'actual_perf':
                    out_value = 2 ** (out_value) * (1 / FLAGS.normalizer)
                if FLAGS.subtask == 'inference':
                    inference_loss += inference_loss_function(out_value,
                                                              _get_y_with_target(data, target_name)[i].item())
                    if out_value != _get_y_with_target(data, target_name)[i].item():
                        print(
                            f'data {i} actual value: {_get_y_with_target(data, target_name)[i].item():.2f}, predicted value: {out_value:.2f}')
                points_dict[target_name]['pred'].append(
                    (getattr(data, 'y')[i, num].item(), out_value))
                points_dict[target_name]['true'].append(
                    (getattr(data, 'y')[i, num].item(),
                     getattr(data, 'y')[i, num].item()))

        i += 1
    return total / len(loader), {key: v / len(loader) for key, v in loss_dict.items()}

def inference(dataset):
    from torch.utils.data import random_split  # TODO: inductive

    num_graphs = len(dataset)
    r1 = int(num_graphs * 1)
    r2 = int(num_graphs * 0.0)
    li = random_split(dataset, [r1, r2, len(dataset) - r1 - r2],
                      generator=torch.Generator().manual_seed(100))
    print(f'{num_graphs} graphs in total:'
                   f' {len(li[0])} train {len(li[1])} val '
                   f'{len(li[2])} test')
    # test set ratio为0.2
    train_loader = DataLoader(li[0], batch_size=FLAGS.batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(li[1], batch_size=FLAGS.batch_size,
                            pin_memory=True)  # TODO: split make sure no seen kernels in val/test
    test_loader = DataLoader(li[2], batch_size=FLAGS.batch_size, pin_memory=True)  # TODO
    data_ini = None
    for step, data in enumerate(train_loader):
        if step == 0:
            data_ini = data
            break
    model = Net(gumbel_args=gumbel_args, env_args=env_args, action_args=action_args).to(FLAGS.device)
    old_state_dict = torch.load('/home/xxx/CoGNN-DSE_1/baseline_dataset/model/CoGNN_NEW.pth', map_location=torch.device(FLAGS.device))
    model.load_state_dict(old_state_dict)
    testr, loss_dict = test(train_loader, 'test', model)
    value_1 = 0
    len_1 = 0
    for key, values in loss_dict.items():
        if key in ['uram', 'srl']:
            continue
        len_1 += 1
        value_1 += values
        print(f'{key} RMSE: {values}')
    value_1 /= len_1
    print(f'avg RMSE: {value_1}')
    print('Test loss: {:.7f}'.format(testr))

# inference process
if __name__ == "__main__":
    dataset_dir = os.path.abspath('/home/xxx/CoGNN-DSE_1/baseline_dataset/std')
    dataset = os.listdir(dataset_dir)
    dataset_list = generate_dataset(dataset_dir, FLAGS.dataset_unseen, print_info=False)
    inference(dataset_list)
# train process
# if __name__ == "__main__":
#     batch_size = 64
#     dataset_dir = os.path.abspath('/home/xxx/CoGNN-DSE_1/baseline_dataset/std')
#     dataset_list = generate_dataset(dataset_dir, FLAGS.dataset_seen, print_info=False)
#     print(f'Reading dataset from {dataset_dir}')
#     train_ds, test_ds, val_ds = split_dataset(dataset_list, shuffle=True, seed=128)
#     print(f'train_ds size = {len(train_ds)}, test_ds size = {len(test_ds)}, val_ds size = {len(val_ds)}')
#
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)
#     data_ini = None
#     for step, data in enumerate(train_loader):
#         if step == 0:
#             data_ini = data
#             break
#     model = Net(gumbel_args=gumbel_args, env_args=env_args, action_args=action_args).to(FLAGS.device)
#     print(model)
#     model_path = '/home/xxx/CoGNN-DSE_1/baseline_dataset/model/CoGNN_NEW.pth'
#     if os.path.isfile(model_path):
#         old_state_dict = torch.load(model_path, map_location=torch.device(FLAGS.device))
#         model.load_state_dict(old_state_dict)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     train_losses = []
#     val_losses = []
#     test_losses = []
#     epochs = range(1000)
#
#     for epoch in epochs:
#         print(f'Epoch {epoch + 1} train')
#         loss, loss_dict_train = train(epoch, model, train_loader, optimizer)
#         print(f'Epoch {epoch + 1} test')
#         val, loss_dict_val = test(val_loader, 'val', model)
#         print(f'Epoch {epoch + 1} val')
#         testr, loss_dict_test = test(test_loader, 'test', model)
#         print(('Epoch: {:03d}, Train Loss: {:.4f}, Val loss: {:.4f}, Test: {:.4f})'.format(epoch + 1, loss, val, testr)))
#         train_losses.append(loss)
#         test_losses.append(testr)
#         val_losses.append(val)
#         torch.save(model.state_dict(), model_path)
#     import matplotlib
#
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#
#     plt.plot(epochs, train_losses, 'g', label='Training loss')
#     if len(val_loader) > 0:
#         plt.plot(epochs, val_losses, 'b', label='Validation loss')
#     plt.plot(epochs, test_losses, 'r', label='Testing loss')
#     plt.title('Training, Validation, and Testing loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()
#     print(f'min test loss at epoch: {test_losses.index(min(test_losses)) + 1}')
#     print(f'min train loss at epoch: {train_losses.index(min(train_losses)) + 1}')

