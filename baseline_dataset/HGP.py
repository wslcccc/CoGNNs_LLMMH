from torch_geometric.loader import DataLoader
from dataset_utils import *
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
from collections import OrderedDict, defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix
from scipy.stats import rankdata, kendalltau
import os
import sys
import pandas as pd
from tqdm import tqdm
jknFlag = 0

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


class HierNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, hls_dim, num_layers, conv_type, drop_out=0.0, pool_ratio=0.5):
        super(HierNet, self).__init__()
        self.loss_fucntion = torch.nn.MSELoss()
        self.drop_out = drop_out
        self.pool_ratio = pool_ratio
        self.target = FLAGS.target_1
        _target_list = self.target
        self.target_list = [t for t in _target_list]
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
        self.channels = [hidden_channels * 2 + hls_dim, 64, 64, 1]
        self.MLPs = nn.ModuleDict()
        for target in self.target_list:
            self.MLPs[target] = MLP(d, out_dim, activation_type=FLAGS.activation,
                                    hidden_channels=self.channels,
                                    num_hidden_lyr=len(self.channels))
        self.first_MLP = Linear(128, d)

    def forward(self, data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch
        x = x.to(torch.float32)
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
    model = HierNet(in_channels=data_ini.num_features, hidden_channels=64, num_layers=3, conv_type='sage',
                    hls_dim=6, drop_out=0.0)
    old_state_dict = torch.load('/home/wslcccc/CoGNN-DSE_1/baseline_dataset/model/HGP_NEW.pth', map_location=torch.device(FLAGS.device))
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
    dataset_dir = os.path.abspath('/home/wslcccc/CoGNN-DSE_1/baseline_dataset/std')
    dataset = os.listdir(dataset_dir)
    dataset_list = generate_dataset(dataset_dir, FLAGS.dataset_unseen, print_info=False)
    inference(dataset_list)
# train process
# if __name__ == "__main__":
#     batch_size = 128
#     dataset_dir = os.path.abspath('/home/wslcccc/CoGNN-DSE_1/baseline_dataset/std')
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
#     model = HierNet(in_channels=data_ini.num_features, hidden_channels=64, num_layers=3, conv_type='sage',
#                     hls_dim=6, drop_out=0.0)
#     model = model.to(FLAGS.device)
#     print(model)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     train_losses = []
#     val_losses = []
#     test_losses = []
#     epochs = range(FLAGS.epoch_num)
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
#         torch.save(model.state_dict(),
#                    '/home/wslcccc/CoGNN-DSE_1/baseline_dataset/model/HGP_NEW.pth')
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

