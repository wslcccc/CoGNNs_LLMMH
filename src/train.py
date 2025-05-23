from config import FLAGS
import sys
from saver import saver
from utils import MLP, OurTimer, get_save_path, _get_y_with_target
import programl_data
SAVE_DIR = programl_data.SAVE_DIR
from torch_geometric.utils import degree
from model import Net
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error
from scipy.stats import rankdata, kendalltau
from torch.nn import Sequential, Linear, ReLU
from tqdm import tqdm
from os.path import join
from collections import OrderedDict, defaultdict
import pandas as pd
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
from typing import NamedTuple, Any, Callable
import numpy as np
from baseline.HGP.HGP import HierNet
from baseline.ironman.Ironman import GCNNet
from baseline.pnar.pna import PNANet

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
EnvArgs(model_type=FLAGS.env_model_type, num_layers=FLAGS.env_num_layers, env_dim=FLAGS.env_dim,
        layer_norm=FLAGS.layer_norm, skip=FLAGS.skip, batch_norm=FLAGS.batch_norm, dropout=FLAGS.dropout,
        in_dim=FLAGS.num_features , out_dim=FLAGS.D, dec_num_layers=FLAGS.dec_num_layers, gin_mlp_func=gin_mlp_func,
        act_type=ActivationType.RELU)
action_args = \
        ActionNetArgs(model_type=FLAGS.act_model_type, num_layers=FLAGS.act_num_layers,
        hidden_dim=FLAGS.act_dim, dropout=FLAGS.dropout, act_type=ActivationType.RELU,
        gin_mlp_func=gin_mlp_func, env_dim=FLAGS.env_dim)
def report_class_loss(points_dict):
    d = points_dict[FLAGS.target[0]]
    labels = [data for data,_ in d['pred']]
    pred = [data for _,data in d['pred']]
    target_names = ['invalid', 'valid']
    saver.info('classification report')
    saver.log_info(classification_report(labels, pred, target_names=target_names))
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    saver.info(f'Confusion matrix:\n{cm}')

def _report_rmse_etc(points_dict, label, print_result=True):
    if print_result:
        saver.log_info(label)
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
        saver.log_info(f'Error {v}')
        data = defaultdict(list)

    # data['rmse'].append(f'{tot_rmse:.4f}')
    # data['mse'].append(f'{tot_mse:.4f}')
    # data['tau'].append(f'{tot_tau / len(points_dict):.4f}')
    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    if print_result:
        saver.log_info(num_data)
        saver.log_info(df.round(4))
    # exit()
    return df
    # exit()


def inference(dataset):
    from torch.utils.data import random_split  # TODO: inductive

    num_graphs = len(dataset)
    r1 = int(num_graphs * 0.8)
    r2 = int(num_graphs * 0.0)
    li = random_split(dataset, [r1, r2, len(dataset) - r1 - r2],
                          generator=torch.Generator().manual_seed(100))
    saver.log_info(f'{num_graphs} graphs in total:'
          f' {len(li[0])} train {len(li[1])} val '
          f'{len(li[2])} test')
    #test set ratio为0.2
    train_loader = DataLoader(li[0], batch_size=FLAGS.batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(li[1], batch_size=FLAGS.batch_size, pin_memory=True)  # TODO: split make sure no seen kernels in val/test
    test_loader = DataLoader(li[2], batch_size=FLAGS.batch_size, pin_memory=True)  # TODO
    edge_dim = train_loader.dataset[0].edge_attr.shape[1]
    if FLAGS.comparative_if:
        if FLAGS.comparative_model == 'HGP':
            model = HierNet(in_channels=FLAGS.num_features, hidden_channels=FLAGS.hidden_num, num_layers=3, conv_type='sage', drop_out=0.0)
        elif FLAGS.comparative_model == 'ironman':
            model = GCNNet(in_channels=FLAGS.num_features)
        else:
            max_degree = -1
            for data in li[2]:
                d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                max_degree = max(max_degree, int(d.max()))

            # compute the in-degree histogram tensor
            deg = torch.zeros(max_degree + 1, dtype=torch.long)
            for data in li[2]:
                d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                deg += torch.bincount(d, minlength=deg.numel())
            model = PNANet(in_dim=FLAGS.num_features, deg=deg, num_layer=2, emb_dim=200, edge_dim=edge_dim,
                           drop_ratio=0.5)

    else:
        model = Net(gumbel_args=gumbel_args, env_args=env_args, action_args=action_args).to(FLAGS.device)
    if FLAGS.task == 'regression':
        if FLAGS.model_path != None:
            old_state_dict = torch.load(FLAGS.model_path, map_location=torch.device(FLAGS.device))
            model.load_state_dict(old_state_dict)
        else:
            saver.error(f'model path should be set during inference')
            raise RuntimeError()
        print(model)
    else:
        if FLAGS.class_model_path != None:
            old_state_dict = torch.load(FLAGS.class_model_path, map_location=torch.device(FLAGS.device))
            model.load_state_dict(old_state_dict)
        else:
            saver.error(f'model path should be set during inference')
            raise RuntimeError()
        print(model)
    saver.log_model_architecture(model)

    if FLAGS.task == 'regression':
        testr, loss_dict, encode_loss = test(test_loader, 'test', model, 0, plot_test = True)
        saver.log_info((f'{loss_dict}'))
        saver.log_info(('Test loss: {:.7f}, encode loss: {:.7f}'.format(testr, encode_loss)))
    else:
        testr, loss_dict_test = test(test_loader, 'test', model, 0)
        saver.log_info(('Test loss: {:.3f}'.format(testr)))
    

def train_main(dataset, pragma_dim = None):
    saver.info(f'Reading dataset from {SAVE_DIR}')
    num_graphs = len(dataset)
    from torch.utils.data import random_split  # TODO: inductive
    #train_ratio:0.7 val_ratio:0.15 teat_ratio:0.15
    r1 = int(num_graphs * (1.0 - 2*(FLAGS.val_ratio)))
    r2 = int(num_graphs * (FLAGS.val_ratio))
    li = random_split(dataset, [r1, r2, len(dataset) - r1 - r2],
                          generator=torch.Generator().manual_seed(100))
    saver.log_info(f'{num_graphs} graphs in total:'
          f' {len(li[0])} train {len(li[1])} val '
          f'{len(li[2])} test')
    train_loader = DataLoader(li[0], batch_size=FLAGS.batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(li[1], batch_size=FLAGS.batch_size, pin_memory=True)  # TODO: split make sure no seen kernels in val/test
    test_loader = DataLoader(li[2], batch_size=FLAGS.batch_size, pin_memory=True)  # TODO

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = train_loader.dataset[0].num_features
    # graph-level embedding的shape [7, 1]
    edge_dim = train_loader.dataset[0].edge_attr.shape[1]
    print(edge_dim)

    if FLAGS.comparative_if:
        if FLAGS.comparative_model == 'HGP':
            model = HierNet(in_channels=FLAGS.num_features, hidden_channels=FLAGS.hidden_num, num_layers=3, conv_type='sage', drop_out=0.0)
        elif FLAGS.comparative_model == 'ironman':
            model = GCNNet(in_channels=FLAGS.num_features)
        else:
            max_degree = -1
            for data in li[2]:
                d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                max_degree = max(max_degree, int(d.max()))

            # compute the in-degree histogram tensor
            deg = torch.zeros(max_degree + 1, dtype=torch.long)
            for data in li[2]:
                d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                deg += torch.bincount(d, minlength=deg.numel())
            model = PNANet(in_dim=FLAGS.num_features, deg=deg, num_layer=2, emb_dim=200, edge_dim=edge_dim,
                           drop_ratio=0.5)

    else:
        model = Net(gumbel_args=gumbel_args, env_args=env_args, action_args=action_args).to(FLAGS.device)

    if FLAGS.model_path != None:
        model.load_state_dict(torch.load(FLAGS.model_path, map_location=torch.device(FLAGS.device)))
        saver.info(f'loaded model from {FLAGS.model_path}')
    print(model)

    #classification
    # if FLAGS.class_model_path != None:
    #     model.load_state_dict(torch.load(FLAGS.class_model_path, map_location=torch.device('cpu')))
    #     saver.info(f'loaded model from {FLAGS.class_model_path}')
    # print(model)
    # saver.log_model_architecture(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize to optimal attention weights:
    # model.pool1.weight.data = torch.tensor([0., 1., 0., 0.]).view(1,4).to(device)
    train_losses = []
    val_losses = []
    test_losses = []
    epochs = range(FLAGS.epoch_num)
    plot_test = False

    for epoch in epochs:
        plot_test = False
        timer = OurTimer()
        saver.log_info(f'Epoch {epoch + 1} train')
        loss, loss_dict_train = train(epoch, model, train_loader, optimizer)

        if len(val_loader) > 0:
            saver.log_info(f'\nEpoch {epoch + 1} val')
            val, loss_dict_val = test(val_loader, 'val', model, epoch)
            saver.writer.add_scalar('val/val', val, epoch)

        saver.log_info(f'\nEpoch {epoch + 1} test')
        testr, loss_dict_test = test(test_loader, 'test', model, epoch, plot_test, test_losses)
        saver.writer.add_scalar('test/test', testr, epoch)

        saver.log_info((f'\nTrain loss breakdown {loss_dict_train}'))
        saver.log_info((f'\nTest loss breakdown {loss_dict_test}'))
        if len(val_loader) > 0:
            saver.log_info((f'\nVal loss breakdown {loss_dict_val}'))
            saver.log_info(('Epoch: {:03d}, Train Loss: {:.4f}, Val loss: {:.4f}, '
                        'Test: {:.4f}) Time: {}'.format(
            epoch + 1, loss, val, testr, timer.time_and_clear())))
            val_losses.append(val)
        else:
            saver.log_info(('Epoch: {:03d}, Loss: {:.4f}, Train loss: {:.3f}, '
                        'Test: {:.3f}) Time: {}'.format(
            epoch + 1, loss, loss, testr, timer.time_and_clear())))

        train_losses.append(loss)
        test_losses.append(testr)

        if len(train_losses) > 50:
            if len(set(train_losses[-50:])) == 1 and len(set(test_losses[-50:])) == 1:
                break
        if FLAGS.task == 'regression':
            torch.save(model.state_dict(),
                       '/home/xxx/CoGNNs_LLMMH/save_models_and_data/regression_model_state_dict.pth')
        else:
            torch.save(model.state_dict(), '/home/xxx/CoGNNs_LLMMH/save_models_and_data/class_model_state_dict.pth')

    epochs = range(epoch+1)
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    if len(val_loader) > 0:
        plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.plot(epochs, test_losses, 'r', label='Testing loss')
    plt.title('Training, Validation, and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(saver.get_log_dir(), 'losses.png'), bbox_inches='tight')
    plt.show()
    saver.log_info(f'min test loss at epoch: {test_losses.index(min(test_losses)) + 1}')
    saver.log_info(f'min train loss at epoch: {train_losses.index(min(train_losses)) + 1}')
    if len(val_loader) > 0:
        saver.log_info(f'min val loss at epoch: {val_losses.index(min(val_losses)) + 1}')
    if FLAGS.task == 'regression':
        torch.save(model.state_dict(), '/home/xxx/CoGNNs_LLMMH/save_models_and_data/regression_model_state_dict.pth')
    else:
        torch.save(model.state_dict(), '/home/xxx/CoGNNs_LLMMH/save_models_and_data/class_model_state_dict.pth')



def train(epoch, model, train_loader, optimizer):
    model.train()

    total_loss = 0
    correct = 0
    i = 0
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]
    if FLAGS.task =='regression':
        target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    else:
        target_list = [_target_list[0]]
    loss_dict = {}
    for t in target_list:
        loss_dict[t] = 0.0
    for data in tqdm(train_loader, position=0, total=len(train_loader), file=sys.stdout):
        data = data.to(FLAGS.device)
        optimizer.zero_grad()
        out, loss, loss_dict_ = model.to(FLAGS.device)(data)
        loss.backward()
        if FLAGS.task == 'regression':
            total_loss += loss.item() * data.num_graphs
            for t in target_list:
                loss_dict[t] += loss_dict_[t].item()
        else:
            loss_, pred = torch.max(out[FLAGS.target[0]], 1)
            labels = _get_y_with_target(data, FLAGS.target[0])
            correct += (pred == labels).sum().item()
            total_loss += labels.size(0)
        optimizer.step()
        saver.writer.add_scalar('loss/loss', loss, epoch * len(train_loader) + i)
        # if i % FLAGS.print_every_iter == 0:
        #     print(f'Iter {i + 1}: Loss {loss}')
        i += 1
    if FLAGS.task == 'regression':    
        return total_loss / len(train_loader.dataset), {key: v / len(train_loader) for key, v in loss_dict.items()}
    else:
        return 1 - correct / total_loss, {key: v / len(train_loader) for key, v in loss_dict.items()}


def inference_loss_function(pred, true):
    return (pred - true) ** 2


def test(loader, tvt, model, epoch, plot_test = False, test_losses = [-1]):
    model.eval()

    inference_loss = 0
    correct, total = 0, 0
    loss_dict = {}
    i = 0
    points_dict = OrderedDict()
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]
    if FLAGS.task =='regression':
        target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    else:
        target_list = [_target_list[0]]
    
    for t in target_list:
        loss_dict[t] = 0.0
    for target_name in target_list:
        points_dict[target_name] = {'true': [], 'pred': []}
    for data in tqdm(loader, position=0, total=len(loader), file=sys.stdout):
        data = data.to(FLAGS.device)
        out_dict, loss, loss_dict_ = model.to(FLAGS.device)(data)

        if FLAGS.task == 'regression':
            total += loss.item()
            for t in target_list:
                loss_dict[t] += loss_dict_[t].item()
        else:
            loss, pred = torch.max(out_dict[FLAGS.target[0]], 1)
            labels = _get_y_with_target(data, FLAGS.target[0])
            correct += (pred == labels).sum().item()
            total += labels.size(0)
                
            
        for target_name in target_list:
            if FLAGS.subtask == 'inference':
                saver.info(f'{target_name}')
            if FLAGS.task == 'class':
                out = pred
            elif FLAGS.encode_log and 'perf' in target_name:
                out = out_dict['perf'] 
            else:
                out = out_dict[target_name]
            for i in range(len(out)):
                out_value = out[i].item()
                if FLAGS.encode_log and target_name == 'actual_perf':
                    out_value = 2**(out_value) * (1 / FLAGS.normalizer)
                if FLAGS.subtask == 'inference':
                    inference_loss += inference_loss_function(out_value, _get_y_with_target(data, target_name)[i].item())
                    if out_value != _get_y_with_target(data, target_name)[i].item():
                        saver.info(f'data {i} actual value: {_get_y_with_target(data, target_name)[i].item():.2f}, predicted value: {out_value:.2f}')
                points_dict[target_name]['pred'].append(
                    (_get_y_with_target(data, target_name)[i].item(), out_value))
                points_dict[target_name]['true'].append(
                    (_get_y_with_target(data, target_name)[i].item(),
                        _get_y_with_target(data, target_name)[i].item()))

        i += 1
    if FLAGS.plot_pred_points and tvt == 'test' and (plot_test or (test_losses and (total / len(loader)) < min(test_losses))):
        from utils import plot_points, plot_points_with_subplot
        saver.log_info(f'@@@ plot_pred_points')
        if not FLAGS.multi_target:
            plot_points({f'{FLAGS.target[0]}-pred_points': points_dict[f'{FLAGS.target[0]}']['pred'], f'{FLAGS.target[0]}-true_points': points_dict[f'{FLAGS.target[0]}']['true']},
                        f'epoch_{epoch+1}_{tvt}', saver.get_log_dir())
            print(f'done plotting with {correct} corrects out of {total}')
        else:
            assert(isinstance(FLAGS.target, list))
            plot_points_with_subplot(points_dict,
                        f'epoch_{epoch+1}_{tvt}', saver.get_log_dir(), target_list)
    if FLAGS.subtask == 'inference':
        if FLAGS.task == 'regression':
            result_df = _report_rmse_etc(points_dict, f'epoch {epoch}:', True)
        elif FLAGS.task == 'class':
            report_class_loss(points_dict)
    if FLAGS.task == 'regression':
        if FLAGS.subtask == 'inference':
            return (total / len(loader), {key: v / len(loader) for key, v in loss_dict.items()}, inference_loss / len(loader) / FLAGS.batch_size)
        else:
            return total / len(loader), {key: v / len(loader) for key, v in loss_dict.items()}
    else:
        return 1 - correct / total, {key: v / len(loader) for key, v in loss_dict.items()}
