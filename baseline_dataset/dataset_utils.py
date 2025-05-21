import os
import torch
import random
import numpy as np
from torch.utils.data import random_split
import torch.nn as nn
from torch import tensor

def create_act(act, num_parameters=None):
    if act == 'relu' or act == 'ReLU':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity' or act == 'None':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    if act == 'elu' or act == 'elu+1':
        return nn.ELU()
    else:
        raise ValueError('Unknown activation function {}'.format(act))

class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2,
                 hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(len(self.layer_channels) - 1)])))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        # model.store_layer_output(self, layer_inputs[-1])
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
        return layer_inputs[-1]

def _get_y_with_target(data, target):
    return getattr(data, target.replace('-', '_'))

def mse_loss(output, target):
    output = torch.log(abs(output) + 1)
    target = torch.log(abs(target) + 1)
    return torch.mean(torch.square(output - target))


def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target))


def mae_loss(output, target):
    return torch.mean(torch.abs(target - output))


def generate_dataset(dataset_dir, dataset_name_list, print_info=False):
    dataset_list = list()
    for ds in dataset_name_list:
        ds_path = os.path.join(dataset_dir, ds + '.pt')
        if os.path.isfile(ds_path):
            tem_data = torch.load(ds_path)
            dataset_list = dataset_list + tem_data
            if print_info:
                print(ds_path)
    for i in dataset_list:
        one_x = torch.ones(i.x.data.shape)
        i.x.data = torch.log(i.x.data + one_x)
        one_y = torch.ones(i.y.data.shape)
        i.y.data = torch.log(i.y.data + one_y)

    return dataset_list


def split_dataset(all_list, shuffle=True, seed=6666):
    first_10_y = []
    for i in all_list[0:10]:
        first_10_y.append(i.y)
    print("first ten train graphs Y before shuffle:", first_10_y)

    if shuffle and seed is not None:
        np.random.RandomState(seed=seed).shuffle(all_list)
        print("seed number:", seed)
    elif shuffle and seed is None:
        random.shuffle(all_list)
        print("seed number:", seed)

    first_10_y = []
    for i in all_list[0:10]:
        first_10_y.append(i.y)
    print("first ten train graphs Y after shuffle:", first_10_y)

    train_ds, test_ds, val_ds = random_split(all_list, [round(0.7 * len(all_list)), round(0.15 * len(all_list)), len(all_list) - round(0.7 * len(all_list)) - round(0.15 * len(all_list))],
                                     generator=torch.Generator().manual_seed(42))

    return train_ds, test_ds, val_ds
