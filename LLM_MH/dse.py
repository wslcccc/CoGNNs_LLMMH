

from src.config import FLAGS, temperature
from src.saver import saver
from datetime import datetime
from src.utils import MLP, load, get_save_path, argsort, get_root_path, get_src_path, \
    _get_y_with_target, _get_y
from src.programl_data import print_data_stats, _check_any_in_str, NON_OPT_PRAGMAS, WITH_VAR_PRAGMAS, \
    _in_between, _encode_edge_dict, _encode_edge_torch, _encode_X_torch, create_edge_index
from src.model import Net
from src.parameter import DesignSpace, DesignPoint, DesignParameter, get_default_point, topo_sort_param_ids, \
    compile_design_space, gen_key_from_design_point
from src.config_ds import build_config
from src.result import Result
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
import json
import os
from LLM_MH.LLM import llm_process_ec, llm_process_aco
from math import ceil, inf, exp, log10
from os.path import join
import time
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Deque, Dict, List, Optional, Set, Union, Generator, Any
import sys
import copy
import itertools
import networkx as nx
from collections import OrderedDict
from glob import glob
import pickle
from torch.nn import Sequential, Linear, ReLU
from typing import NamedTuple, Any, Callable
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from random import uniform, randint, random

SAVE_DIR = join(get_root_path(), f'save_models_and_data')
SAVE_DIR_CLASS = join(get_root_path(), f'save_models_and_data')
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

class GNNModel():
    def __init__(self, path, saver, multi_target=True, task='regression', num_layers=FLAGS.num_layers, D=FLAGS.D,
                 target=FLAGS.target, model_name=f'{FLAGS.model_tag}_model_state_dict.pth', encoder_name='encoders'):
        """
        >>> self.encoder.keys()
        dict_keys(['enc_ntype', 'enc_ptype', 'enc_itype', 'enc_ftype', 'enc_btype', 'enc_ftype_edge', 'enc_ptype_edge'])

        """
        model_name = f'{task}_model_state_dict.pth'
        self.log = saver
        self.path = path
        if task == 'regression':
            if FLAGS.model_path == None:
                self.model_path = join(self.path, model_name)
            else:
                self.model_path = FLAGS.model_path
        else:
            if FLAGS.class_model_path == None:
                self.model_path = join(self.path, model_name)
            else:
                self.model_path = FLAGS.class_model_path
        if FLAGS.encoder_path == None:
            self.encoder_path = join(self.path, encoder_name)
        else:
            self.encoder_path = FLAGS.encoder_path
        self.num_features = FLAGS.num_features  # 153
        self.model = Net(gumbel_args=gumbel_args, env_args=env_args, action_args=action_args).to(
            FLAGS.device)
        self.model.load_state_dict(torch.load(join(self.model_path), map_location=torch.device('cpu')))
        saver.info(f'loaded {self.model_path}')
        self.encoder = load(self.encoder_path)

    def encode_node(self, g, point: DesignPoint):  # prograML graph
        X_ntype = []  # node type <attribute id="3" title="type" type="long" />
        X_ptype = []  # pragma type
        X_numeric = []
        X_itype = []  # instruction type (text) <attribute id="2" title="text" type="string" />
        X_ftype = []  # function type <attribute id="1" title="function" type="long" />
        X_btype = []  # block type <attribute id="0" title="block" type="long" />

        for node, ndata in g.nodes(data=True):  # TODO: node ordering
            numeric = 0
            if 'full_text' in ndata and 'pragma' in ndata['full_text']:
                # print(ndata['content'])
                p_text = ndata['full_text'].rstrip()
                assert p_text[0:8] == '#pragma '
                p_text_type = p_text[8:].upper()

                if _check_any_in_str(NON_OPT_PRAGMAS, p_text_type):
                    p_text_type = 'None'
                else:
                    if _check_any_in_str(WITH_VAR_PRAGMAS, p_text_type):
                        # HLS DEPENDENCE VARIABLE=CSIYIY ARRAY INTER FALSE
                        # HLS DEPENDENCE VARIABLE=<> ARRAY INTER FALSE
                        t_li = p_text_type.split(' ')
                        for i in range(len(t_li)):
                            if 'VARIABLE=' in t_li[i]:
                                t_li[i] = 'VARIABLE=<>'
                            elif 'DEPTH=' in t_li[i]:
                                t_li[i] = 'DEPTH=<>'  # TODO: later add back
                            elif 'DIM=' in t_li[i]:
                                numeric = int(t_li[i][4:])
                                t_li[i] = 'DIM=<>'
                            elif 'LATENCY=' in t_li[i]:
                                numeric = int(t_li[i][8:])
                                t_li[i] = 'LATENCY=<>'
                        p_text_type = ' '.join(t_li)

                    if point is not None:
                        t_li = p_text_type.split(' ')
                        for i in range(len(t_li)):
                            if 'AUTO{' in t_li[i]:
                                # print(t_li[i])
                                auto_what = _in_between(t_li[i], '{', '}')
                                numeric = point[auto_what]
                                if type(numeric) is not int:
                                    t_li[i] = numeric
                                    numeric = 0  # TODO: ? '', 'off', 'flatten'
                                else:
                                    t_li[i] = 'AUTO{<>}'
                                break
                        p_text_type = ' '.join(t_li)
                    else:
                        assert 'AUTO' not in p_text_type
                    # t = ' '.join(t.split(' ')[0:2])
                # print('@@@@@', t)
                ptype = p_text_type
            else:
                ptype = 'None'

            X_ntype.append([ndata['type']])
            X_ptype.append([ptype])
            X_numeric.append([numeric])
            X_itype.append([ndata['text']])
            X_ftype.append([ndata['function']])
            X_btype.append([ndata['block']])

        node_dict = {'X_ntype': X_ntype, 'X_ptype': X_ptype,
                     'X_numeric': X_numeric, 'X_itype': X_itype,
                     'X_ftype': X_ftype, 'X_btype': X_btype}

        enc_ntype = self.encoder['enc_ntype']
        enc_ptype = self.encoder['enc_ptype']
        enc_itype = self.encoder['enc_itype']
        enc_ftype = self.encoder['enc_ftype']
        enc_btype = self.encoder['enc_btype']

        return _encode_X_torch(node_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)

    def encode_edge(self, g):
        edge_dict = _encode_edge_dict(g)
        enc_ptype_edge = self.encoder['enc_ptype_edge']
        enc_ftype_edge = self.encoder['enc_ftype_edge']

        return _encode_edge_torch(edge_dict, enc_ftype_edge, enc_ptype_edge)

    def perf_as_quality(self, new_result: Result) -> float:
        """Compute the quality of the point by (1 / latency).

        Args:
            new_result: The new result to be qualified.

        Returns:
            The quality value. Larger the better.
        """
        return 1.0 / new_result.perf

    def quantify_util(self, result: Result) -> float:
        """Quantify the resource utilization to a float number.

        util' = 5 * ceil(util / 5) for each util,
        area = sum(2^1(1/(1-util))) for each util'

        Args:
            result: The evaluation result.

        Returns:
            The quantified area value with the range (2*N) to infinite,
            where N is # of resources.
        """

        # Reduce the sensitivity to (100 / 5) = 20 intervals
        utils = [
        5 * ceil(max(0.0, u) * 100 / 5) / 100 for k, u in result.res_util.items()
            if k.startswith('util')
        ]
        # Compute the area
        res = sum([2.0 ** u for u in utils])
        return res

    def eff_as_quality(self, new_result: Result) -> float:
        """Compute the quality of the point by resource efficiency.

        Args:
            new_result: The new result to be qualified.
            ref_result: The reference result.

        Returns:
            The quality value (negative finite differnece). Larger the better.
        """
        area = sum([0.25 * u for k, u in new_result.res_util.items() if k.startswith('util')])
        return log10(abs(1 / (new_result.perf * area)) + 1)

    def test(self, loader, config, mode='regression'):
        self.model.eval()

        i = 0
        results: List[Result] = []
        target_list = FLAGS.target
        if not isinstance(FLAGS.target, list):
            target_list = [FLAGS.target]
        for data in loader:
            data = data.to(FLAGS.device)
            out_dict, loss, loss_dict = self.model(data)
            if mode == 'regression':
                for i in range(len(out_dict['perf'])):
                    curr_result = Result()
                    curr_result.point = data[i].point
                    for target_name in target_list:
                        out = out_dict[target_name]
                        out_value = out[i].item()
                        if target_name == 'perf':
                            curr_result.perf = out_value
                            if FLAGS.encode_log:
                                curr_result.actual_perf = 2 ** out_value
                            else:
                                curr_result.actual_perf = out_value
                        elif target_name in curr_result.res_util.keys():
                            curr_result.res_util[target_name] = out_value
                        else:
                            raise NotImplementedError()
                    quality = self.perf_as_quality(curr_result)
                    curr_result.area = 1 / self.quantify_util(curr_result)
                    curr_result.quality = max(quality, curr_result.area)
                    max_utils = config['max-util']
                    results.append(curr_result)
            elif mode == 'class':
                _, pred = torch.max(out_dict['perf'], 1)
                labels = _get_y_with_target(data, 'perf')
                # saver.debug(f'pred: {pred}, labels: {labels}')
                return (pred == labels)
            else:
                raise NotImplementedError()

        return results

class Explorer():
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True, prune_invalid=False):
        """Constructor.

        Args:
            ds: DesignSpace
        """
        self.run_dse = run_dse
        self.log = saver
        self.kernel_name = kernel_name
        self.config_path = join(path_kernel, f'{kernel_name}_ds_config.json')
        self.config = self.load_config()
        self.timeout = 60 * 60
        self.ds, self.ds_size = compile_design_space(
            self.config['design-space']['definition'],
            None,
            self.log)
        self.batch_size = 1
        # Status checking
        self.num_top_designs = 3
        self.key_perf_dict = OrderedDict()
        self.best_results_dict = {}
        self.best_result: Result = Result()
        self.explored_point = 0
        self.ordered_pids = self.topo_sort_param_ids(self.ds)
        self.GNNmodel = GNNModel(SAVE_DIR, self.log, multi_target=True, task='regression', num_layers=FLAGS.num_layers,
                                 D=FLAGS.D)
        self.best_save_results = {}
        if FLAGS.separate_perf:
            perf_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP']
            self.GNNmodel_perf = GNNModel(SAVE_DIR, self.log, multi_target=True, task='regression_perf', num_layers=8,
                                          D=64, target=perf_target)
        gexf_file = sorted([f for f in glob(path_graph + "/*") if f.endswith('.gexf') and kernel_name in f])
        # print(gexf_file, glob(path_graph))
        assert len(gexf_file) >= 1
        # self.graph_path = join(path_graph, f'{kernel_name}_processed_result.gexf')

        self.graph_path = join(path_graph, gexf_file[0])
        self.graph = nx.read_gexf(self.graph_path)
        self.prune_invalid = prune_invalid
        if self.prune_invalid:
            self.GNNmodel_valid = GNNModel(SAVE_DIR_CLASS, self.log, multi_target=False, task='class',
                                           num_layers=FLAGS.num_layers, D=FLAGS.D)
        if self.ds_size <= 500:
            self.result_number = 10
            self.stop_cond = ceil(0.5 * self.ds_size)
        elif 500 < self.ds_size <= 10000:
            self.result_number = 20
            self.stop_cond = ceil(0.3 * self.ds_size)
        elif 10000 < self.ds_size <= 100000:
            self.result_number = 30
            self.stop_cond = ceil(0.05 * self.ds_size)
        elif 100000 < self.ds_size <= 1e6:
            self.result_number = 30
            self.stop_cond = ceil(0.005 * self.ds_size)
        elif 1e6 < self.ds_size <= 1e7:
            self.result_number = 30
            self.stop_cond = ceil(0.0005 * self.ds_size)
        elif 1e7 < self.ds_size <= 1e8:
            self.result_number = 40
            self.stop_cond = ceil(0.00005 * self.ds_size)
        elif 1e8 < self.ds_size <= 1e9:
            self.result_number = 50
            self.stop_cond = ceil(0.000005 * self.ds_size)
        elif 1e9 < self.ds_size <= 1e10:
            self.result_number = 50
            self.stop_cond = ceil(0.0000005 * self.ds_size)
        elif 1e10 < self.ds_size <= 1e11:
            self.result_number = 50
            self.stop_cond = ceil(0.0000005 * self.ds_size)
        elif 1e11 < self.ds_size <= 1e12:
            self.result_number = 50
            self.stop_cond = ceil(0.00000005 * self.ds_size)
        elif 1e12 < self.ds_size <= 1e13:
            self.result_number = 50
            self.stop_cond = ceil(0.000000005 * self.ds_size)
        else:
            self.result_number = 60
            self.stop_cond = ceil(0.0000000005 * self.ds_size)

    def topo_sort_param_ids(self, space: DesignSpace) -> List[str]:
        return topo_sort_param_ids(space)

    def load_config(self) -> Dict[str, Any]:
        """Load the DSE configurations.

        Returns:
            A dictionary of configurations.
        """

        try:
            if not os.path.exists(self.config_path):
                self.log.error(('Config JSON file not found: %s', self.config_path))
                raise RuntimeError()

            self.log.info('Loading configurations')
            with open(self.config_path, 'r', errors='replace') as filep:
                try:
                    user_config = json.load(filep)
                except ValueError as err:
                    self.log.error(('Failed to load config: %s', str(err)))
                    raise RuntimeError()

            config = build_config(user_config, self.log)
            if config is None:
                self.log.error(('Config %s is invalid', self.config_path))
                raise RuntimeError()
        except RuntimeError:
            sys.exit(1)

        return config

    def apply_design_point(self, g, point: DesignPoint, mode='regression') -> Data:
        X = self.GNNmodel.encode_node(g, point)
        edge_attr = self.GNNmodel.encode_edge(g)
        edge_index = create_edge_index(g)

        d_node = dict()
        resources = ['BRAM', 'DSP', 'LUT', 'FF']
        keys = ['perf', 'actual_perf', 'quality']
        for r in resources:
            keys.append('util-' + r)
            keys.append('total-' + r)
        for key in keys:
            d_node[key] = 0
        if mode == 'class':  ## default: point is valid
            d_node['perf'] = 1

        if 'regression' in mode:
            data = Data(
                x=X,
                edge_index=edge_index,
                perf=d_node['perf'],
                actual_perf=d_node['actual_perf'],
                quality=d_node['quality'],
                util_BRAM=d_node['util-BRAM'],
                util_DSP=d_node['util-DSP'],
                util_LUT=d_node['util-LUT'],
                util_FF=d_node['util-FF'],
                total_BRAM=d_node['total-BRAM'],
                total_DSP=d_node['total-DSP'],
                total_LUT=d_node['total-LUT'],
                total_FF=d_node['total-FF'],
                point=point,
                edge_attr=edge_attr
            )
        elif mode == 'class':
            data = Data(
                x=X,
                edge_index=edge_index,
                perf=d_node['perf'],
                edge_attr=edge_attr,
                kernel=self.kernel_name
            )
        else:
            raise NotImplementedError()

        return data

    def update_best(self, result: Result):
        """Keep tracking the best result found in this explorer.

        Args:
            result: The new result to be checked.

        """
        # if result.valid and result.quality > self.best_result.quality:
        update_flag = False
        if 'speedup' in FLAGS.norm_method:
            REF = min
        else:
            REF = max
        if self.key_perf_dict:
            key_refs_perf = REF(self.key_perf_dict, key=(lambda key: self.key_perf_dict[key]))
            refs_perf = self.key_perf_dict[key_refs_perf]
        else:
            if REF == min:
                refs_perf = float(-inf)
            else:
                refs_perf = float(inf)
        point_key = gen_key_from_design_point(result.point)
        if point_key not in self.key_perf_dict and result.valid and REF(result.quality,
                                                                        refs_perf) != result.quality:  # if the new result is better than the references designs
            self.best_result = result
            self.log.info(('Found a better result at {}: Quality {:.1e}, Perf {:.1e}'.format(
                self.explored_point, result.quality, result.perf)))
            if len(self.key_perf_dict.keys()) >= self.num_top_designs:
                # replace maxmimum performance value
                key_refs_perf = REF(self.key_perf_dict, key=(lambda key: self.key_perf_dict[key]))
                self.best_results_dict.pop((self.key_perf_dict[key_refs_perf], key_refs_perf))
                self.key_perf_dict.pop(key_refs_perf)
            attrs = vars(result)
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
            self.key_perf_dict[point_key] = result.quality
            self.best_results_dict[(result.quality, point_key)] = result
            update_flag = True
        return update_flag
    def gen_options(self, point: DesignPoint, pid: str, default=False) -> List[Union[int, str]]:
        """Evaluate available options of the target design parameter.

        Args:
            point: The current design point.
            pid: The target design parameter ID.

        Returns:
            A list of available options.
        """
        if default:
            dep_values = {dep: point[dep].default for dep in self.ds[pid].deps}
        else:
            dep_values = {dep: point[dep] for dep in self.ds[pid].deps}
        dep_values = {dep: point[dep] for dep in self.ds[pid].deps}
        options = eval(self.ds[pid].option_expr, dep_values)
        if options is None:
            self.log.error(f'Failed to evaluate {self.ds[pid].option_expr} with dep {str(dep_values)}')
            print('Error: failed to manipulate design points')
            sys.exit(1)

        return options

    def get_order(self, point: DesignPoint, pid: str) -> int:
        """Evaluate the order of the current value.

        Args:
            point: The current design point.
            pid: The target design parameter ID.

        Returns:
            The order.
        """

        if not self.ds[pid].order:
            return 0

        order = eval(self.ds[pid].order['expr'], {self.ds[pid].order['var']: point[pid]})
        if order is None or not isinstance(order, int):
            self.log.warning(f'Failed to evaluate the order of {pid} with value {str(point[pid])}: {str(order)}')
            return 0

        return order

    def update_child(self, point: DesignPoint, pid: str) -> None:
        """Check values of affect parameters and update them in place if it is invalid.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.
        """

        pendings = [child for child in self.ds[pid].child if self.validate_value(point, child)]
        for child in pendings:
            self.update_child(point, child)

    def validate_point(self, point: DesignPoint) -> bool:
        """Check if the current point is valid and set it to the closest value if not.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.

        Returns:
            True if the value is changed.
        """

        changed = False
        for pid in point.keys():
            options = self.gen_options(point, pid)
            value = point[pid]
            if not options:  # All invalid (something not right), set to default
                self.log.warning(f'No valid options for {pid} with point {str(point)}')
                point[pid] = self.ds[pid].default
                changed = True
                continue

            if isinstance(value, int):
                # Note that we assume all options have the same type (int or str)
                cand = min(options, key=lambda x: abs(int(x) - int(value)))
                if cand != value:
                    point[pid] = cand
                    changed = True
                    continue

            if value not in options:
                point[pid] = self.ds[pid].default
                changed = True
                continue

        return changed

    def validate_value(self, point: DesignPoint, pid: str) -> bool:
        """Check if the current value is valid and set it to the closest value if not.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.

        Returns:
            True if the value is changed.
        """

        options = self.gen_options(point, pid)
        value = point[pid]
        if not options:  # All invalid (something not right), set to default
            self.log.warning(f'No valid options for {pid} with point {str(point)}')
            point[pid] = self.ds[pid].default
            return False

        if isinstance(value, int):
            # Note that we assume all options have the same type (int or str)
            cand = min(options, key=lambda x: abs(int(x) - int(value)))
            if cand != value:
                point[pid] = cand
                return True

        if value not in options:
            point[pid] = self.ds[pid].default
            return True
        return False

    def move_by(self, point: DesignPoint, pid: str, step: int = 1) -> int:
        """Move N steps of pid parameter's value in a design point in place.

        Args:
            point: The design point to be manipulated.
            pid: The target design parameter.
            step: The steps to move. Note that step can be positive or negatie,
                  but we will not move cirulatory even the step is too large.

        Returns:
            The actual move steps.
        """

        try:
            options = self.gen_options(point, pid)
            idx = options.index(point[pid])
        except (AttributeError, ValueError) as err:
            self.log.error(
                f'Fail to identify the index of value {point[pid]} of parameter {pid} at design point {str(point)}: {str(err)}')
            print('Error: failed to manipulate design points')
            sys.exit(1)

        target = idx + step
        if target >= len(options):
            target = len(options) - 1
        elif target < 0:
            target = 0

        if target != idx:
            point[pid] = options[target]
            self.update_child(point, pid)
        return target - idx
    def traverse(self, point: DesignPoint, idx: int) -> Generator[DesignPoint, None, None]:
        """DFS traverse the design space and yield leaf points.

        Args:
            point: The current design point.
            idx: The current manipulated parameter index.

        Returns:
            A resursive generator for traversing.
        """

        if idx == len(self.ordered_pids):
            # Finish a point
            yield point
        else:
            yield from self.traverse(point, idx + 1)

            # Manipulate idx-th point
            new_point = self.clone_point(point)
            while self.move_by(new_point, self.ordered_pids[idx]) == 1:
                yield from self.traverse(new_point, idx + 1)
                new_point = self.clone_point(new_point)

    @staticmethod
    def clone_point(point: DesignPoint) -> DesignPoint:
        return dict(point)

    def get_results(self, population: List[DesignPoint]) -> List[Result]:
        data_list = []
        for point in population:
            data_list.append(self.apply_design_point(self.graph, point))

        test_loader = DataLoader(data_list, batch_size=self.batch_size)  # TODO
        results = self.GNNmodel.test(test_loader, self.config['evaluate'], mode='regression')
        return results

    # Large language model-evolutionary computation part
    def get_config_dafault_options(self):

        defaults_dict = {key: self.ds[key].default for key in self.ordered_pids}
        config_options = {}
        config_cond = {}
        for key in self.ordered_pids:
            if self.ds[key].deps:
                value = self.ds[key].option_expr.split('if')
                config_options[key] = eval(value[0] + ']')
                config_cond[key] = value[1][:-1]
            else:
                config_options[key] = eval(self.ds[key].option_expr)
                config_cond[key] = ''
        return defaults_dict, config_options, config_cond

    def load_llm_process_ec(self, fitness, current_population, pragmas_possible_value, result_number, temperature):
        tokens = 0
        llm = ChatOpenAI(model=FLAGS.llm_model, temperature=temperature, openai_api_key=FLAGS.api_key,
                         openai_api_base=FLAGS.api_base, request_timeout=2000, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
        start_time = datetime.now()
        secs = (datetime.now() - start_time).total_seconds()
        if secs >= 60:
            secs = 60
            tokens = 0
        res, tokens = llm_process_ec(llm=llm, tokens=tokens, secs=secs, fitness=fitness, current_population=current_population,
                                     pragmas_possible_value=pragmas_possible_value,
                                     result_number=result_number)
        return res, tokens

    def transfer_res_to_config(self, res, logger, co):
        logger.info('starting to transfer res to config')
        res = res.split('\n')
        res = [i for i in res if i.find('[') >= 0 and i.find(']') >= 0]
        k = list(co.keys())
        v = list(co.values())
        res_1 = []
        if len(res) < 2:
            return []
        for r in res:
            flag = False
            d1 = {}
            # r_1 = eval((('[' + r.split('[')[1]).split(']')[0]) + ']')
            r_1 = eval((r[r.find('['):r.find(']') + 1]))
            if len(r_1) < len(k):
                continue
            for nums, i in enumerate(k):
                if r_1[nums] in v[nums]:
                    d1[i] = r_1[nums]
                else:
                    flag = True
                    break
            if flag:
                continue
            res_1.append(d1)
        return res_1

    def generate_all_solutions(self, default_dict, pragmas_possible_values, config_cond):
        # Extract the order of pragmas from default_dict
        pragma_order = list(default_dict.keys())
        # Get the possible values for each pragma in the correct order
        value_lists = [pragmas_possible_values[pragma] for pragma in pragma_order]
        # Generate all possible combinations of values
        all_combinations = itertools.product(*value_lists)
        solutions = []

        for combination in all_combinations:
            # Create a configuration dictionary from the combination
            config = dict(zip(pragma_order, combination))
            is_valid = True
            temp_dict = config
            for key in temp_dict.keys():
                if config_cond[key] != '':
                    cond = config_cond[key]
                    dep_list = self.ds[key].deps
                    x = temp_dict[key]
                    temp = cond
                    for dep in dep_list:
                        if type(temp_dict[dep]) == int:
                            temp = temp.replace(dep, str(temp_dict[dep]))
                        else:
                            temp = temp.replace(dep, f'\'{temp_dict[dep]}\'')
                    if eval(temp) == False:
                        is_valid = False
                        break
            if is_valid:
                solutions.append(config)

        return solutions



    def run(self) -> None:
        """The main function of the explorer to launch the search algorithm.

        Args:
            algo_name: The corresponding algorithm name for running this exploration.
            algo_config: The configurable values for the algorithm.
        """
        raise NotImplementedError()


class LMEAExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(LMEAExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join(f'/home/xxx/CoGNNs-LLMMH/best_result/LLMEA/{FLAGS.llm_model}', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    def run(self) -> None:
        timer = time.time()
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        best_population = []
        current_population = []
        # evolutionary algorithm body
        tu = 1.0
        osst_val = 0
        osst = ceil(0.1 * self.stop_cond)
        while (time.time() - timer) < self.timeout and self.explored_point <= self.stop_cond:
            if len(current_population):
                results = self.get_results(current_population)
                for r in results:
                    self.explored_point += 1
                    self.best_save_results[self.explored_point] = [i.quality for i in self.best_results_dict.values()]
                    if isinstance(r, Result):
                        attrs = vars(r)
                    flag = self.update_best(r)
                    if flag == True:
                        osst_val = 0
                    else:
                        osst_val += 1
                if osst_val >= osst and tu >= 0.1:
                    tu -= 0.1
                    osst_val = 0
                fitness_1 = [r.quality for r in results]
                population_list = [i.point for i in self.best_results_dict.values()]
                fitness = [i.quality for i in self.best_results_dict.values()]
                best_population = []
                for i in population_list:
                    for j in i.keys():
                        temp = i[j]
                        if torch.is_tensor(temp):
                            i[j] = int(temp)
                    best_population.append(i)
                res, tokens = self.load_llm_process_ec(fitness, best_population, config_options, self.result_number, tu)
            else:
                fitness = []
                res, tokens = self.load_llm_process_ec(fitness, current_population, config_options, self.result_number, tu)
            self.log.info(f'The LLM model successfully returns the result！！！')
            config = self.transfer_res_to_config(res, self.log, config_options)
            self.log.info(f'The LLM generates {len(config)} design points')
            current_population = []
            current_population += config
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print(f'LLM temperature {tu}')
            print('------------------------------------------------------')
        self.log.info(f'Explored {self.explored_point} points')

class LMSAExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(LMSAExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join(f'/home/xxx/CoGNNs-LLMMH/best_result/LLMSA/{FLAGS.llm_model}', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'runtime: {end_time- start_time}')

    def run(self) -> None:
        timer = time.time()
        stop_flag = 0
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        best_population = []
        current_population = []
        # evolutionary algorithm body
        tu = 1.0
        osst_val = 0
        osst = ceil(0.1 * self.stop_cond)
        t_1 = FLAGS.temperature
        while (time.time() - timer) < self.timeout and self.explored_point <= self.stop_cond or t_1 >= FLAGS.stop_temperature:
            if len(current_population):
                results = self.get_results(current_population)
                temp_dict = {}
                for r in results:
                    self.explored_point += 1
                    self.best_save_results[self.explored_point] = [i.quality for i in self.best_results_dict.values()]
                    point_key = gen_key_from_design_point(r.point)
                    temp_dict[(r.quality, point_key)] = r
                    flag = self.update_best(r)
                    if flag == True:
                        osst_val = 0
                    else:
                        osst_val += 1
                best_population = []
                population_list = [i.point for i in self.best_results_dict.values()]
                for i in population_list:
                    for j in i.keys():
                        temp = i[j]
                        if torch.is_tensor(temp):
                            i[j] = int(temp)
                    best_population.append(i)
                fitness_1 = [r.quality for r in results]
                fitness = [i.quality for i in self.best_results_dict.values()]
                fit_avg = sum(fitness) / len(fitness)
                fit_avg_1 = sum(fitness_1) / len(fitness_1)
                temp_list = sorted(temp_dict.items(), key=lambda b: b[0][0], reverse=True)[:3]
                temp_dict1 = {key: value for key, value in temp_list}
                dla = fit_avg - fit_avg_1
                if dla > 0:
                    prob = 1 - exp(dla / t_1)
                    if prob > random.random():
                        self.best_results_dict = temp_dict1
                elif fit_avg - fit_avg_1 < 0:
                        self.best_results_dict = temp_dict1
                if osst_val >= osst and tu >= 0.1:
                    tu -= 0.1
                    osst_val = 0
                res, tokens = self.load_llm_process_ec(fitness, best_population, config_options, self.result_number, tu)
            else:
                fitness = []
                res, tokens = self.load_llm_process_ec(fitness, current_population, config_options, self.result_number, tu)
            self.log.info(f'The LLM model successfully returns the result！！！')
            config = self.transfer_res_to_config(res, self.log, config_options)
            self.log.info(f'The LLM generates {len(config)} design points')
            current_population = []
            current_population += config
            if t_1 >= FLAGS.stop_temperature:
                t_1 /= (1 + FLAGS.cooling_rate)
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print(f'LLM temperature {tu}')
            print('------------------------------------------------------')
        self.log.info(f'Explored {self.explored_point} points')


import numpy as np



class LMACOExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(LMACOExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.defaults_dict, self.config_options, self.config_cond = self.get_config_dafault_options()
        self.batch_size = 1
        self.params = self.config_options
        self.param_names = list(self.config_options.keys())
        self.n_params = len(self.param_names)
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.01
        self.current_population = []
        self.best_population = []
        self.fitness_c = []
        self.fitness_b = []
        self.pheromone = {param: np.ones(len(values)) for param, values in self.params.items()}

        self.pareto_front = []
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join(f'/home/xxx/CoGNNs-LLMMH/best_result/LLMACO/{FLAGS.llm_model}', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    def load_llm_process_aco(self, fitness, current_population, pragmas_possible_value, result_number, temperature,
                             pheromone_matrix):
        tokens = 0
        llm = ChatOpenAI(model=FLAGS.llm_model, temperature=temperature, openai_api_key=FLAGS.api_key,
                         openai_api_base=FLAGS.api_base, request_timeout=2000, streaming=True,
                         callbacks=[StreamingStdOutCallbackHandler()])
        start_time = datetime.now()
        secs = (datetime.now() - start_time).total_seconds()
        if secs >= 60:
            secs = 60
            tokens = 0
        res, tokens = llm_process_aco(llm=llm, tokens=tokens, secs=secs, fitness=fitness,
                                      current_population=current_population,
                                      pragmas_possible_value=pragmas_possible_value,
                                      result_number=result_number, pheromone_matrix=pheromone_matrix)
        return res, tokens
    class Ant:
        def __init__(self, aco):
            self.aco = aco
            self.solution = {}
            self.fitness = 0
            self.violation = 0

    def run(self) -> None:
        osst_val = 0
        osst = ceil(0.1 * self.stop_cond)
        tu = 1.0
        current_population = []
        fitness_c = []
        while self.explored_point <= self.stop_cond:
            ants = [self.Ant(self) for _ in range(self.result_number)]
            temp_population = []
            temp_c = []
            res, tokens = self.load_llm_process_aco(self.fitness_b, self.best_population, self.params, self.result_number, tu, self.pheromone)
            print('\n')
            self.log.info(f'The LLM model successfully returns the result！！！')
            config = self.transfer_res_to_config(res, self.log, self.config_options)
            self.log.info(f'The LLM generates {len(config)} design points')
            if len(config) < len(ants):
                continue
            for nums, ant in enumerate(ants):
                fitness = [i.quality for i in self.best_results_dict.values()]
                self.best_save_results[self.explored_point] = fitness
                flag = False
                ant.solution = config[nums]
                result = self.get_results([ant.solution])
                for i in result:
                    self.explored_point += 1
                    flag = self.update_best(i)
                    if flag:
                        osst_val = 0
                    else:
                        osst_val += 1
                ant.fitness = result[0].quality
                temp_population.append(ant.solution)
                temp_c.append(ant.fitness)
            if osst_val >= osst and tu >= 0.1:
                tu -= 0.1
                osst_val = 0
            self.update_pareto_front(ants)
            flag = self.update_pheromone(ants)
            population_list = [i.point for i in self.best_results_dict.values()]
            self.best_population = []
            for i in population_list:
                for j in i.keys():
                    temp = i[j]
                    if torch.is_tensor(temp):
                        i[j] = int(temp)
                self.best_population.append(i)
            self.fitness_b = [i.quality for i in self.best_results_dict.values()]
            current_population = temp_population
            fitness_c = temp_c
            self.current_population = current_population
            self.fitness_c = fitness_c
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print(f'LLM temperature {tu}')
            print('------------------------------------------------------')

        self.log.info(f'Explored {self.explored_point} points')

    def calculate_probability(self, param):
        """计算参数选项的选择概率"""
        value = self.params[param]
        pheromone = self.pheromone[param]
        if type(value[0]) == str:
            values = [i + 1 for i in range(len(value))]
        else:
            values = value
        heuristic = np.array([1.0 / (v + 1e-6) for v in values])
        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        probabilities /= np.sum(probabilities)
        return probabilities

    def update_pheromone(self, ants: List[Ant]):
        """更新信息素（全局更新与挥发）"""
        for param in self.param_names:
            self.pheromone[param] *= (1 - self.rho)

        for ant in self.pareto_front:
            for nums, param in enumerate(self.param_names):
                if ant.solution[param] in self.params[param]:
                    idx = self.params[param].index(ant.solution[param])
                    self.pheromone[param][idx] += 0.1
                else:
                    return False


    def update_pareto_front(self, ants: List[Ant]):
        """更新帕累托前沿"""
        for ant in ants:
            if ant.violation > 0:
                continue
            is_pareto = True
            for front_sol in self.pareto_front[:]:
                if self.is_dominated(ant.fitness, front_sol.fitness):
                    self.pareto_front.remove(front_sol)
                elif self.is_dominated(front_sol.fitness, ant.fitness):
                    is_pareto = False
                    break
            if is_pareto:
                self.pareto_front.append(ant)

    def is_dominated(self, sol_a, sol_b) -> bool:
        """检查sol_a是否被sol_b支配"""
        better = False
        if sol_a > sol_b:
            return True
        else:
            return better
import random
class EAExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(EAExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join('/home/xxx/CoGNNs-LLMMH/best_result/EA', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')
    def run(self) -> None:
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        # init population
        population = []
        for i in range(self.result_number):
            init_solution = {}
            for key, value in config_options.items():
                init_solution[key] = value[randint(0, len(value) - 1)]
            population.append(init_solution)
        p_cs = 0.1
        p_mt = 0.1
        while self.explored_point <= self.stop_cond:
            # evaluate solution
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            results = self.get_results(population)
            for r in results:
                self.explored_point += 1
                fitness = [i.quality for i in self.best_results_dict.values()]
                self.best_save_results[self.explored_point] = fitness
                if isinstance(r, Result):
                    attrs = vars(r)
                    self.log.debug(f'Evaluating Design')
                    self.log.debug(', '.join("%s: %s" % item for item in attrs.items()))
                    flag = self.update_best(r)
            # generate population
            # selection
            fitness_1 = [i.quality for i in results]
            selected = []
            for _ in range(self.result_number):
                contestants = random.sample(population, k=3)
                inx = [population.index(i) for i in contestants]
                val_inx = [fitness_1[i] for i in inx]
                max_inx = fitness_1.index(max(val_inx))
                selected.append(population[max_inx])
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 >= len(selected):
                    break
                c1, c2 = {}, {}
                p1, p2 = selected[i], selected[i+1]
                for para in config_options.keys():
                    if random.random() < p_cs:
                        c1[para] = p2[para]
                        c2[para] = p1[para]
                    else:
                        c1[para] = p1[para]
                        c2[para] = p2[para]
                offspring.append(c1)
                offspring.append(c2)
            temp = {}
            population_1 = []
            for i in offspring:
                temp = i.copy()
                for j in config_options.keys():
                    if random.random() < p_mt:
                        temp[j] = config_options[j][randint(0, len(config_options[j]) - 1)]
                population_1.append(temp)
            population = population_1
        self.log.info(f'Explored {self.explored_point} points')

class SAExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(SAExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join('/home/xxx/CoGNNs-LLMMH/best_result/SA', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    def run(self) -> None:
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        # get total solutions
        temperature_1 = FLAGS.initial_temperature
        # init solution
        cand_solutions = []
        init_solution = {}
        for i in range(self.result_number):
            for key, value in config_options.items():
                init_solution[key] = value[randint(0, len(value) - 1)]
            cand_solutions.append(init_solution)
        config_len = {key:len(value) for key, value in config_options.items()}
        neighbor_dis = [ceil(value * FLAGS.neighbor_distance_rate) for value in config_len.values()]
        while self.explored_point <= self.stop_cond and temperature_1 >= FLAGS.stop_temperature:
            # evaluate solution
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            results = self.get_results(cand_solutions)
            for r in results:
                self.explored_point += 1
                fitness = [i.quality for i in self.best_results_dict.values()]
                self.best_save_results[self.explored_point] = fitness
                if isinstance(r, Result):
                    attrs = vars(r)
                    self.log.debug(f'Evaluating Design')
                    self.log.debug(', '.join("%s: %s" % item for item in attrs.items()))
                    flag = self.update_best(r)
            solution_list = [i.point for i in self.best_results_dict.values()]
            cand_values = [list(i)[0] for i in self.best_results_dict.keys()]
            best_solutions = []
            for i in solution_list:
                for j in i.keys():
                    temp = i[j]
                    if torch.is_tensor(temp):
                        i[j] = int(temp)
                best_solutions.append(i)
            temp_solution = copy.deepcopy(best_solutions)
            cand_solutions = []
            for i in range(self.result_number):
                for num, solution in enumerate(temp_solution):
                    new_solution = {}
                    for nums, (key, value) in enumerate(solution.items()):
                        config_inx = config_options[key]
                        inx = config_inx.index(value)
                        df = neighbor_dis[nums]
                        dis = randint(-df, df)
                        inx += dis
                        if 0 <= inx <= len(config_inx) - 1:
                            new_solution[key] = config_inx[inx]
                        else:
                            new_solution[key] = value
                    cur_val = cand_values[num]
                    new_val = self.get_results([new_solution])[0].quality
                    delta_val = cur_val - new_val
                    if delta_val < 0 or random.random() < exp(-delta_val/ temperature_1):
                        cand_solutions.append(new_solution)
            temperature_1 = temperature_1 / (1 + FLAGS.cooling_rate)
        self.log.info(f'Explored {self.explored_point} points')



class ACOExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(ACOExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.defaults_dict, self.config_options, self.config_cond = self.get_config_dafault_options()
        self.batch_size = 1
        self.params = self.config_options
        self.param_names = list(self.config_options.keys())
        self.n_params = len(self.param_names)
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.1

        self.pheromone = {param: np.ones(len(values)) for param, values in self.params.items()}

        self.pareto_front = []
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join('/home/xxx/CoGNNs-LLMMH/best_result/ACO', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    class Ant:
        def __init__(self, aco):
            self.aco = aco
            self.solution = {}
            self.fitness = 0
            self.violation = 0

        def construct_solution(self):
            if self.aco.explored_point == 0:
                init_solution = {}
                for key, value in self.aco.config_options.items():
                    init_solution[key] = value[0]
                self.solution = init_solution
            else:
                for key in self.aco.params.keys():
                    prob = self.aco.calculate_probability(key)
                    selected_idx = np.random.choice(len(prob), p=prob)
                    self.solution[key] = self.aco.params[key][selected_idx]

    def run(self) -> None:
        while self.explored_point <= self.stop_cond:
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            ants = [self.Ant(self) for _ in range(self.result_number)]
            solutions = []
            for ant in ants:
                ant.construct_solution()
                result = self.get_results([ant.solution])
                for i in result:
                    self.explored_point += 1
                    self.best_save_results[self.explored_point] = [i.quality for i in self.best_results_dict.values()]
                    self.update_best(i)
                ant.fitness = result[0].quality
            self.update_pareto_front(ants)
            self.update_pheromone(ants)
        self.log.info(f'Explored {self.explored_point} points')

    def calculate_probability(self, param):
        """计算参数选项的选择概率"""
        value = self.params[param]
        pheromone = self.pheromone[param]
        if type(value[0]) == str:
            values = [i + 1 for i in range(len(value))]
        else:
            values = value
        heuristic = np.array([1.0 / (v + 1e-6) for v in values])
        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        probabilities /= np.sum(probabilities)
        return probabilities

    def update_pheromone(self, ants: List[Ant]):
        for param in self.param_names:
            self.pheromone[param] *= (1 - self.rho)

        for ant in self.pareto_front:
            for param in self.param_names:
                idx = self.params[param].index(ant.solution[param])
                self.pheromone[param][idx] += ant.fitness

    def update_pareto_front(self, ants: List[Ant]):
        for ant in ants:
            if ant.violation > 0:
                continue  # 跳过违反约束的解
            is_pareto = True
            for front_sol in self.pareto_front[:]:
                if self.is_dominated(ant.fitness, front_sol.fitness):
                    self.pareto_front.remove(front_sol)
                elif self.is_dominated(front_sol.fitness, ant.fitness):
                    is_pareto = False
                    break
            if is_pareto:
                self.pareto_front.append(ant)

    def is_dominated(self, sol_a, sol_b) -> bool:
        better = False
        if sol_a > sol_b:
            return True
        else:
            return better

class ExhaustiveExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(ExhaustiveExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Done init')

        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join('/home/xxx/CoGNNs-LLMMH/best_result/ref', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))

    def gen(self) -> Generator[List[DesignPoint], Optional[Dict[str, Result]], None]:
        # pylint:disable=missing-docstring

        self.log.info('Launch exhaustive search algorithm')

        traverser = self.traverse(get_default_point(self.ds), 0)
        iter_cnt = 0
        while True:
            next_points: List[DesignPoint] = []
            try:
                iter_cnt += 1
                self.log.debug(f'Iteration {iter_cnt}')
                while len(next_points) < self.batch_size:
                    next_points.append(next(traverser))
                    self.log.debug(f'Next point: {str(next_points[-1])}')
                yield next_points
            except StopIteration:
                if next_points:
                    yield next_points
                break

        self.log.info('No more points to be explored, stop.')

    def run(self) -> None:
        # pylint:disable=missing-docstring

        # Create a search algorithm generator
        gen_next = self.gen()

        timer = time.time()
        duplicated_iters = 0
        while (time.time() - timer) < self.timeout:
            try:
                # Generate the next set of design points
                next_points = next(gen_next)
                self.log.debug(f'The algorithm generates {len(next_points)} design points')
            except StopIteration:
                break

            results = self.get_results(next_points)
            for r in results:
                if isinstance(r, Result):
                    attrs = vars(r)
                    self.log.debug(f'Evaluating Design')
                    self.log.debug(', '.join("%s: %s" % item for item in attrs.items()))
                    self.update_best(r)
            self.explored_point += len(results)

        self.log.info(f'Explored {self.explored_point} points')