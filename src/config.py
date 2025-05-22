from networkx.classes import neighbors
import torch
from src.utils import get_user, get_host
import argparse
from CoGNN.layers import ModelType
TARGETS = ['perf', 'quality', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF',
           'total-BRAM', 'total-DSP', 'total-LUT', 'total-FF']
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen',
                'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv',
                'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']


parser = argparse.ArgumentParser()

parser.add_argument('--model', default='simple')

dataset = 'programl'
parser.add_argument('--dataset', default=dataset)

benchmark = ['machsuite', 'poly']
parser.add_argument('--benchmarks', default=benchmark)

tag = 'whole-machsuite-poly'
parser.add_argument('--tag', default=tag)

# encoder_path = None
encoder_path = '/home/xxx/CoGNNs-LLMMH/save_models_and_data/encoders.klepto'
parser.add_argument('--encoder_path', default=encoder_path)

model_path = None
# model_path = '/home/xxx/CoGNNs-LLMMH/save_models_and_data/regression_model_state_dict.pth'
parser.add_argument('--model_path', default=model_path)

class_model_path = None
# class_model_path = '/home/xxx/CoGNNs-LLMMH/save_models_and_data/class_model_state_dict.pth'
parser.add_argument('--class_model_path', default=class_model_path)
parser.add_argument('--num_features', default=153)

# TASK = 'class'
TASK = 'regression'
parser.add_argument('--task', default=TASK)

SUBTASK = 'dse'
# SUBTASK = 'inference'
# SUBTASK = 'train'
parser.add_argument('--subtask', default=SUBTASK)
parser.add_argument('--val_ratio', type=float, default=0.15) # ratio of database for validation set

explorer = 'LLMEA'
# explorer = 'LLMSA'
# explorer = 'LLMACO'
# explorer = 'EA'
# explorer = 'SA'
# explorer = 'ACO'
# explorer = 'Exhastive'

parser.add_argument('--explorer', default=explorer)

model_tag = 'test'
parser.add_argument('--model_tag', default=model_tag)

parser.add_argument('--activation', default='elu')

parser.add_argument('--prune_util', default=True)
parser.add_argument('--prune_class', default=False)
# parser.add_argument('--prune_class', default=True)

parser.add_argument('--force_regen', type=bool, default=False)

parser.add_argument('--no_pragma', type=bool, default=False)

pids = ['__PARA__L3', '__PIPE__L2', '__PARA__L1', '__PIPE__L0', '__TILE__L2', '__TILE__L0', '__PARA__L2', '__PIPE__L0']
parser.add_argument('--ordered_pids', default=pids)

multi_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
target = 'perf'
parser.add_argument('--target', default=multi_target)

parser.add_argument('--separate_perf', type = bool, default=False )

parser.add_argument('--num_layers', type=int, default=6)

parser.add_argument('--encode_edge', type=bool, default=False)

parser.add_argument('--loss', type=str, default='RMSE')


EPSILON = 1e-3
parser.add_argument('--epsilon', default=EPSILON)
NORMALIZER = 1e7
parser.add_argument('--normalizer', default=NORMALIZER)
# MAX_NUMBER = 3464510.00
MAX_NUMBER = 1e10
parser.add_argument('--max_number', default=MAX_NUMBER)

norm = 'speedup-log2' # 'const' 'log2' 'speedup' 'off' 'speedup-const' 'const-log2' 'none' 'speedup-log2'
parser.add_argument('--norm_method', default=norm)

parser.add_argument('--invalid', type = bool, default=False ) # False: do not include invalid designs

parser.add_argument('--all_kernels', type = bool, default=True)

parser.add_argument('--multi_target', type = bool, default=True)

parser.add_argument('--save_model', type = bool, default=False)

parser.add_argument('--encode_log', type = bool, default=False)

parser.add_argument('--D', type=int, default=64)

batch_size = 64
parser.add_argument('--batch_size', type=int, default=batch_size)

epoch_num = 1000
parser.add_argument('--epoch_num', type=int, default=epoch_num)

gpu = 0
device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
# device = 'cpu'
parser.add_argument('--device', default=device)

parser.add_argument('--print_every_iter', type=int, default=100)

parser.add_argument('--plot_pred_points', type=bool, default=False)

best_result_path = '/best_result'
parser.add_argument('--best_result_path', type=str, default=best_result_path)

dse_unseen_kernel = ['bicg', 'doitgen', 'gesummv', '2mm']
parser.add_argument('--dse_unseen_kernel', type=list, default=dse_unseen_kernel)

out_dim = 1 if TASK == 'regression' else 2
parser.add_argument('--out_dim', type=int, default=out_dim)

#gumbel
parser.add_argument("--learn_temp", default=False)
parser.add_argument("--temp_model_type", dest="temp_model_type", default=ModelType.LIN,
                        type=ModelType.from_string, choices=list(ModelType))
parser.add_argument("--tau0", default=0.5, type=float)
parser.add_argument("--temp", default=0.01, type=float)

#enviroment
parser.add_argument("--env_model_type", default=ModelType.SUM_GNN,
                        type=ModelType.from_string, choices=list(ModelType))
parser.add_argument("--env_num_layers", default=3, type=int)
parser.add_argument("--env_dim", default=128, type=int)
parser.add_argument("--skip", default=False)
parser.add_argument("--batch_norm", default=False)
parser.add_argument("--layer_norm", default=False)
parser.add_argument("--dec_num_layers", default=1, type=int)
parser.add_argument("--dropout", default=0.2, type=float)

# policy cls parameters
parser.add_argument("--act_model_type", default=ModelType.MEAN_GNN,
                        type=ModelType.from_string, choices=list(ModelType))
parser.add_argument("--act_num_layers", default=2, type=int)
parser.add_argument("--act_dim", default=16, type=int)

# LLMMH parameters
# llm_model = "gpt-4o"
# llm_model = "gpt-4.1"
# llm_model = "o3-mini"
llm_model = "deepseek-r1"


parser.add_argument("--llm_model", default=llm_model, type=str)

open_ai_keys = "sk-xxxxx"


parser.add_argument("--api_key", default=open_ai_keys, type=str)

open_ai_base = "Api interface URL"


parser.add_argument("--api_base", default=open_ai_base, type=str)

temperature = 1.0
parser.add_argument("--temperature", default=temperature, type=int)

crossover_mutation_ratio = 0.02
parser.add_argument("--crossover_mutation_ratio", default=crossover_mutation_ratio, type=int)
stop_iteration_ratio = 0.3
parser.add_argument("--stop_iteration_ratio", default=stop_iteration_ratio, type=int)
content = '''|----Description of problem and solution properties----|
                You are addressing a multi-objective optimization problem. Your task is to find all the Pareto design configurations(solutions), that no other design configuration 
                in the search space has simultaneously less area and less latency than the Pareto configurations. Each design configuration is associated with two conflicting 
                objective values: area and latency. To find the Pareto optimal solution, you are now required to generate better solutions based on the following information.
                The possible values of pragma, the template of solution and the number of returned solutions are provided below. 
                In addition, fitness is the fitness of each method in the current population. 
                    pragmas_possible_values = {pragma_1:possible_values_1,...,pragma_k:possible_values_k}
                    solution = {pragma_1:values_1_1,...,pragma_k:values_k_1}
                    result_number = N
                    fitness = [fitness_1,fitness_2,...,fitness_n]
                |----In-context examples----|
                The examples of pragmas_possible_values and solutions are as follows:
                    pragmas_possible_values = {'__PIPE__L1': ['off', '', 'flatten'], '__PIPE__L2': ['off', '', 'flatten'], '__TILE__L2': [1, 2, 4, 8, 13]}
                    solution_1 = {'__PIPE__L1': '', '__PIPE__L2': '', '__TILE__L2': 1}
                    solution_2 = {'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 13}
                    solution_3 = {'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 4}
                |----Task instructions----|
                Please follow the instruction step-by-step to generate new design configurations(Please do not show any analysis or thinking process and just return the result):
                1. If the content of current_population is empty, randomly initialize the method based on the values of pragmas_possible_values and result_number.
                2. Otherwise, Generate potentially better configurations based on the current population and corresponding fitness.
                3. New solutions are generated based on the value of result_number and only the values of solutions are returned in the form of a list. For example:
                    (1)<start>['flatten', 'off', 13]<end>
                    (2)<start>['flatten', '', 1]<end>
                    (3)<start>['', '', 2]<end>
                    (4)<start>['', 'off', 4]<end>
                    (5)<start>['flatten', 'off', 8]<end> 
                4. Note that each list needs to use <start> and <end> to distinguish the beginning and the end of the list.
                '''
parser.add_argument("--content", default=content, type=str)

content1 =      '''
                You are addressing a multi-objective optimization problem. Your task is to find all the Pareto design configurations(solutions), that no other design configuration 
                in the search space has simultaneously less area and less latency than the Pareto configurations. Each design configuration is associated with two conflicting 
                objective values: area and latency. To find the Pareto optimal solution, you are now required to generate a better solution based on the following information.
                This task has the background of the aco algorithm, but you only need to generate better solutions based on the provided pheromone_matrix and refer to other information.
                The possible values of pragma, the template of solution and the number of returned solutions are provided below. 
                In addition, fitness is the fitness of each solution in the current population and pheromone_matrix.
                The form of the pheromone_matrix is a python dictionary, where each key corresponds to a pragma name, 
                and values represent the pheromone concentrations corresponding to possible values.
                    pragmas_possible_values = {pragma_1:possible_values_1,...,pragma_k:possible_values_k}
                    solution = {pragma_1:values_1_1,...,pragma_k:values_k_1}
                    result_number = N
                    fitness = [fitness_1,fitness_2,...,fitness_n]
                    pheromone_matrix = {pragma_1:[pc_1_1,...,pc_1_n],...,pragma_k:[pc_k_1,...,pc_k_m]}
                |----In-context examples----|
                The examples of pragmas_possible_values, solutions and pheromone_matrix are as follows:
                    pragmas_possible_values = {'__PIPE__L1': ['off', '', 'flatten'], '__PIPE__L2': ['off', '', 'flatten'], '__TILE__L2': [1, 2, 4, 8, 13]}
                    solution_1 = {'__PIPE__L1': '', '__PIPE__L2': '', '__TILE__L2': 1}
                    solution_2 = {'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 13}
                    solution_3 = {'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 4}
                    pheromone_matrix = {'__PIPE__L1': [0.5, 0.5, 1], '__PIPE__L2':[1, 0.5, 0.5], '__TILE__L2':[1, 0.95, 0.85，0.65，0.55]}
                |----Task instructions----|
                Please follow the instruction step-by-step to generate new design configurations(Please do not show any analysis or thinking process and just return the result.):
                1. If the content of current_population is empty, randomly initialize the method based on the values of pragmas_possible_values and result_number.
                2. Generate potentially better configurations based on the current population, corresponding fitness and pheromone_matrix(The higher the pheromone concentration, the greater the selection probability. 
                    However, to avoid local superiority, at least half of the new solutions need to be generated randomly. 
                3. New solutions are generated based on the value of result_number and only the values of solutions are returned in the form of a list. For example:
                    (1)<start>['flatten', 'off', 13]<end>
                    (2)<start>['flatten', '', 1]<end>
                    (3)<start>['', '', 2]<end>
                    (4)<start>['', 'off', 4]<end>
                    (5)<start>['flatten', 'off', 8]<end> 
                4. Note that each list needs to use <start> and <end> to distinguish the beginning and the end of the list.
                
                '''
parser.add_argument("--content1", default=content1, type=str)

# EA parameter
crossover_mutation_rate = 0.1
parser.add_argument("--crossover_mutation_rate", default=crossover_mutation_rate, type=int)
iter_stop_num = 0.1
parser.add_argument("--iter_stop_num", default=iter_stop_num, type=int)

# SA parameter
initial_temperature = 100
parser.add_argument("--initial_temperature", default=initial_temperature, type=int)
stop_temperature = 0.1
parser.add_argument("--stop_temperature", default=stop_temperature, type=int)
cooling_rate = 0.3
parser.add_argument("--cooling_rate", default=cooling_rate, type=int)
neighbor_distance_rate = 0.1
parser.add_argument("--neighbor_distance_rate", default=neighbor_distance_rate, type=int)


# comparative experiment parameter
comparative_if = False
parser.add_argument("--comparative_if", default=comparative_if, type=bool)
# comparative_model = "HGP"
# comparative_model = "ironman"
comparative_model = "pna"
parser.add_argument("--comparative_model", default=comparative_model, type=str)
parser.add_argument("--hidden_num", default=128, type=int)

# HGP dataset comparative experiment parameter
target_1 = ['lut', 'ff', 'dsp', 'bram', 'uram', 'srl', 'cp', 'power']
parser.add_argument("--target_1", default=target_1)
dataset_seen = ['aes', 'bfs', 'fft', 'gemm', 'md', 'nw']
parser.add_argument("--dataset_seen", default=dataset_seen)
dataset_unseen = ['sort', 'spmv', 'stencil', 'vitberbi']
parser.add_argument("--dataset_unseen", default=dataset_unseen)
"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

FLAGS = parser.parse_args()
