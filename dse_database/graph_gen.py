######################################################################################
### generated the graph to be processed with GNN-DSE                               ###
### requires: ProGraML [ICML'21]                                                   ###
######################################################################################


import os
import networkx as nx
import json
from os.path import join, abspath, basename
from subprocess import Popen, PIPE
from collections import OrderedDict
from copy import deepcopy
import ast
from pprint import pprint
from shutil import copy
from glob import glob
import re

from utils import create_dir_if_not_exists, get_root_path

# PRJ_PATH = os.getenv('PRJ_PATH')
# if PRJ_PATH is None:
#     print('PRJ_PATH is not set. Terminating!')
#     sys.exit()

MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen', 
                'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv', 
                'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']
ALL_KERNEL = {'machsuite': MACHSUITE_KERNEL, 'poly': poly_KERNEL}

PRAGMA_POSITION = {'PIPELINE': 0, 'TILE': 2, 'PARALLEL': 1}
BENCHMARK = 'machsuite'
BENCHMARK = 'poly'
processed_gexf_folder = join(get_root_path(), f'programl/{BENCHMARK}/processed/')

def add_to_graph(g_nx, nodes, edges):
    g_nx.add_nodes_from(nodes)
    g_nx.add_edges_from(edges)
    
def copy_files(name, src, dest):
    '''
        copy the generated files to the project directory
        
        args:
            name: the kernel name
            src: the path to the files
            dest: where you want to copy the files
    '''
    gen_files = [f for f in sorted(glob(join(src, f'{name}.*')))]
    gen_files.append(join(src, f'{name}_pretty.json'))
    gen_files.append(join(src, f'ds_info.json'))
    for f in gen_files:
        if 'ds_info.json' in f:
            source_dest = join(os.getcwd(), BENCHMARK, 'config', f'{name}_ds_config.json')
            copy(f, source_dest)
            continue
        if f.endswith('.c') or f.endswith('.cpp'):
            new_f_name = basename(f).replace(f'{name}', f'{name}_kernel')
            source_dest = join(os.getcwd(), BENCHMARK, 'sources', new_f_name)
            copy(f, source_dest)
        copy(f, dest)
    
def read_json_graph(name, readable=True):
    '''
        reads a graph in json format as a netwrokx graph
        
        args:
            name: name of the json file/ kernel's name
            reaable: whether to store a readable format of the json file
            
        returns:
            g_nx: graph in networkx format
    '''
    filename = name + '.json'
    with open(filename) as f:
        js_graph=json.load(f)
    g_nx=nx.readwrite.json_graph.node_link_graph(js_graph)
    if readable:
        make_json_readable(name, js_graph)
    
    return g_nx

def make_json_readable(name, js_graph):
    '''
        gets a json file and beautifies it to make it readable
        
        args:
            name: kernel name
            js_graph: the graph in networkx format read from the json file
            
        writes:
            a readable json file with name {name}_pretty.json    
    '''
    filename = name + '_pretty.json'
    f_json=open((filename), "w+")
    json.dump(js_graph, f_json, indent=4, sort_keys=True)
    f_json.close()
    
def extract_function_names(c_code):
    '''
        extract the names of the function in c code along with their line number
        
        args:
            c_code: the c_code read with code.read()
            
        return:
            a list of tuples of (function name, line number)
    '''
    pattern = r'\b\w+\s+\w+\s*\([^)]*\)\s*{'
    function_matches = re.finditer(pattern, c_code)
    function_names = []
    for match in function_matches:
        function_name = match.group().split()[1]
        line_number = c_code.count('\n', 0, match.start()) + 1
        function_names.append((function_name.split('(')[0], line_number))
    return function_names
    

def get_tc_for_loop(for_loop_text):
    '''
        get trip count of the for loop
    '''
    comp = for_loop_text.split(';')[1].strip()
    delims = ['<=', '>=', '<', '>', '--'] 
    delim = None
    for d in delims:
        if d in comp:
            delim = d
            break
    if delim:
        if delim == '--':
            TC = 0
        else:
            TC = int(eval(comp.replace(" ", "").split(delim)[-1].strip()))
        return TC
    else:
        print(f'no comparison sign found in {for_loop_text}')
        raise RuntimeError()

def get_icmp(path, name, log=False):
    '''
        gets an llvm file and returns the icmp instructions of each for loop
        
        args:
            path: parent directory of the llvm file
            name: kernel name
                    
        returns:
            a dictionary corresponding to the icmp instructions: 
                {for loop id: [icmp instruction, for.cond line number, icmp line number]}
            number of for loops
    '''
    for_dict_llvm = OrderedDict() ## {function inst: {for loop id: [icmp instruction, for.cond line number, icmp line number]}} ## function inst is the LLVM-equivalent of function defintion starting with "define"
    f_llvm = open(join(path, f'{name}.ll'), 'r')
    lines_llvm = f_llvm.readlines()
    for_count_llvm, local_for_count_llvm = 0, 0
    func_inst = None
    for idx, line in enumerate(lines_llvm):
        if line.strip().startswith('define'):
            for_dict_llvm[line.strip()] = OrderedDict()
            func_inst = line.strip()
            local_for_count_llvm = 0
        elif line.strip().startswith('for.cond'):
            for_count_llvm += 1
            local_for_count_llvm += 1
            for idx2, line2 in enumerate(lines_llvm[idx+1:]):
                if line2.strip().startswith('for.body'):
                    print(f'Do you have the right LLVM code? no icmp instruction found for loop at line {idx}.')
                    raise RuntimeError()
                elif 'icmp' in line2.strip():
                    assert func_inst != None, 'no function scope found'
                    for_dict_llvm[func_inst][local_for_count_llvm] = [line2.strip(), idx, idx2 + idx + 1]
                    break
    if log:
        print(json.dumps(for_dict_llvm, indent=4))
    return for_dict_llvm, for_count_llvm


def get_pragmas_loops(path, name, EXT='c', log=False):
    '''
        gets a c kernel and returns the pragmas of each for loop
        
        args:
            path: parent directory of the kernel file
            name: kernel name
                    
        returns:
            a dictionary with each entry showing the for loop and its pragmas
                {for loop id: [for loop source code, [list of pragmas]]}
            number of for loops
    '''
    
    for_dict_source = OrderedDict() ## {function name: {for loop id: [for loop source code, [list of pragmas]]}}
    f_source = open(join(path, f'{name}.{EXT}'), 'r')
    lines_source = f_source.readlines()
    f_source.close()
    with open(join(path, f'{name}.{EXT}'), 'r') as f_source:
        function_names_list = extract_function_names(f_source.read())
    for_count_source, local_for_count_source = 0, 0
    pragma_zone = False
    for f_id, (f_name, idx_start) in enumerate(function_names_list):
        for_dict_source[f_name] = OrderedDict()
        local_for_count_source = 0
        last_line = -1
        if f_id + 1 < len(function_names_list): last_line = function_names_list[f_id+1][1]
        for idx_, line in enumerate(lines_source[idx_start:last_line]):
            idx = idx_ + idx_start
            line = line.strip()
            if not line or 'scop' in line: ## if it's a blank line or #pragma scop in it, skip it
                continue
            if line.startswith('for(') or line.startswith('for '):
                for_count_source += 1
                local_for_count_source += 1
            if pragma_zone:
                if ':' in line:
                    continue ## if it is a loop label, skip it
                if line.startswith('#pragma'):
                    pragma_list.append(line)
                elif line.startswith('for'):
                    for_dict_source[f_name][local_for_count_source] = [line.strip('{'), pragma_list]
                    pragma_zone = False
                else:
                    print(f'Do you have the right source code? expected either for loop or pragma at line {idx} but got {line}.')
                    raise RuntimeError()
            else:
                if line.startswith('#pragma') and not 'KERNEL' in line.upper():
                    pragma_list = [line]
                    pragma_zone = True
    
    if log:
        print(json.dumps(for_dict_source, indent=4))
        
    return for_dict_source, for_count_source


def create_pragma_nodes(g_nx, g_nx_nodes, for_dict_source, for_dict_llvm, log = True):
    '''
        creates nodes for each pragma to be added to the graph
        
        args:
            g_nx: the graph object
            g_nx_nodes: number of nodes of the graph object
            for_dict_source: the for loops along with their pragmas
            for_dict_llvm: the for loops along with their icmp instruction in llvm
                    
        returns:
            a list of nodes and a list of edges to be added to the graph
    '''
    new_nodes, new_edges = [], []
    new_node_id = g_nx_nodes
    for f_name, f_content in for_dict_source.items():
        if len(f_content) == 0: ## no pragma exists in this function
            continue
        llvm_content = [f for f in for_dict_llvm if f_name in f]
        assert len(llvm_content) == 1
        llvm_content = for_dict_llvm[llvm_content[0]]
        for for_loop_id, [for_loop_text, pragmas] in f_content.items():
            icmp_inst = llvm_content[for_loop_id][0]
            icmp_inst_ = icmp_inst.split('!dbg')[0] ## if it has line number, strip it
            split_icmp_inst = icmp_inst_.split(',')
            TC_id = -1
            for sp in split_icmp_inst[::-1]:
                if sp.strip() == '': ## sp is white space ''
                    TC_id -= 1
                else:
                    break
            TC_icmp = int(eval(split_icmp_inst[TC_id].strip()))
            TC_for = get_tc_for_loop(for_loop_text)
            assert TC_for == TC_icmp, f'trip count of loop {for_loop_text} did not match {icmp_inst}.'

            node_id, block_id, function_id = None, None, None
            for node, ndata in g_nx.nodes(data=True):
                if 'features' in ndata:
                    feat = ast.literal_eval(str(ndata['features']))
                    if icmp_inst == feat['full_text'][0]:
                        print(f"found {icmp_inst} with id {node}")
                        node_id = int(node)
                        block_id = int(ndata['block'])
                        function_id = int(ndata['function'])
                        break
            if not node_id:
                print(f'icmp instruction {icmp_inst} not found.')
                raise RuntimeError()
            
            for pragma in pragmas:
                p_dict = {}
                p_dict['type'] = 3
                p_dict['block'] = block_id
                p_dict['function'] = function_id
                p_dict['features'] = {'full_text': [pragma]}
                p_dict['text'] = pragma.split(' ')[2]
                new_nodes.append((new_node_id, p_dict))
                
                e_dict = {'flow': 3, 'position': PRAGMA_POSITION[p_dict['text'].upper()]}
                new_edges.append((node_id, new_node_id, e_dict))
                new_edges.append((new_node_id, node_id, e_dict))
                
                new_node_id += 1
    if log:        
        pprint(new_nodes)
        pprint(new_edges)
        
    return new_nodes, new_edges

def prune_redundant_nodes(g_new):
    while True:
        remove_nodes = set()
        for node in g_new.nodes():
            if len(list(g_new.neighbors(node))) == 0 or node is None:
                print(node)
                remove_nodes.add(node)
                remove_nodes.add(node)
        for node in remove_nodes:
            g_new.remove_node(node)
        if not remove_nodes:
            break
        
def process_graph(name, g):
    '''
        adjusts the node/edge attributes, removes redundant nodes, 
            and writes the final graph to be used by GNN-DSE
        
        args:
            name: kernel name
            dest: where to store the graph
    '''

    g_new = nx.MultiDiGraph()
    for node, ndata in g.nodes(data=True):
        attrs = deepcopy(ndata)
        if 'features' in ndata:
            feat = ndata['features']
            attrs['full_text'] = feat['full_text'][0]
            del attrs['features']
            
        g_new.add_node(node)
        nx.set_node_attributes(g_new, {node: attrs})

    edge_list = []  
    id = 0
    for nid1, nid2, edata in g.edges(data=True):
        edata['id'] = id
        edge_list.append((nid1, nid2, edata))
        id += 1
    g_new.add_edges_from(edge_list)

    prune_redundant_nodes(g_new)

    new_gexf_file = join(processed_gexf_folder, f'{name}_processed_result.gexf')
    print(len(g_new.nodes), len(g.nodes))
    print(len(g_new.edges), len(g.edges))
    nx.write_gexf(g_new, new_gexf_file)



def graph_generator(name, path, benchmark, generate_programl = False):
    """
        runs ProGraML [ICML'21] to generate the graph, adds the pragma nodes,
            processes the final graph to be accepted by GNN-DSE

        args:
            name: kernel name
            path: path to parent directory of the kernel file
            benchmark: [machsuite|poly]

        graph info:
            flow: 1 --> data
            flow: 2 --> call
            flow: 0 --> there is no flow in the definition of the edge. an example is where 2 store nodes are connected together
            flow: 3 --> pragma
            
            position: 0 --> pipeline
            position: 1 --> parallel
            position: 2 --> tile

            type: 0 --> instruction
            type: 1 --> variable
            type: 2 --> immediate
            type: 3 --> pragma
    """
    ## generate PrograML graph
    
    if generate_programl:
        p = Popen(f"{get_root_path()}/programl_script.sh {name} {path}", shell = True, stdout = PIPE)
        p.wait()
    
    ## convert it to networkx format
    g_nx = read_json_graph(join(path, name))
    g_nx_nodes, g_nx_edges = g_nx.number_of_nodes(), len(g_nx.edges)
    
    ## find for loops and icmp instructions in llvm code
    for_dict_llvm, for_count_llvm = get_icmp(path, name)
    
    ## find for loops and their pragmas in the C/C++ code
    for_dict_source, for_count_source = get_pragmas_loops(path, name)
    assert for_count_llvm == for_count_source, f'the number of for loops from the LLVM code and source code do not match ' \
                                               f'{for_count_llvm} in llvm vs {for_count_source} in the code'
    
    print(f'number of nodes: {g_nx_nodes} and number of edges: {g_nx_edges}')
    graph_path = join(path, name+'.gexf')
    nx.write_gexf(g_nx, graph_path)
    
    augment_graph = True
    if augment_graph:
        ## create pragma nodes and their edges
        new_nodes, new_edges = create_pragma_nodes(g_nx, g_nx_nodes, for_dict_source, for_dict_llvm)
        
        add_to_graph(g_nx, new_nodes, new_edges)
        print(f'number of new nodes: {g_nx.number_of_nodes()} and number of new edges: {len(g_nx.edges)}')
        process = True
        if process:
            process_graph(name, g_nx)

    copy_files_ = True
    if generate_programl: copy_files = True
    local = True # True: programl is run in the directories inside this project
    if copy_files_:
        if not local:
            dest = join(os.getcwd(), f'programl', benchmark, name)
            create_dir_if_not_exists(dest)
            copy_files(name, path, dest)
        else:
            dest = path
        
        


if __name__ == '__main__':
    for BENCHMARK in ['machsuite', 'poly']:
        processed_gexf_folder = join(get_root_path(), f'programl/{BENCHMARK}/processed/')
        create_dir_if_not_exists(processed_gexf_folder)
        for kernel in ALL_KERNEL[BENCHMARK]:
            print('####################')
            print('now processing', kernel)
            path = join(get_root_path(), f'programl/{BENCHMARK}/', f'{kernel}')
            graph_generator(kernel, path, BENCHMARK, generate_programl = False)
            print()
    
    # KERNEL = ['fdtd-2d', '3mm', 'jacobi-1d']

    # for kernel in KERNEL:
    #     ## modify the path below:
    #     path = join('merlin_prj', f'{kernel}', 'xilinx_dse', f'{kernel}')
    #     graph_generator(kernel, path, BENCHMARK)
        
    

    
    