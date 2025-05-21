import pickle


# MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
# poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen',
#                 'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv',
#                 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']
dse_KERNELS = ['bicg', 'gesummv', 'doitgen', 'atax', 'mvt']
if __name__ == '__main__':
    # kernel = 'aes'
    # with open(join(best_result_path , f'{kernel}.pickle'), 'rb') as f:
    #     data = pickle.load(f)
    #     print(list(data))
    res_1 = []
    dict1 = {}
    for kernel in dse_KERNELS:
        with open(f'/home/wslcccc/CoGNNs-LLMMH/best_result/ref/{kernel}.pickle', 'rb') as f:
            data = pickle.load(f)
            data_list = []
            for i in data:
                data_1, _ = i
                data_list.append(data_1)
            dict1[kernel] = data_list
    av1 = 0
    for kernel in dse_KERNELS:
        with open(f'/home/wslcccc/CoGNNs-LLMMH/best_result/LLMEA/deepseek-r1/{kernel}.pickle', 'rb') as f:
            data = pickle.load(f)
            tmp = []
            for i in data.values():
                l1 = i
                temp = 0
                beh = sum(dict1[kernel]) / len(dict1[kernel])
                if len(l1) == 0:
                    l1_a = 0
                else:
                    l1_a = sum(l1) / len(l1)
                temp += abs(l1_a - beh) / beh
                if l1_a >= beh:
                    tmp.append(0.0)
                else:
                    tmp.append(temp)
                tmp.sort(reverse=True)
            print(tmp)
            print(f'{kernel} avg：', sum(tmp) / len(tmp))
            print(f'{kernel} min: ', tmp[-1])
            av1 += tmp[-1]
    print(av1 / len(dse_KERNELS))
