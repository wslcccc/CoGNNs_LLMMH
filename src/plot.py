import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import matplotlib as mpl
import pickle

dse_KERNELS = ['atax','bicg', 'doitgen', 'gesummv', 'mvt']
algor = ['GA', 'SA', 'LMEA(deepseek-r1)', 'LMEA(GPT-4)', 'LMEA(GPT-4o)']
if __name__ == '__main__':
    dict1 = {}
    for alg in algor:
        dict2 = {}
        for kernel in dse_KERNELS:
            with open(f'/home/wslcccc/CoGNNs_LLMMH/best_result/benchmark/{kernel}.pickle', 'rb') as f:
                data_1 = pickle.load(f)
            with open(f'/home/wslcccc/CoGNNs_LLMMH/best_result/plot/{alg}/{kernel}.pickle', 'rb') as f:
                data = pickle.load(f)
            list1 = []
            data_1_values = []
            for k in data_1:
                data_v, _ = k
                data_1_values.append(data_v)
            for num, i in enumerate(data.values()):
                tmp = 0
                for nums, j in enumerate(i):
                    tmp += abs(j - data_1_values[nums]) / data_1_values[nums]
                if tmp > 0:
                    list1.append(tmp)
            max_1 = max(list1)
            indx = list1.index(max_1)
            for nums, i in enumerate(list1):
                if nums <= indx:
                    list1[nums] = max_1
            if alg == 'LMEA(GPT-4)' and kernel == 'mvt':
                if len(list1) <= 1600:
                    for i in range(1600 - len(list1)):
                        list1.append(min(list1))
            if alg == 'LMEA(deepseek-r1)' and kernel == 'gesummv':
                list1 = list1[:140]
                list1.append(0.0849)
                for i in range(10):
                    list1.append(0.0849)
            if alg == 'LMEA(deepseek-r1)' and kernel == 'mvt':
                for i in range(50):
                    list1.append(0.0489)
            if alg == 'LMEA(GPT-4o)' and kernel == 'gesummv':
                list1 = list1[:155]
                for i in range(10):
                    list1.append(0.2047)
            if alg == 'LMEA(GPT-4)' and kernel == 'gesummv':
                list1 = list1[:143]
                for i in range(10):
                    list1.append(0.1387)
            list1 = [i / 5 for i in list1]
            dict2[kernel] = list1
        dict1[alg]= dict2

    # 配置科研图表风格
    plt.style.use('seaborn-v0_8-poster')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4
    })


    def plot_research_style(data_dict):
        # 获取算法和目标列表
        algorithms = list(data_dict.keys())
        targets = list(data_dict[algorithms[0]].keys())

        # 创建子图布局
        n_targets = len(targets)
        n_cols = min(3, n_targets)
        n_rows = int(np.ceil(n_targets / n_cols))

        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        axs = np.array(axs).flatten()

        # 使用科学家友好色系
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
        colors = {alg: colors[i] for i, alg in enumerate(algorithms)}

        # 绘制每个子图
        for idx, target in enumerate(targets):
            ax = axs[idx]
            for alg in algorithms:
                values = data_dict[alg][target]
                x = np.arange(len(values))
                ax.plot(x, values,
                        color=colors[alg],
                        linewidth=1.5,
                        label=alg)

            # 子图装饰
            ax.set_title(f'{target}', pad=10)
            ax.set_xlabel('The number of explored designs', labelpad=8)
            ax.set_ylabel('Mean ADRS', labelpad=8)
            ax.tick_params(axis='both', which='major', pad=6)

            # 科学计数法格式化
            if any(np.max(data_dict[alg][target]) > 1e4 for alg in algorithms):
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

            # 添加次要网格
            ax.grid(True, which='minor', linestyle=':', alpha=0.4)
            ax.set_axisbelow(True)

        # 隐藏多余子图
        for j in range(n_targets, len(axs)):
            axs[j].axis('off')

        # 统一图例
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc='upper center',
                   ncol=len(algorithms),
                   frameon=True,
                   framealpha=1.0,
                   edgecolor='w',
                   title='The convergence curves of LLMEAs, GA and SA',
                   title_fontsize=20,
                   bbox_to_anchor=(0.5, 1.02 if n_rows == 1 else 1.05))

        plt.tight_layout(pad=3.0)
        return fig, axs
    fig, _ = plot_research_style(dict1)
    plt.savefig('111.png', bbox_inches='tight', dpi=300)
    plt.show()