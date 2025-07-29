"""
在16个测试问题上运行提出的演化帕累托集学习（EPSL）方法。
"""


"""
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch
import schedulefree

from matplotlib import pyplot as plt

from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions

from problem import get_problem
import pandas as pd


# Das-Dennis递归生成参考方向的函数
def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)
            
# 生成Das-Dennis参考方向的函数
def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)
    
    
# 真实的HV值字典，用于计算目标值的参考
hv_value_true_dict = {'re21':0.888555388212816, 're22':0.762745852861501, 're23':1.1635525932784616, 're24':1.171256434424709, 're25':1.0811754149670405,
                      're31':1.330998828712509, 're32':1.3306416000738288, 're33':1.014313740464521, 're34': 1.0505616850845179,
                      're35':1.3032487679703135, 're36':1.0059624890382002, 're37':0.8471959081902014,
                      're41':0.8213476947317269, 're42':0.9223397435426084, 're61':1.183, 're91':0.0662,
                      'zdt1': 0.87661645, 'zdt2': 0.5432833299998329, 'zdt3': 1.331738735248482,
                      'f1':0.87661645, 'f2':0.87661645, 'f3':0.87661645, 'f4':0.87661645, 'f5':0.87661645, 
                      'syn':0.87661645}

# 测试问题列表
ins_list = ['re21', 're22', 're23','re24','re25', 're31', 're32', 're33','re34','re35', 're36','re37','re41', 're42', 're61', 're91']

# 只运行一个测试问题
ins_list = ['zdt1']

# 独立运行的次数
n_run = 1 


# PSL设置
# 学习步骤的数量
n_steps = 4000 
# 梯度估计的样本数
n_sample = 10 
n_pref_update = 10 # N 个偏好向量
sampling_method = 'Bernoulli-Shrinkage'

# 设备选择
device = 'cpu'

# -----------------------------------------------------------------------------


# EPSL模型类型
model_type = 'variable_shared_component_syn'  # 模型类型可选 ['normal', 'key_point']

# 根据模型类型导入不同的Pareto集模型
if model_type == 'normal':
    from model.model_stch import ParetoSetModel
    
if model_type == 'variable_shared_component_syn':  # 关系模型，用于SYN（例如'ins_list'中的'syn'）
    from model.model_syn_shared_component import ParetoSetModel 
    
if model_type == 'variable_shared_component_re_2obj':  # 关系模型，用于RE21（例如'ins_list'中的're21'）
    from model.model_2obj_shared_component import ParetoSetModel 
    
if model_type == 'variable_relation_re21':  # 关系模型，用于RE21（例如'ins_list'中的're21'）
    from model.model_2obj_variable_relation_re import ParetoSetModel 
    
if model_type == 'key_point':
    from model.model_keypoint import ParetoSetModel


# 初始化参考向量
r = np.linspace(start=0, stop=1,num=5)
ref_vec_test = torch.tensor(np.array([1-r, r])).T.to(device).float()

# 用于存储每个测试问题的HV差距
hv_gap_list = {}
hv_large_set_gap_list = {}

# 遍历所有测试问题
for test_ins in ins_list:
    print(test_ins)
    hv_gap_list[test_ins] = []
    hv_large_set_gap_list[test_ins] = []
    
    # 加载实际的帕累托前沿和理想点、最大点
    if test_ins in ['re21', 're22', 're23','re24','re25', 're31', 're32', 're33','re34','re35', 're36','re37','re41', 're42', 're61', 're91']:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ParetoFront/{test_ins}.dat')
        pf = np.loadtxt(file_path)
        ideal_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ideal_nadir_points/ideal_point_{test_ins}.dat'))
        nadir_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ideal_nadir_points/nadir_point_{test_ins}.dat'))
    else:
        ideal_point = np.zeros(2)
        nadir_point = np.ones(2)

    # 获取问题信息
    hv_value_true = hv_value_true_dict[test_ins]
    hv_all_value = np.zeros([n_run, n_steps])
    problem = get_problem(test_ins)
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    ref_point = problem.nadir_point
    ref_point = [1.1*x for x in ref_point]
    ref_point = torch.Tensor(ref_point).to(device)
    
    # 重复运行n_run次
    for run_iter in range(n_run): # 1
        
        # 初始化模型和优化器
        psmodel = ParetoSetModel(n_dim, n_obj)
        psmodel.to(device)
            
        # 初始化优化器
        optimizer = schedulefree.AdamWScheduleFree(psmodel.parameters(), lr=0.0025, warmup_steps = 10)
       
        z = torch.ones(n_obj).to(device) 
        
        # EPSL学习步骤
        for t_step in range(n_steps):
            psmodel.train()
            optimizer.train()
            
            sigma = 0.01  # 噪声标准差
            
            # 随机生成n_pref_update个偏好 (10个)
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha, n_pref_update)
            pref_vec = torch.tensor(pref).to(device).float() 
            # (10, 2)
            # 获取当前对应的解 (10, 4)
            x = psmodel(pref_vec)
           
            grad_es_list = []
            
            # 遍历每个偏好向量，计算梯度估计
            for k in range(pref_vec.shape[0]):
                
                # 根据不同的采样方法生成样本
                if sampling_method == 'Gaussian':
                    delta = torch.randn(n_sample, n_dim).to(device).double()
                    
                if sampling_method == 'Bernoulli':
                    delta = (torch.bernoulli(0.5 * torch.ones(n_sample, n_dim)) - 0.5) / 0.5 # u_k
                    delta = delta.to(device).double()
                    
                if sampling_method == 'Bernoulli-Shrinkage':
                    m = np.sqrt((n_sample + n_dim - 1) / (4 * n_sample))
                    delta = (torch.bernoulli(0.5 * torch.ones(n_sample, n_dim)) - 0.5) / m
                    delta = delta.to(device).double()
                
                # 计算扰动后的解
                # n_sample = 5
                x_plus_delta = x[k] + sigma * delta
                delta_plus_fixed = delta
              

                # 限制解的范围
                x_plus_delta[x_plus_delta > 1] = 1
                x_plus_delta[x_plus_delta < 0] = 0
                
                # 计算新的目标值
                # 这5个是用来估算梯度的
                value_plus_delta = problem.evaluate(x_plus_delta)
                # value_plus_delta shape is [5, 2] 有 5 个样本和 2 个目标
                ideal_point_tensor = torch.tensor(ideal_point).to(device)
                # 归一化
                value_plus_delta = (value_plus_delta - ideal_point_tensor) / (ref_point - ideal_point_tensor)
              
                # z 参考点
                z = torch.min(torch.cat((z.reshape(1, n_obj), value_plus_delta - 0.1)), axis = 0).values.data
                
                # STCH更新
                u = 0.1 
                #                                   1/ \lambda_k * |f(x + \delta) - z| / u
                #                                 sum( shape(2) * shape(5, 2) - shape(2) / shape(2)) = sum(shape(5, 2)) = shape(5)
                tch_value = u * torch.logsumexp((1/pref_vec[k]) * torch.abs(value_plus_delta - z) / u, axis=1)
                tch_value = tch_value.detach()
                
               # 计算 Tchebycheff 目标值的排名索引
                rank_idx = torch.argsort(tch_value)

                # 创建一个与 tch_value 形状相同的张量，并初始化为 1
                tch_value_rank = torch.ones(len(tch_value)).to(device)

                # 赋值排名 (-0.5 到 0.5 之间的等差数列)
                tch_value_rank[rank_idx] = torch.linspace(-0.5, 0.5, len(tch_value)).to(device)

                # 计算梯度估计值
                # 公式: 1 / (sigma * K) * Σ u_k * r_k
                grad_es_k = 1.0 / (n_sample * sigma) * torch.sum(
                    tch_value_rank.reshape(len(tch_value), 1) * delta_plus_fixed, axis=0
                )

                # 将当前计算的梯度存入列表
                # 10, 4
                grad_es_list.append(grad_es_k)

            # 堆叠所有采样点的梯度，形成最终的梯度估计
            # 10, 4
            grad_es = torch.stack(grad_es_list)

            # -----------------------------------------------
            # 使用基于梯度的优化算法更新 Pareto Set 模型
            # -----------------------------------------------

            # 清空优化器的梯度
            optimizer.zero_grad()

            # 计算损失函数的梯度，并进行反向传播
            psmodel(pref_vec).backward(grad_es)

            # 使用优化器更新模型参数
            optimizer.step()


        # -----------------------------------------------
        # 评估模型性能（禁用梯度计算，提高效率）
        # -----------------------------------------------

        psmodel.eval()
        optimizer.eval()
        # 临时禁用梯度计算
        with torch.no_grad():
            # 生成不同维度的偏好向量
            if n_obj == 2:
                pref_size = 100
                pref = np.stack([np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)]).T
                pref = torch.tensor(pref).to(device).float()
                
                # test
                pref_0 = np.array([0.5, 0.5]).T
                pref_0 = torch.tensor(pref_0).to(device).float()

            elif n_obj == 3:
                pref_size = 105
                pref = torch.tensor(das_dennis(13, 3)).to(device).float()

            elif n_obj == 4:
                pref_size = 120
                pref = torch.tensor(das_dennis(7, 4)).to(device).float()

            elif n_obj == 6:
                pref_size = 182
                pref = torch.tensor(get_reference_directions(
                    "multi-layer",
                    get_reference_directions("das-dennis", 6, n_partitions=4, scaling=1.0),
                    get_reference_directions("das-dennis", 6, n_partitions=3, scaling=0.5)
                )).to(device).float()

            elif n_obj == 9:
                pref_size = 90
                pref = torch.tensor(get_reference_directions(
                    "multi-layer",
                    get_reference_directions("das-dennis", 9, n_partitions=2, scaling=1.0),
                    get_reference_directions("das-dennis", 9, n_partitions=2, scaling=0.5)
                )).to(device).float()

            # 计算模型的解集
            sol = psmodel(pref)
            obj = problem.evaluate(sol)

            # 解析得到 Pareto Set (PS) 和 Pareto Front (PF)
            generated_ps = sol.cpu().numpy()
            generated_pf = obj.cpu().numpy()

            data = pd.DataFrame(generated_ps, columns=[f'x{i+1}' for i in range(n_dim)])
            data.to_csv(f'./results/zdt1_shareComponent_{n_steps}_generated_ps.csv', index=False)

            # -----------------------------------------------
            # 计算超体积 (HV, Hypervolume)
            # -----------------------------------------------

            # 归一化目标值，以便进行超体积计算
            results_F_norm = (generated_pf - ideal_point) / (nadir_point - ideal_point)

            # 设置超体积计算的参考点
            hv = HV(ref_point=np.array([1.1] * n_obj))

            # 计算超体积值
            hv_value = hv(results_F_norm)

            # 计算 HV 差值（真实值 - 计算值）
            hv_gap_value = hv_value_true - hv_value
            hv_gap_list[test_ins].append(hv_gap_value)

            # 输出 HV 差值
            print("hv_gap", "{:.2e}".format(hv_gap_value))

            # 输出当前实验的平均 HV 差值
            if run_iter == (n_run - 1):
                print("hv_gap_mean", "{:.2e}".format(np.mean(hv_gap_list[test_ins])))


            # -----------------------------------------------
            # 可视化结果（绘制 Pareto Front 和 EPSL 结果）
            # -----------------------------------------------

            if n_obj == 2:
                fig = plt.figure()
                
                # 绘制真实 Pareto Front
                if test_ins in ['re21', 're22', 're23', 're24', 're25', 're31', 're32', 're33', 're34', 're35', 're36', 're37']:
                    plt.scatter(pf[:, 0], pf[:, 1], c='k', marker='.', s=2, alpha=1, label='Pareto Front', zorder=2)

                # 绘制 EPSL 生成的 Pareto Front
                plt.plot(generated_pf[:, 0], generated_pf[:, 1], c='tomato', alpha=1, lw=5, label='EPSL', zorder=1)

                

                # 设置坐标轴标签
                plt.xlabel(r'$f_1(x)$', size=16)
                plt.ylabel(r'$f_2(x)$', size=16)

                # 设置图例
                handles = []
                pareto_front_label = plt.Line2D((0, 1), (0, 0), color='k', marker='o', linestyle='', label='Pareto Front')
                epsl_label = plt.Line2D((0, 1), (0, 0), color='tomato', marker='o', lw=2, label='EPSL')

                handles.extend([pareto_front_label, epsl_label])
                plt.legend(handles=handles, fontsize=14, scatterpoints=3, bbox_to_anchor=(1, 1))
                plt.grid()


            # -----------------------------------------------
            # 3D 可视化（适用于 3 维目标问题）
            # -----------------------------------------------

            if n_obj == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # 绘制 EPSL 生成的 Pareto Front
                ax.scatter(generated_pf[:, 0], generated_pf[:, 1], generated_pf[:, 2], c='tomato', s=10, label='STCH')

                # 计算坐标轴范围
                max_lim = np.max(generated_pf, axis=0)
                min_lim = np.min(generated_pf, axis=0)

                ax.set_xlim(min_lim[0], max_lim[0])
                ax.set_ylim(max_lim[1], min_lim[1])
                ax.set_zlim(min_lim[2], max_lim[2])

                # 设置坐标轴标签
                ax.set_xlabel(r'$f_1(x)$', size=12)
                ax.set_ylabel(r'$f_2(x)$', size=12)
                ax.set_zlabel(r'$f_3(x)$', size=12)

                # 设置图例
                plt.legend(loc=1, bbox_to_anchor=(1, 1))
            plt.savefig(f'./results/zdt1_shareComponent_{n_steps}.png')
        