import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体

def save_history_array(history, key, filename):
    """
    将 nsga.history 中的某个键的最后一代数据保存为 CSV 文件。

    :param history: nsga.history 字典
    :param key: 要保存的数据键，如 "x"、"f"、"cv"、"scv"
    """
    if key not in history:
        raise ValueError(f"'{key}' not found in history")

    array = np.array(history[key][-1])
    array = array.reshape(-1, array.shape[-1]) if array.ndim > 1 else array.reshape(-1, 1)

    # 列名自动设置
    n_cols = array.shape[1]
    columns = [f"{key}{i}" for i in range(n_cols)]

    df = pd.DataFrame(array, columns=columns)
    df.to_csv(filename, index=False)

def plot_objectives(f, pf=None, title="Objective Visualization", filename=None):
    """
    根据目标维度自动绘图：
    - 2D：普通二维图
    - 3D：三维图
    - ≥4D：平行坐标图
    :param f: 当前目标值数组 (n_subpop, n_ind, n_obj)
    :param pf: 可选，Pareto 前沿数组 (n_ref, n_obj)
    :param title: 图标题
    :param filename: 保存路径（可选）
    """
    n_obj = f[0].shape[1]  # 目标维度
    fig = plt.figure(figsize=(10, 6))

    if n_obj == 2:
        # 2D 绘图
        if pf is not None:
            plt.plot(pf[:, 0], pf[:, 1], label='True Pareto Front', linewidth=2)
        plt.plot(f[0][:, 0], f[0][:, 1], linewidth=1, alpha=0.6)
        for sub_f in f:
            plt.plot(sub_f[:, 0], sub_f[:, 1], linewidth=0.5, alpha=0.6)
        plt.xlabel("f1")
        plt.ylabel("f2")

    elif n_obj == 3:
        # 3D 绘图
        ax = fig.add_subplot(111, projection='3d')
        if pf is not None:
            ax.plot(pf[:, 0], pf[:, 1], pf[:, 2], label='True Pareto Front', linewidth=2)
        for sub_f in f:
            ax.plot(sub_f[:, 0], sub_f[:, 1], sub_f[:, 2], linewidth=0.5, alpha=0.6)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")

    else:
        # 高维绘图 - 平行坐标图
        _plot_parallel_coordinates(f, title=title)
        return  # 已经内部 show/save

    plt.title(title)
    plt.grid(True)
    plt.legend()
    if filename:
        plt.savefig(filename)
    plt.show()


def _plot_parallel_coordinates(f, title="Parallel Coordinates", filename=None, alpha=0.3):
    """
    高维目标的平行坐标绘图（内部调用）
    """
    f_all = np.concatenate(f, axis=0)
    n_obj = f_all.shape[1]
    n_ind = f_all.shape[0]

    f_norm = (f_all - f_all.min(axis=0)) / (f_all.ptp(axis=0) + 1e-12)

    plt.figure(figsize=(12, 6))
    for i in range(n_ind):
        plt.plot(range(n_obj), f_norm[i, :], alpha=alpha)

    plt.xticks(range(n_obj), [f"f{i+1}" for i in range(n_obj)])
    plt.ylabel("Normalized Objective Value")
    plt.title(title)
    plt.grid(True)

    if filename:
        plt.savefig(filename)
    plt.show()


def generate_filename(problem_class, wrapper_class=None, generations=100, pop_size=50,
                      sub_pop_size=10, n_var=30, x_idx=None, y_idx=None,
                      prefix="", suffix="", folder="data"):
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prob_name = problem_class.__name__
    wrapper_name = wrapper_class.__name__ if wrapper_class else "NoWrapper"
    
    x_idx_str = str(x_idx if x_idx is not None else [])
    y_idx_str = str(y_idx if y_idx is not None else [])
    
    filename = f"{prefix}{prob_name}_{wrapper_name}_{generations}_{pop_size}_{sub_pop_size}_{n_var}_{x_idx_str}_{y_idx_str}_{time_str}{suffix}"
    
    return os.path.join(folder, filename)


def plot_scv_3d(history_scv):
    """
    绘制结构约束违反量 history['scv'] 的 3D 散点图。
    """
    data = np.array(history_scv)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(data.shape[0]):
        x = np.full(data.shape[1] * data.shape[2], i)  # 第 i 代
        y = np.tile(np.arange(data.shape[1]), data.shape[2])  # 各子种群/个体编号
        z = data[i, :, :].flatten()
        ax.scatter(x, y, z, alpha=0.5)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Sub-population Index')
    ax.set_zlabel('SCV Value')
    plt.title('3D View of Structural Constraint Violations')
    plt.tight_layout()
    plt.show()
 