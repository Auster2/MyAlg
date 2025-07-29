import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义目标函数和梯度
def f1(x1, s):
    return x1 + 1.5 * np.cos(0.5 * np.pi * s)

def f2(x1, s):
    return 1 - np.sqrt(x1) + np.sin(0.5 * np.pi * s)

def grad_f1_s(x1, s):
    return -1.5 * np.pi * np.sin(0.5 * np.pi * s) / 2  # f1 对 s 的梯度

def grad_f2_s(x1, s):
    return 0.5 * np.pi * np.cos(0.5 * np.pi * s)  # f2 对 s 的梯度

# 梯度下降算法
def gradient_descent(x1, learning_rate=0.01, max_iter=1000, tol=1e-6):
    s = np.random.uniform(0, 1)  # 初始化 s 在 [0, 1] 区间
    for _ in range(max_iter):
        grad_f1 = grad_f1_s(x1, s)
        grad_f2 = grad_f2_s(x1, s)

        # 更新 s，利用梯度下降法
        s -= learning_rate * (grad_f1 + grad_f2)

        # 保证 s 在 [0, 1] 范围内
        s = np.clip(s, 0, 1)

        # 检查是否收敛
        if abs(grad_f1 + grad_f2) < tol:
            break

    return s

# 遍历 x1 的值，找到最优解
results = []
for x1 in np.linspace(0, 1, 100):  # 遍历 x1 从 0 到 1
    optimal_s = gradient_descent(x1)
    optimal_f1 = f1(x1, optimal_s)
    optimal_f2 = f2(x1, optimal_s)
    results.append((x1, optimal_s, optimal_f1, optimal_f2))

# 输出结果
for res in results:
    print(f"x1 = {res[0]:.2f}, s = {res[1]:.2f}, f1 = {res[2]:.4f}, f2 = {res[3]:.4f}")

# 绘制结果
f_1 = [res[2] for res in results]
f_2 = [res[3] for res in results]
# 按f_1排序
sorted_indices = np.argsort(f_1)
f_1 = np.array(f_1)[sorted_indices]
f_2 = np.array(f_2)[sorted_indices]

x2 = [res[0] for res in results]
x3 = [res[0] for res in results]
s1 = [0.7 for _ in results]
s0 = [0.3 for _ in results]

f_11 = [f1(x, s) for x, s in zip(x2, s1)]
f_21 = [f2(x, s) for x, s in zip(x2, s1)]

f_10 = [f1(x, s) for x, s in zip(x3, s0)]
f_20 = [f2(x, s) for x, s in zip(x3, s0)]

plt.figure(figsize=(10, 6))
plt.plot(f_11, f_21, marker='o', linestyle='--', color='r', label='s=1', linewidth=2)
plt.plot(f_10, f_20, marker='o', linestyle='--', color='g', label='s=0', linewidth=2)
plt.plot(f_1, f_2, marker='x', linestyle='-', color='b')
plt.xlabel('Objective 1 (f1)')
plt.ylabel('Objective 2 (f2)')
plt.title('Objective Space Visualization')
plt.grid()
plt.show()