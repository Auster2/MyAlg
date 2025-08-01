'''1. 计算值小于0的元素占总数的比例'''
# import numpy as np

# # 创建一个形状为 (5, 3, 3) 的小数组
# # x = np.xay([
# #     [[-1, 2, 3], [4, -5, 6], [7, 8, -9]],
# #     [[-10, -11, -12], [-13, -14, -15], [-16, -17, -18]],
# #     [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
# #     [[-28, -29, -30], [-31, -32, -33], [-34, -35, -36]],
# #     [[37, 38, 39], [40, 41, 42], [43, 44, 45]]
# # ])

# x = np.array([[-1, -2, -3], [4, -5, 6], [7, 8, -9]],)

# # 计算值小于0的元素数量
# count_negative = np.sum(x < 0)

# # 假设我们用小数组的总数来代替5000，这里总数为5 * 3 = 15
# total_elements = x.size

# # 计算这些元素占总数的比例
# proportion_negative = count_negative / total_elements

# print("值小于0的元素占总数的比例:", proportion_negative)

# # 计算每个5个子数组中，所有3个值都小于0的子数组数量
# count_all_negative = np.sum(np.all(x < 0, axis=1))

# # 计算这些子数组占5个的比例
# proportion_all_negative = count_all_negative / x.shape[0]

# print("所有3个值都小于0的子数组占5个的比例:", proportion_all_negative)

'''2. 自定义比较逻辑的排序和排名计算'''

# import numpy as np

# # 示例数组
# arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
# compare_arr1 = np.array([10, 5, 8, 5, 3, 1, 7, 2, 3, 6])  # 第一个用于排序的比较值数组
# compare_arr2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 第二个用于排序的比较值数组

# # 将三个数组组合成一个元组列表
# combined = list(zip(arr, compare_arr1, compare_arr2))

# # 根据自定义的逻辑对元组列表进行排序
# # 例如，我们按照 compare_arr1 的值升序排序，如果 compare_arr1 的值相同，则按照 compare_arr2 的值降序排序
# sorted_combined = sorted(combined, key=lambda x: (x[1], -x[2]))

# # 提取排序后的 arr 的值
# sorted_arr = np.array([x[0] for x in sorted_combined])

# # 获取排序后的索引
# sorted_indices = np.argsort([x[1] for x in combined], kind='stable')  # 使用 stable 排序以保持相同 compare_arr1 值的原始顺序

# # 计算排名
# ranks = np.argsort(sorted_indices) + 1  # 排名从1开始

# print("原始数组:", arr)
# print("排序后的数组:", sorted_arr)
# print("排名:", ranks)

# # [
# #     1 0 1
# #     7 0 1
# #     5 2 3
# #     7 2 4
# #     8 0 1
# # ]

'''np.max'''

# import numpy as np

# f = np.array([[[1, 2], [3, 8]], [[5, 4], [0, 4]]])

# x1 = np.max(f[:, :, 0])
# x2 = np.max(f[:, :, 1])

# print("max x1: ", x1)
# print("max x2: ", x2)

'''list to string'''

# same_idx = [1, 2]
# print(str(same_idx))

'''list /, max'''

# import numpy as np

# x = [[1, 2, 3], [1, 2, 3]]
# y = [[3, 4, 48], [4, 18, 99]]

# x = np.array(x)
# y = np.array(y)

# print(y / x)

# print(np.maximum(x, y))

'''evaluate'''

import torch
import numpy as np

def evaluate1(x):
    """计算目标函数值和约束违反量"""
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    
    f = np.zeros((x.shape[0], 2))
    
    g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (x.shape[1] - 1)
    f[:, 0] = x[:, 0]
    f[:, 1] = g * (1 - np.sqrt(x[:, 0] / g))

    return f

def evaluate2(x):
    n = x.shape[1]
    
    sum1 =  torch.sum(torch.stack([x[:,i+1] for i in range(n - 1)]), axis = 0)
    g = 1 + 9 * sum1 / (n - 1)
    
    f1 = x[:,0]
    f2 =  g * ( 1 - torch.sqrt(x[:,0]/g))   
        
    objs = torch.stack([f1,f2]).T
    
    return objs

# 比较两个输出格式
def compare_outputs(output1, output2):
    if isinstance(output1, np.ndarray) and isinstance(output2, np.ndarray):
        return np.array_equal(output1, output2)
    elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
        return torch.equal(output1, output2)
    else:
        return False
    
# 测试 evaluate 函数
x = np.array([[0.5, 0.2, 0.3],
                [0.1, 0.4, 0.5],
                [0.8, 0.1, 0.1]])
output1 = torch.tensor(evaluate1(x), dtype=torch.float32)
output2 = evaluate2(torch.tensor(x, dtype=torch.float32))
print("Output 1:", output1)
print("Output 2:", output2)
# 检查两个输出是否相同
print("Outputs are equal:", compare_outputs(output1, output2))
