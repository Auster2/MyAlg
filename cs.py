import numpy as np
import matplotlib.pyplot as plt
from problem.prob import ZDT1
from problem.sc_prob import SharedComponents

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
 
class SubNSGA:
    def fast_non_dominated_sort(self, f, cv):
        """快速非支配排序（考虑约束）"""
        n_points = f.shape[0]
        S = [[] for _ in range(n_points)]  # 每个个体支配的解集合
        n = np.zeros(n_points)  # 支配每个个体的解的数量
        rank = np.zeros(n_points, dtype=int)  # 每个个体的支配等级
        
        # 计算每个解的约束违反总量
        cv_sum = np.sum(np.maximum(0, cv), axis=1)
        
        # 对每对个体比较支配关系
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # 约束支配关系:
                    # 1. 如果i可行且j不可行，则i支配j
                    # 2. 如果i和j都不可行，约束违反量更小的支配另一个
                    # 3. 如果i和j都可行，使用传统的Pareto支配关系
                    
                    if cv_sum[i] == 0 and cv_sum[j] > 0:
                        # i可行且j不可行
                        S[i].append(j)
                        n[j] += 1
                    elif cv_sum[i] > 0 and cv_sum[j] > 0:
                        # i和j都不可行
                        if cv_sum[i] < cv_sum[j]:
                            # i的约束违反量更小
                            S[i].append(j)
                            n[j] += 1
                        elif cv_sum[j] < cv_sum[i]:
                            # j的约束违反量更小
                            S[j].append(i)
                            n[i] += 1
                    elif cv_sum[i] == 0 and cv_sum[j] == 0:
                        # i和j都可行，使用传统Pareto支配
                        if np.all(f[i] <= f[j]) and np.any(f[i] < f[j]):
                            S[i].append(j)
                            n[j] += 1
                        elif np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
                            S[j].append(i)
                            n[i] += 1
        
        # 找出第一个front
        fronts = []
        current_front = []
        for i in range(n_points):
            if n[i] == 0:
                rank[i] = 0
                current_front.append(i)
        
        fronts.append(current_front)
        
        # 找出剩余的fronts
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for j in fronts[i]:
                for k in S[j]:
                    n[k] -= 1
                    if n[k] == 0:
                        rank[k] = i + 1
                        next_front.append(k)
            
            if next_front:
                fronts.append(next_front)
            i += 1
        
        return fronts, rank
    
    def crowding_distance(self, f, front, prob_obj):
        """计算拥挤度距离"""
        n_points = len(front)
        if n_points <= 2:
            return np.full(n_points, np.inf)
        
        dist = np.zeros(n_points)
        
        for obj in range(prob_obj):
            # 按目标函数值排序
            idx = np.argsort(f[front, obj])
            sorted_front = np.array(front)[idx]
            
            # 边界点拥挤度无穷大
            dist[0] = np.inf
            dist[-1] = np.inf
            
            if n_points > 2:
                # 计算其他点的拥挤度
                f_max = f[sorted_front[-1], obj]
                f_min = f[sorted_front[0], obj]
                
                if f_max == f_min:
                    continue
                
                # 计算中间点的拥挤度
                for i in range(1, n_points - 1):
                    dist[idx[i]] += (f[sorted_front[i+1], obj] - f[sorted_front[i-1], obj]) / (f_max - f_min)
        
        return dist
    
    def tournament_selection(self, rank, crowding_dist, pop_size, k=2):
        """锦标赛选择"""
        selected = np.zeros(pop_size, dtype=int)
        
        for i in range(pop_size):
            # 随机选择k个个体
            candidates = np.random.randint(0, len(rank), k)
            
            # 选择非支配等级更低的个体，如果相同则选择拥挤度更大的
            best = candidates[0]
            for j in range(1, k):
                if (rank[candidates[j]] < rank[best]) or \
                   (rank[candidates[j]] == rank[best] and crowding_dist[candidates[j]] > crowding_dist[best]):
                    best = candidates[j]
            
            selected[i] = best
        
        return selected
    
class NSGA:
    def __init__(self, problem, pop_size=100, sub_pop_size=100):
        self.problem = problem
        self.pop_size = pop_size
        self.sub_pop_size = sub_pop_size
        self.sub_alg = SubNSGA()
        self.history = {'x': [], 'f': [], 'cv': [], 'scv': []}
    
    def initialize_population(self):
        x = np.random.rand(self.pop_size, self.sub_pop_size, self.problem.sub_prob.n_var)
        x = self.problem.sub_prob.xl + x * (self.problem.sub_prob.xu - self.problem.sub_prob.xl)
        f, cv, scv = self.problem.evaluate(x)
        return x, f, cv, scv
    
    def sort(self, f, cv, scv):
        rank = np.array(range(len(f)))
        comb = list(zip(rank, f, cv, scv))
        """xxx"""
        # sorted_comb = sorted(comb, key=lambda x : (np.sum(x[3]), np.sum(x[2][:, 0]), np.sum(x[2][:, 1]), np.sum(x[1][:, 0]), np.sum(x[1][:, 1])))
        sorted_comb = sorted(comb, key=lambda x : np.sum(x[3]))
        sorted_rank = np.array([x[0] for x in sorted_comb])
        return sorted_rank
    
    def tournament_selection(self, rank, k=2):
        """锦标赛选择"""
        selected = np.zeros(self.pop_size, dtype=int)
        
        for i in range(self.pop_size):
            # 随机选择k个个体
            candidates = np.random.randint(0, len(rank), k)
            
            # 选择非支配等级更低的个体，如果相同则选择拥挤度更大的
            best = candidates[0]
            for j in range(1, k):
                if (rank[candidates[j]] < rank[best]):
                    best = candidates[j]
            
            selected[i] = best
        
        return selected
    
    def run(self, generations=500):
        x, f, cv, scv = self.initialize_population()
        for gen in range(generations):
            if gen % 10 == 0:
                print(f"Generation {gen}:")
                count_negative = np.sum(scv <= 1e-9)
                print(f"结构可行个体比例: {count_negative / (self.pop_size * self.sub_pop_size)}")
                count_all_negative = np.sum(np.all(scv <= 1e-9, axis=1))
                print(f"结构可行子种群比例: {count_all_negative / self.pop_size}")
            rank = self.sort(f, cv, scv)
            # print("排序后的个体:", rank)
            selected = self.tournament_selection(rank)
            # print("选择的个体:", selected)
            offspring_pop = x[selected]
            offspring_pop_scv = scv[selected]
            
            for i in range(self.pop_size):
                for j in range(0, self.sub_pop_size-1, 2):
                    if (offspring_pop_scv[i][j] < offspring_pop_scv[i][j + 1]):
                        offspring_pop[i][j+1] = offspring_pop[i][j] * 0.7 + offspring_pop[i][j+1] * 0.3
                    else:
                        offspring_pop[i][j] = offspring_pop[i][j] * 0.3 + offspring_pop[i][j+1] * 0.7
                        
            # 变异
            prob = np.random.rand(self.pop_size, self.sub_pop_size, self.problem.sub_prob.n_var)
            offspring_pop += (np.random.randint(0, 1) - 0.5) * prob * (self.problem.sub_prob.xu - self.problem.sub_prob.xl) * 0.1
            # 处理越界
            offspring_pop = np.clip(offspring_pop, self.problem.sub_prob.xl, self.problem.sub_prob.xu)
            
            
            offspring_f, offspring_cv, offspring_scv = self.problem.evaluate(offspring_pop)
            # 合并父代和子代
            combined_pop = np.concatenate((x, offspring_pop), axis=0)
            combined_f = np.concatenate((f, offspring_f), axis=0)
            combined_cv = np.concatenate((cv, offspring_cv), axis=0)
            combined_scv = np.concatenate((scv, offspring_scv), axis=0)
            
            # 重新排序
            combined_rank = self.sort(combined_f, combined_cv, combined_scv)
            # 选择前pop_size个个体
            selected_combined = combined_rank[:self.pop_size]
            x = combined_pop[selected_combined]
            f = combined_f[selected_combined]
            cv = combined_cv[selected_combined]
            scv = combined_scv[selected_combined]
            
            # 记录历史
            self.history['x'].append(x.copy())
            self.history['f'].append(f.copy())
            self.history['cv'].append(cv.copy())
            self.history['scv'].append(scv.copy())
            
        
        data = np.array(self.history['scv'])
        # 创建 3D 图像
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 遍历第一个维度（50 个点）
        for i in range(data.shape[0]):
            # 获取第 i 个 x 轴点的数据
            x = np.full(data.shape[1] * data.shape[2], i)  # x 轴值，重复 100×50 次
            y = np.tile(np.arange(data.shape[1]), data.shape[2])  # y 轴值，重复 50 次
            z = data[i, :, :].flatten()  # 将 100×50 的数据展平为 1D

            # 绘制散点图
            ax.scatter(x, y, z, alpha=0.5)

        # 设置坐标轴标签
        ax.set_xlabel('X axis (50 points)')
        ax.set_ylabel('Y axis (100 points)')
        ax.set_zlabel('Z axis (50 points)')
        plt.show()
        
        return f, cv, scv
        
if __name__ == "__main__":
    prob = SharedComponents(ZDT1, n_var=30, same_idx=[2, 3])
    nsga = NSGA(prob, pop_size=100, sub_pop_size=20)
    f, cv, scv = nsga.run(generations=50)
    print("最终种群: ", np.array(nsga.history['x'][-1]))
    print("目标函数值:", f)
    print("约束违反量:", cv)
    print("结构约束违反量:", scv)
    
"""
目标函数值: 
[[[0.22592531 6.13211199]
  [0.06304659 5.49067298]
  [0.81492631 2.75876339]
  [0.08813237 4.88040454]
  [0.50134098 4.92693418]]

 [[0.22592531 6.13211199]
  [0.06304659 5.49067298]
  [0.81492631 2.75876339]
  [0.08813237 4.88040454]
  [0.50134098 4.92693418]]

 [[0.22592531 6.13211199]
  [0.06304659 5.49067298]
  [0.81492631 2.75876339]
  [0.08813237 4.88040454]
  [0.50134098 4.92693418]]

 [[0.22592531 6.13211199]
  [0.06304659 5.49067298]
  [0.81492631 2.75876339]
  [0.08813237 4.88040454]
  [0.50134098 4.92693418]]

 [[0.22592531 6.13211199]
  [0.06304659 5.49067298]
  [0.81492631 2.75876339]
  [0.08813237 4.88040454]
  [0.50134098 4.92693418]]

 [[0.22592531 6.13211199]
  [0.06304659 5.49067298]
  [0.81492631 2.75876339]
  [0.08813237 4.88040454]
  [0.50134098 4.92693418]]

 [[0.22592531 6.13211199]
  [0.06304659 5.49067298]
  [0.81492631 2.75876339]
  [0.08813237 4.88040454]
  [0.50134098 4.92693418]]

 [[0.22592531 6.13211199]
  [0.06304659 5.49067298]
  [0.81492631 2.75876339]
  [0.08813237 4.88040454]
  [0.50134098 4.92693418]]

 [[0.22592531 6.13211199]
  [0.06304659 5.49067298]
  [0.81492631 2.75876339]
  [0.08813237 4.88040454]
  [0.50134098 4.92693418]]

 [[0.22592531 6.13211199]
  [0.06304659 5.49067298]
  [0.81492631 2.75876339]
  [0.08813237 4.88040454]
  [0.50134098 4.92693418]]]
约束违反量: 
[[[-0.85847974  5.1580373 ]
  [-0.4190591   4.35371957]
  [-0.68716367  2.3736897 ]
  [-0.35622762  3.7685369 ]
  [-0.50082627  4.22827516]]

 [[-0.85847974  5.1580373 ]
  [-0.4190591   4.35371957]
  [-0.68716367  2.3736897 ]
  [-0.35622762  3.7685369 ]
  [-0.50082627  4.22827516]]

 [[-0.85847974  5.1580373 ]
  [-0.4190591   4.35371957]
  [-0.68716367  2.3736897 ]
  [-0.35622762  3.7685369 ]
  [-0.50082627  4.22827516]]

 [[-0.85847974  5.1580373 ]
  [-0.4190591   4.35371957]
  [-0.68716367  2.3736897 ]
  [-0.35622762  3.7685369 ]
  [-0.50082627  4.22827516]]

 [[-0.85847974  5.1580373 ]
  [-0.4190591   4.35371957]
  [-0.68716367  2.3736897 ]
  [-0.35622762  3.7685369 ]
  [-0.50082627  4.22827516]]

 [[-0.85847974  5.1580373 ]
  [-0.4190591   4.35371957]
  [-0.68716367  2.3736897 ]
  [-0.35622762  3.7685369 ]
  [-0.50082627  4.22827516]]

 [[-0.85847974  5.1580373 ]
  [-0.4190591   4.35371957]
  [-0.68716367  2.3736897 ]
  [-0.35622762  3.7685369 ]
  [-0.50082627  4.22827516]]

 [[-0.85847974  5.1580373 ]
  [-0.4190591   4.35371957]
  [-0.68716367  2.3736897 ]
  [-0.35622762  3.7685369 ]
  [-0.50082627  4.22827516]]

 [[-0.85847974  5.1580373 ]
  [-0.4190591   4.35371957]
  [-0.68716367  2.3736897 ]
  [-0.35622762  3.7685369 ]
  [-0.50082627  4.22827516]]

 [[-0.85847974  5.1580373 ]
  [-0.4190591   4.35371957]
  [-0.68716367  2.3736897 ]
  [-0.35622762  3.7685369 ]
  [-0.50082627  4.22827516]]]
结构约束违反量: 
[[1.42917276 1.51695001 1.47697899 1.30699811 2.72176765]
 [1.42917276 1.51695001 1.47697899 1.30699811 2.72176765]
 [1.42917276 1.51695001 1.47697899 1.30699811 2.72176765]
 [1.42917276 1.51695001 1.47697899 1.30699811 2.72176765]
 [1.42917276 1.51695001 1.47697899 1.30699811 2.72176765]
 [1.42917276 1.51695001 1.47697899 1.30699811 2.72176765]
 [1.42917276 1.51695001 1.47697899 1.30699811 2.72176765]
 [1.42917276 1.51695001 1.47697899 1.30699811 2.72176765]
 [1.42917276 1.51695001 1.47697899 1.30699811 2.72176765]
 [1.42917276 1.51695001 1.47697899 1.30699811 2.72176765]]
"""