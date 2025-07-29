import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV

class SubNSGA:
    """子种群NSGA算法类，处理子种群内的非支配排序"""
    
    def fast_non_dominated_sort(self, f, cv):
        """快速非支配排序（考虑约束）"""
        n_points = f.shape[0]
        S = [[] for _ in range(n_points)]
        n = np.zeros(n_points)
        rank = np.zeros(n_points, dtype=int)
        
        # 计算每个解的约束违反总量
        cv_sum = np.sum(np.maximum(0, cv), axis=1)
        
        # 对每对个体比较支配关系
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    if cv_sum[i] == 0 and cv_sum[j] > 0:
                        # i可行且j不可行
                        S[i].append(j)
                        n[j] += 1
                    elif cv_sum[i] > 0 and cv_sum[j] > 0:
                        # i和j都不可行
                        if cv_sum[i] < cv_sum[j]:
                            S[i].append(j)
                            n[j] += 1
                        elif cv_sum[j] < cv_sum[i]:
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
        
        # 构建front层级
        fronts = []
        current_front = [i for i in range(n_points) if n[i] == 0]
        for i in current_front:
            rank[i] = 0
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
            idx = np.argsort(f[front, obj])
            sorted_front = np.array(front)[idx]
            
            # 边界点拥挤度无穷大
            dist[0] = np.inf
            dist[-1] = np.inf
            
            if n_points > 2:
                f_max = f[sorted_front[-1], obj]
                f_min = f[sorted_front[0], obj]
                
                if f_max == f_min:
                    continue
                
                # 计算中间点的拥挤度
                for i in range(1, n_points - 1):
                    dist[idx[i]] += (f[sorted_front[i+1], obj] - f[sorted_front[i-1], obj]) / (f_max - f_min)
        
        return dist


class BaseNSGA:
    """NSGA算法基类，定义通用框架和接口"""
    
    def __init__(self, problem, pop_size=100, sub_pop_size=100):
        self.problem = problem
        self.pop_size = pop_size
        self.sub_pop_size = sub_pop_size
        self.sub_alg = SubNSGA()
        self.history = {'x': [], 'f': [], 'cv': [], 'scv': [], 'hv': []}
    
    def initialize_population(self):
        """初始化种群"""
        x = np.random.rand(self.pop_size, self.sub_pop_size, self.problem.sub_prob.n_var)
        x = self.problem.sub_prob.xl + x * (self.problem.sub_prob.xu - self.problem.sub_prob.xl)
        f, cv, scv = self.problem.evaluate(x)
        return x, f, cv, scv
    
    def selection(self, hv, cv, scv, n, generation=None, max_gen=None, **kwargs):
        """
        选择操作 - 子类应该重写此方法
        
        Args:
            hv: 超体积值
            cv: 约束违反矩阵
            scv: 结构约束违反向量
            n: 需要选择的个体数量
            generation: 当前代数（可选）
            max_gen: 最大代数（可选）
            **kwargs: 其他参数
        
        Returns:
            selected: 选中个体的索引数组
        """
        k = 2  # 每次选择的候选个体数量
        prob = 0.5
        cv_sum = np.sum(cv, axis=(1, 2))
        scv_sum = np.sum(scv, axis=1)
        selected = np.zeros(n, dtype=int)
        for i in range(n):
            ran = np.random.rand()
            if ran < prob:
                candidates = np.random.randint(0, len(hv), k)
                best = candidates[0]
                for j in range(1, k):
                    if (scv_sum[candidates[j]] < scv_sum[best]) or \
                       (scv_sum[candidates[j]] == scv_sum[best] and cv_sum[candidates[j]] < cv_sum[best]):
                        best = candidates[j]
                selected[i] = best
            else:
                candidates = np.random.randint(0, len(hv), k)
                best = candidates[0]
                for j in range(1, k):
                    if hv[candidates[j]] > hv[best]:
                        best = candidates[j]
                selected[i] = best

        return selected
    
    def mutation(self, offspring_pop, generation=None, max_gen=None, **kwargs):
        """
        变异操作 - 子类可以重写此方法
        
        Args:
            offspring_pop: 子代种群
            generation: 当前代数（可选）
            max_gen: 最大代数（可选）
            **kwargs: 其他参数
        
        Returns:
            mutated_pop: 变异后的种群
        """
        # 默认变异实现
        # offspring_pop += (np.random.randint(0, 1) - 0.5) * prob * (self.problem.sub_prob.xu - self.problem.sub_prob.xl) * 0.1
        for i in range(self.pop_size):  
            idx = np.random.randint(0, self.problem.n_var)
            diff = (np.random.randint(0, 2) - 0.5) * np.random.rand() * \
                   (self.problem.sub_prob.xu[idx] - self.problem.sub_prob.xl[idx]) * 0.1
            offspring_pop[i, :, idx] += diff
        
        # 处理越界
        offspring_pop = np.clip(offspring_pop, self.problem.sub_prob.xl, self.problem.sub_prob.xu)
        return offspring_pop
    
    def crossover(self, x, f, cv, selected):
        """交叉操作 - 通常不需要重写"""
        offspring_pop = []
        selected_len = int(len(selected) / 2)
        
        for i in range(selected_len):
            tmp_f = np.vstack((f[selected[i]], f[selected[i + selected_len]]))
            tmp_cv = np.vstack((cv[selected[i]], cv[selected[i + selected_len]]))
            tmp_x = np.vstack((x[selected[i]], x[selected[i + selected_len]]))

            fronts, _ = self.sub_alg.fast_non_dominated_sort(tmp_f, tmp_cv)
            selected_front = []
            for front in fronts:
                if len(selected_front) + len(front) > self.sub_pop_size:
                    selected_front.extend(front)
                    break
                selected_front.extend(front)
            selected_front = selected_front[:self.sub_pop_size]
            offspring_pop.append(tmp_x[selected_front])
        
        return np.array(offspring_pop)
    
    def environmental_selection(self, combined_hv, combined_cv, combined_scv):
        """
        环境选择 - 子类可以重写此方法
        
        Args:
            combined_hv: 合并后的超体积值
            combined_cv: 合并后的约束违反矩阵  
            combined_scv: 合并后的结构约束违反向量
        
        Returns:
            selected_indices: 选中个体的索引
        """
        # 默认排序策略
        rank = np.array(range(len(combined_hv)))
        comb = list(zip(rank, combined_hv, combined_cv, combined_scv))
        sorted_comb = sorted(comb, key=lambda x: (np.sum(x[3]), -x[1]))
        return np.array([x[0] for x in sorted_comb])[:self.pop_size]
    
    def log_progress(self, generation, scv):
        """记录进度 - 子类可以重写此方法来自定义输出"""
        if generation % 10 == 0:
            print(f"Generation {generation}:")
            count_negative = np.sum(scv <= 1e-9)
            print(f"结构可行个体比例: {count_negative / (self.pop_size * self.sub_pop_size):.4f}")
            count_all_negative = np.sum(np.all(scv <= 1e-9, axis=1))
            print(f"结构可行子种群比例: {count_all_negative / self.pop_size:.4f}")
    
    def run(self, generations=500):
        """主运行循环 - 通常不需要重写"""
        # 初始化
        x, f, cv, scv = self.initialize_population()
        
        # 计算初始超体积
        f_max = np.array([np.max(f[:, :, i]) for i in range(self.problem.sub_prob.n_obj)])
        f_history_max = f_max
        hv_c = HV(ref_point=f_history_max)
        hv = np.array([hv_c.do(sub_f) for sub_f in f])
        
        # 主循环
        for gen in range(generations):
            # 记录进度
            self.log_progress(gen, scv)
            
            # 选择
            selected = self.selection(hv, cv, scv, self.pop_size * 2, gen, generations)
            
            # 交叉
            offspring_pop = self.crossover(x, f, cv, selected)
            
            # 变异
            offspring_pop = self.mutation(offspring_pop, gen, generations)
            
            # 评估子代
            offspring_f, offspring_cv, offspring_scv = self.problem.evaluate(offspring_pop)
            
            # 合并父代和子代
            combined_pop = np.concatenate((x, offspring_pop), axis=0)
            combined_f = np.concatenate((f, offspring_f), axis=0)
            combined_cv = np.concatenate((cv, offspring_cv), axis=0)
            combined_scv = np.concatenate((scv, offspring_scv), axis=0)
            
            # 更新超体积参考点
            f_max = np.array([np.max(combined_f[:, :, i]) for i in range(self.problem.sub_prob.n_obj)])
            f_history_max = np.maximum(f_max, f_history_max)
            hv_c = HV(ref_point=f_max)
            combined_hv = np.array([hv_c.do(sub_f) for sub_f in combined_f])
            
            # 环境选择
            selected_combined = self.environmental_selection(combined_hv, combined_cv, combined_scv)
            
            # 更新种群
            x = combined_pop[selected_combined]
            f = combined_f[selected_combined]
            cv = combined_cv[selected_combined]
            scv = combined_scv[selected_combined]
            hv = combined_hv[selected_combined]
            
            # 记录历史
            self.history['x'].append(x.copy())
            self.history['f'].append(f.copy())
            self.history['cv'].append(cv.copy())
            self.history['scv'].append(scv.copy())
            self.history['hv'].append(hv.copy())
        
        return f, cv, scv

