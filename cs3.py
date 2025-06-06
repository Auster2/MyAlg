import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from alg import NSGA
from problem.prob import *
from problem.sc_prob import *
from utils import *

class ImprovedNSGA(NSGA):
    def __init__(self, problem, pop_size=100, sub_pop_size=100, constraint_threshold=1e-6):
        super().__init__(problem, pop_size, sub_pop_size)
        self.constraint_threshold = constraint_threshold  # 约束满足阈值
        self.adaptive_mutation_rate = 0.1  # 自适应变异率
        
    def adaptive_randSelect(self, hv, cv, scv, n, generation, max_gen, k=3):
        """自适应选择策略"""
        cv_sum = np.sum(cv, axis=(1, 2))
        scv_sum = np.sum(scv, axis=1)
        selected = np.zeros(n, dtype=int)
        
        # 计算结构可行解的比例
        feasible_ratio = np.sum(scv_sum <= self.constraint_threshold) / len(scv_sum)
        
        # 自适应调整约束选择概率
        # 早期更关注约束，后期平衡约束和目标
        constraint_prob = max(0.8 - 0.5 * (generation / max_gen), 0.3)
        
        # 如果可行解很少，增加约束选择概率
        if feasible_ratio < 0.1:
            constraint_prob = 0.9
        elif feasible_ratio > 0.8:
            constraint_prob = 0.3
            
        for i in range(n):
            candidates = np.random.randint(0, len(hv), k)
            
            if np.random.rand() < constraint_prob:
                # 约束优先选择
                best = candidates[0]
                for j in range(1, k):
                    # 优先选择结构约束违反更小的
                    if (scv_sum[candidates[j]] < scv_sum[best]) or \
                       (abs(scv_sum[candidates[j]] - scv_sum[best]) < 1e-10 and 
                        cv_sum[candidates[j]] < cv_sum[best]):
                        best = candidates[j]
                selected[i] = best
            else:
                # 目标优先选择（但仍考虑约束）
                best = candidates[0]
                for j in range(1, k):
                    # 如果都满足约束，选择HV更大的
                    if (scv_sum[candidates[j]] <= self.constraint_threshold and 
                        scv_sum[best] <= self.constraint_threshold):
                        if hv[candidates[j]] > hv[best]:
                            best = candidates[j]
                    # 如果候选解约束更好，选择它
                    elif scv_sum[candidates[j]] < scv_sum[best]:
                        best = candidates[j]
                selected[i] = best
                
        return selected
    
    def precise_mutation(self, offspring_pop, generation, max_gen):
        """精确变异操作"""
        # 自适应变异强度
        mutation_strength = self.adaptive_mutation_rate * (1 - generation / max_gen) * 0.5
        
        for i in range(self.pop_size):
            # 计算当前子种群的结构约束违反
            current_scv = self.problem.evaluate_scv(offspring_pop[i:i+1])[0]
            
            # 对于接近可行的解，使用更精确的变异
            if np.mean(current_scv) < 0.1:  # 如果已经接近可行
                # 针对结构约束的定向变异
                target_val = self.problem.same_val[0]  # 0.2
                
                for j in range(self.sub_pop_size):
                    current_x1 = offspring_pop[i, j, 1]
                    error = current_x1 - target_val
                    
                    # 如果误差较大，进行修正
                    if abs(error) > self.constraint_threshold:
                        # 向目标值方向移动，但保持随机性
                        correction = -error * 0.5 + np.random.normal(0, mutation_strength)
                        offspring_pop[i, j, 1] += correction
                        
                        # 确保不越界
                        offspring_pop[i, j, 1] = np.clip(offspring_pop[i, j, 1], 
                                                       self.problem.sub_prob.xl[1], 
                                                       self.problem.sub_prob.xu[1])
            else:
                # 常规变异
                for j in range(self.sub_pop_size):
                    if np.random.rand() < 0.3:  # 30%概率进行变异
                        idx = np.random.randint(0, self.problem.n_var)
                        diff = np.random.normal(0, mutation_strength) * \
                               (self.problem.sub_prob.xu[idx] - self.problem.sub_prob.xl[idx])
                        offspring_pop[i, j, idx] += diff
        
        # 处理越界
        offspring_pop = np.clip(offspring_pop, self.problem.sub_prob.xl, self.problem.sub_prob.xu)
        return offspring_pop
    
    def enhanced_sort(self, hv, cv, scv):
        """增强的排序策略"""
        rank = np.array(range(len(hv)))
        scv_sum = np.sum(scv, axis=1)
        cv_sum = np.sum(cv, axis=(1, 2))
        
        # 多层次排序
        def sort_key(x):
            idx, hv_val, cv_val, scv_val = x
            scv_total = np.sum(scv_val)
            cv_total = np.sum(cv_val)
            
            # 第一优先级：结构约束违反量
            # 第二优先级：原约束违反量  
            # 第三优先级：超体积（负号表示降序）
            return (scv_total, cv_total, -hv_val)
        
        comb = list(zip(rank, hv, cv, scv))
        sorted_comb = sorted(comb, key=sort_key)
        sorted_rank = np.array([x[0] for x in sorted_comb])
        return sorted_rank
    
    def run(self, generations=500):
        """改进的运行方法"""
        x, f, cv, scv = self.initialize_population()
        
        f_max = []
        for i in range(self.problem.sub_prob.n_obj):
            f_max.append(np.max(f[:, :, i]))
        f_max = np.array(f_max)
        f_history_max = f_max
        hv_c = HV(ref_point=f_history_max)
        hv = [hv_c.do(sub_f) for sub_f in f]
        
        for gen in range(generations):
            if gen % 10 == 0:
                print(f"Generation {gen}:")
                scv_sum = np.sum(scv, axis=1)
                count_feasible = np.sum(scv_sum <= self.constraint_threshold)
                print(f"结构可行子种群数量: {count_feasible}/{self.pop_size}")
                
                # 输出最优解的约束违反情况
                best_idx = np.argmin(scv_sum)
                print(f"最优解结构约束违反: {scv_sum[best_idx]:.8f}")
                if scv_sum[best_idx] < 0.1:
                    print(f"最优解x_1均值: {np.mean(x[best_idx, :, 1]):.8f}")
                
            # 使用改进的选择策略
            selected = self.adaptive_randSelect(hv, cv, scv, self.pop_size*2, gen, generations)
            offspring_pop = self.crossover(x, f, cv, selected)
            
            # 使用精确变异
            offspring_pop = self.precise_mutation(offspring_pop, gen, generations)
            
            offspring_f, offspring_cv, offspring_scv = self.problem.evaluate(offspring_pop)
            
            # 合并父代和子代
            combined_pop = np.concatenate((x, offspring_pop), axis=0)
            combined_f = np.concatenate((f, offspring_f), axis=0)
            combined_cv = np.concatenate((cv, offspring_cv), axis=0)
            combined_scv = np.concatenate((scv, offspring_scv), axis=0)
            
            f_max = []
            for i in range(self.problem.sub_prob.n_obj):
                f_max.append(np.max(combined_f[:, :, i]))
            f_max = np.array(f_max)
            f_history_max = np.maximum(f_max, f_history_max)
            hv_c = HV(ref_point=f_max)
            combined_hv = [hv_c.do(sub_f) for sub_f in combined_f]
            combined_hv = np.array(combined_hv)

            # 使用增强的排序策略
            selected_combined = self.enhanced_sort(combined_hv, combined_cv, combined_scv)[:self.pop_size]

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

class DiversityPreservedNSGA(NSGA):
    def __init__(self, problem, pop_size=100, sub_pop_size=100, 
                 constraint_threshold=1e-6, diversity_weight=0.3):
        super().__init__(problem, pop_size, sub_pop_size)
        self.constraint_threshold = constraint_threshold
        self.diversity_weight = diversity_weight  # 多样性权重
        self.niching_radius = 0.1  # 小生境半径
        self.archive = []  # 精英存档
        self.max_archive_size = pop_size * 2
        
    def calculate_diversity_metrics(self, pop):
        """计算种群多样性指标"""
        if len(pop.shape) == 3:  # (pop_size, sub_pop_size, n_var)
            # 对每个子种群取均值作为代表解
            pop_repr = np.mean(pop, axis=1)
        else:
            pop_repr = pop
            
        if len(pop_repr) < 2:
            return np.array([0.0])
            
        # 计算解之间的欧几里得距离
        distances = pdist(pop_repr)
        
        if len(distances) == 0:
            return np.array([0.0])
            
        # 多样性指标：平均距离、最小距离、距离标准差
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        std_distance = np.std(distances)
        
        return avg_distance, min_distance, std_distance
    
    def niching_selection(self, combined_pop, combined_f, combined_cv, combined_scv, combined_hv):
        """基于小生境的选择策略"""
        n_total = len(combined_pop)
        selected_indices = []
        
        # 计算代表解（每个子种群的均值）
        repr_solutions = np.mean(combined_pop, axis=1)  # (n_total, n_var)
        
        # 计算解之间的距离矩阵
        distances = squareform(pdist(repr_solutions))
        
        # 计算适应度（结合约束、HV和多样性）
        scv_sum = np.sum(combined_scv, axis=1)
        cv_sum = np.sum(combined_cv, axis=(1, 2))
        
        # 标准化各指标到[0,1]
        if np.max(scv_sum) > np.min(scv_sum):
            scv_norm = (scv_sum - np.min(scv_sum)) / (np.max(scv_sum) - np.min(scv_sum))
        else:
            scv_norm = np.zeros_like(scv_sum)
            
        if np.max(combined_hv) > np.min(combined_hv):
            hv_norm = (combined_hv - np.min(combined_hv)) / (np.max(combined_hv) - np.min(combined_hv))
        else:
            hv_norm = np.ones_like(combined_hv)
        
        # 综合适应度 = 约束项 + HV项
        fitness = -scv_norm + hv_norm  # 越大越好
        
        # 小生境选择
        remaining_indices = list(range(n_total))
        
        while len(selected_indices) < self.pop_size and remaining_indices:
            # 在剩余解中选择适应度最高的
            remaining_fitness = [fitness[i] for i in remaining_indices]
            best_idx_in_remaining = np.argmax(remaining_fitness)
            best_idx = remaining_indices[best_idx_in_remaining]
            
            selected_indices.append(best_idx)
            
            # 移除在小生境半径内的解
            to_remove = []
            for i, idx in enumerate(remaining_indices):
                if distances[best_idx, idx] < self.niching_radius:
                    to_remove.append(i)
            
            # 从后往前删除，避免索引错乱
            for i in sorted(to_remove, reverse=True):
                remaining_indices.pop(i)
        
        # 如果选择的解不够，补充剩余的最优解
        while len(selected_indices) < self.pop_size and remaining_indices:
            remaining_fitness = [fitness[i] for i in remaining_indices]
            best_idx_in_remaining = np.argmax(remaining_fitness)
            best_idx = remaining_indices.pop(best_idx_in_remaining)
            selected_indices.append(best_idx)
        
        return np.array(selected_indices[:self.pop_size])
    
    def diversity_mutation(self, offspring_pop, generation, max_gen):
        """保持多样性的变异操作"""
        mutation_rate = 0.1 * (1 + generation / max_gen)  # 后期增加变异率
        
        for i in range(self.pop_size):
            # 计算当前解与种群中其他解的相似度
            current_repr = np.mean(offspring_pop[i], axis=0)
            
            # 检查是否需要增强多样性
            need_diversity = False
            if i > 0:
                other_repr = np.mean(offspring_pop[:i], axis=(0, 1))
                similarity = np.linalg.norm(current_repr - other_repr)
                if similarity < 0.05:  # 相似度过高
                    need_diversity = True
            
            if need_diversity or np.random.rand() < mutation_rate:
                # 多样性导向的变异
                for j in range(self.sub_pop_size):
                    # 随机选择变异的变量数量
                    n_mutate = np.random.randint(1, max(2, self.problem.n_var // 3))
                    mutate_vars = np.random.choice(self.problem.n_var, n_mutate, replace=False)
                    
                    for var_idx in mutate_vars:
                        if var_idx == 1:  # 结构约束变量，小幅变异
                            noise = np.random.normal(0, 0.01)
                        else:  # 其他变量，可以大幅变异
                            noise = np.random.normal(0, 0.1) * \
                                   (self.problem.sub_prob.xu[var_idx] - self.problem.sub_prob.xl[var_idx])
                        
                        offspring_pop[i, j, var_idx] += noise
            else:
                # 常规变异，主要针对约束
                current_scv = self.problem.evaluate_scv(offspring_pop[i:i+1])[0]
                if np.mean(current_scv) < 0.1:  # 接近可行时，精确变异
                    target_val = self.problem.same_val[0]
                    for j in range(self.sub_pop_size):
                        current_x1 = offspring_pop[i, j, 1]
                        error = current_x1 - target_val
                        if abs(error) > self.constraint_threshold:
                            correction = -error * 0.3 + np.random.normal(0, 0.005)
                            offspring_pop[i, j, 1] += correction
        
        # 处理越界
        offspring_pop = np.clip(offspring_pop, self.problem.sub_prob.xl, self.problem.sub_prob.xu)
        return offspring_pop
    
    def maintain_archive(self, pop, f, cv, scv):
        """维护精英存档"""
        # 计算综合质量评分
        scv_sum = np.sum(scv, axis=1)
        cv_sum = np.sum(cv, axis=(1, 2))
        
        # 添加当前种群中的优秀解到存档
        for i, (p, fi, cvi, scvi) in enumerate(zip(pop, f, cv, scv)):
            quality_score = -scv_sum[i] - cv_sum[i]  # 越大越好
            
            # 计算多样性（与存档中解的最小距离）
            repr_sol = np.mean(p, axis=0)
            min_dist = float('inf')
            
            if len(self.archive) > 0:
                archive_repr = np.array([np.mean(arch['pop'], axis=0) for arch in self.archive])
                distances = np.linalg.norm(archive_repr - repr_sol, axis=1)
                min_dist = np.min(distances)
            
            # 如果解足够好且足够多样化，加入存档
            if min_dist > self.niching_radius * 0.5:  # 多样性要求
                self.archive.append({
                    'pop': p.copy(),
                    'f': fi.copy(),
                    'cv': cvi.copy(),
                    'scv': scvi.copy(),
                    'quality': quality_score,
                    'diversity': min_dist
                })
        
        # 限制存档大小
        if len(self.archive) > self.max_archive_size:
            # 按质量和多样性排序
            self.archive.sort(key=lambda x: x['quality'] + 0.1 * x['diversity'], reverse=True)
            self.archive = self.archive[:self.max_archive_size]
    
    def adaptive_crossover_with_archive(self, x, f, cv, selected):
        """结合存档的自适应交叉"""
        offspring_pop = []
        selected_len = int(len(selected) / 2)
        
        for i in range(selected_len):
            parent1_idx = selected[i]
            parent2_idx = selected[i + selected_len]
            
            # 30%概率使用存档中的精英解作为父本
            if len(self.archive) > 0 and np.random.rand() < 0.3:
                archive_idx = np.random.randint(0, len(self.archive))
                if np.random.rand() < 0.5:
                    parent1 = self.archive[archive_idx]['pop']
                    parent2 = x[parent2_idx]
                else:
                    parent1 = x[parent1_idx]
                    parent2 = self.archive[archive_idx]['pop']
                    
                tmp_f = np.vstack((f[parent1_idx] if 'parent1' not in locals() else self.archive[archive_idx]['f'], 
                                 f[parent2_idx]))
                tmp_cv = np.vstack((cv[parent1_idx] if 'parent1' not in locals() else self.archive[archive_idx]['cv'], 
                                  cv[parent2_idx]))
                tmp_x = np.vstack((parent1, parent2))
            else:
                # 常规交叉
                tmp_f = np.vstack((f[parent1_idx], f[parent2_idx]))
                tmp_cv = np.vstack((cv[parent1_idx], cv[parent2_idx]))
                tmp_x = np.vstack((x[parent1_idx], x[parent2_idx]))
            
            # 选择优秀解
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
    
    def run(self, generations=500):
        """改进的运行方法"""
        x, f, cv, scv = self.initialize_population()
        
        f_max = []
        for i in range(self.problem.sub_prob.n_obj):
            f_max.append(np.max(f[:, :, i]))
        f_max = np.array(f_max)
        f_history_max = f_max
        hv_c = HV(ref_point=f_history_max)
        hv = [hv_c.do(sub_f) for sub_f in f]
        
        diversity_history = []
        
        for gen in range(generations):
            if gen % 20 == 0:
                print(f"Generation {gen}:")
                scv_sum = np.sum(scv, axis=1)
                count_feasible = np.sum(scv_sum <= self.constraint_threshold)
                print(f"结构可行子种群数量: {count_feasible}/{self.pop_size}")
                
                # 多样性分析
                avg_dist, min_dist, std_dist = self.calculate_diversity_metrics(x)
                diversity_history.append((gen, avg_dist, min_dist, std_dist))
                print(f"多样性指标 - 平均距离: {avg_dist:.6f}, 最小距离: {min_dist:.6f}")
                
                # 最优解信息
                best_idx = np.argmin(scv_sum)
                print(f"最优解结构约束违反: {scv_sum[best_idx]:.8f}")
                print(f"存档大小: {len(self.archive)}")
            
            # 维护存档
            self.maintain_archive(x, f, cv, scv)
            
            # 选择
            selected = self.randSelect(hv, cv, scv, self.pop_size*2)
            
            # 带存档的交叉
            offspring_pop = self.adaptive_crossover_with_archive(x, f, cv, selected)
            
            # 多样性导向的变异
            offspring_pop = self.diversity_mutation(offspring_pop, gen, generations)
            
            offspring_f, offspring_cv, offspring_scv = self.problem.evaluate(offspring_pop)
            
            # 合并父代和子代
            combined_pop = np.concatenate((x, offspring_pop), axis=0)
            combined_f = np.concatenate((f, offspring_f), axis=0)
            combined_cv = np.concatenate((cv, offspring_cv), axis=0)
            combined_scv = np.concatenate((scv, offspring_scv), axis=0)
            
            f_max = []
            for i in range(self.problem.sub_prob.n_obj):
                f_max.append(np.max(combined_f[:, :, i]))
            f_max = np.array(f_max)
            f_history_max = np.maximum(f_max, f_history_max)
            hv_c = HV(ref_point=f_max)
            combined_hv = [hv_c.do(sub_f) for sub_f in combined_f]
            combined_hv = np.array(combined_hv)

            # 使用小生境选择
            selected_indices = self.niching_selection(combined_pop, combined_f, combined_cv, 
                                                    combined_scv, combined_hv)

            x = combined_pop[selected_indices]
            f = combined_f[selected_indices]
            cv = combined_cv[selected_indices]
            scv = combined_scv[selected_indices]
            hv = combined_hv[selected_indices]
            
            # 记录历史
            self.history['x'].append(x.copy())
            self.history['f'].append(f.copy())
            self.history['cv'].append(cv.copy())
            self.history['scv'].append(scv.copy())
            self.history['hv'].append(hv.copy())
        
        # 输出多样性统计
        print("\n多样性演化历史:")
        for gen, avg_dist, min_dist, std_dist in diversity_history[-5:]:  # 最后5次记录
            print(f"Gen {gen}: 平均距离={avg_dist:.6f}, 最小距离={min_dist:.6f}, 标准差={std_dist:.6f}")
        
        return f, cv, scv

# 使用示例
# if __name__ == "__main__":
#     generations = 400
#     pop_size = 60  # 适当增加种群大小
#     sub_pop_size = 10
#     n_var = 30
#     y_idx = [0.2]
    
#     prob = SharedX_1(ZDT1NoCV, n_var=n_var, same_val=y_idx)
    
#     # 使用保持多样性的算法
#     nsga = DiversityPreservedNSGA(prob, pop_size=pop_size, sub_pop_size=sub_pop_size, 
#                                  constraint_threshold=1e-6, diversity_weight=0.3)
    
#     f, cv, scv = nsga.run(generations=generations)
    
#     # 最终分析
#     print(f"\n=== 最终结果分析 ===")
#     scv_sum = np.sum(scv, axis=1)
    
#     # 可行解分析
#     feasible_mask = scv_sum <= 1e-6
#     print(f"完全可行解数量: {np.sum(feasible_mask)}/{pop_size}")
    
#     if np.sum(feasible_mask) > 0:
#         feasible_x1_values = nsga.history['x'][-1][feasible_mask, :, 1]
#         print(f"可行解x_1统计:")
#         print(f"  均值: {np.mean(feasible_x1_values):.8f}")
#         print(f"  标准差: {np.std(feasible_x1_values):.8f}")
#         print(f"  范围: [{np.min(feasible_x1_values):.8f}, {np.max(feasible_x1_values):.8f}]")
    
#     # 总体多样性
#     final_diversity = nsga.calculate_diversity_metrics(nsga.history['x'][-1])
#     print(f"最终种群多样性: 平均距离={final_diversity[0]:.6f}")

# 使用示例
# if __name__ == "__main__":
#     generations = 300  # 增加代数
#     pop_size = 50
#     sub_pop_size = 10
#     n_var = 30
#     x_idx = []
#     y_idx = [0.2]
#     PROBLEM_CLASS = ZDT1NoCV
#     WRAPPER_CLASS = SharedX_1
#     suffix = 'ImprovedConstraint'

#     prob = WRAPPER_CLASS(PROBLEM_CLASS, n_var=n_var, same_val=y_idx)
    
#     # 使用改进的算法
#     nsga = ImprovedNSGA(prob, pop_size=pop_size, sub_pop_size=sub_pop_size, 
#                        constraint_threshold=1e-6)
    
#     f, cv, scv = nsga.run(generations=generations)
    
#     # 分析最终结果
#     scv_sum = np.sum(scv, axis=1)
#     best_idx = np.argmin(scv_sum)
#     print(f"\n最终结果分析:")
#     print(f"最优解结构约束违反: {scv_sum[best_idx]:.10f}")
#     print(f"最优解x_1值: {nsga.history['x'][-1][best_idx, :, 1]}")
#     print(f"x_1均值: {np.mean(nsga.history['x'][-1][best_idx, :, 1]):.10f}")
#     print(f"x_1标准差: {np.std(nsga.history['x'][-1][best_idx, :, 1]):.10f}")
        

if __name__ == "__main__":
    generations = 1000
    pop_size = 50
    sub_pop_size = 10
    n_var = 30
    x_idx = []
    y_idx = [0.2]
    PROBLEM_CLASS = ZDT1NoCV
    WRAPPER_CLASS = SharedX_1
    suffix = 'DiversityPreservedNSGA'

    prob = WRAPPER_CLASS(PROBLEM_CLASS, n_var=n_var, same_val=y_idx)
    nsga = DiversityPreservedNSGA(prob, pop_size=pop_size, sub_pop_size=sub_pop_size)
    # np.set_printoptions(threshold=np.inf)
    f, cv, scv = nsga.run(generations=generations)
    
    filename = generate_filename(PROBLEM_CLASS, WRAPPER_CLASS, generations, pop_size, sub_pop_size, n_var, x_idx, y_idx, suffix=suffix)
    
    save_history_array(nsga.history, "x", filename)
    
    print("目标函数值:", f)
    print("约束违反量:", cv < 1e-9)
    print("结构约束违反量:", scv < 1e-9)
    
    pf = prob.sub_prob.pareto_front(100)
    plot_objectives(f, pf, WRAPPER_CLASS.__name__ + "_" + PROBLEM_CLASS.__name__, filename + ".png")