import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from alg import *
from problem.prob import *
from problem.sc_prob import *
from utils import *


class StandardNSGA(BaseNSGA):
    """标准NSGA算法实现"""
    
    def selection(self, hv, cv, scv, n, generation=None, max_gen=None, prob=0.5, k=2):
        """标准选择策略"""
        cv_sum = np.sum(cv, axis=(1, 2))
        scv_sum = np.sum(scv, axis=1)
        selected = np.zeros(n, dtype=int)
        
        for i in range(n):
            if np.random.rand() < prob:
                # 约束优先选择
                candidates = np.random.randint(0, len(hv), k)
                best = candidates[0]
                for j in range(1, k):
                    if (scv_sum[candidates[j]] < scv_sum[best]) or \
                       (scv_sum[candidates[j]] == scv_sum[best] and cv_sum[candidates[j]] < cv_sum[best]):
                        best = candidates[j]
                selected[i] = best
            else:
                # 目标优先选择
                candidates = np.random.randint(0, len(hv), k)
                best = candidates[0]
                for j in range(1, k):
                    if hv[candidates[j]] > hv[best]:
                        best = candidates[j]
                selected[i] = best
        
        return selected
    
class ImprovedMutation(BaseNSGA):
    def __init__(self, problem, pop_size=100, sub_pop_size=100, constraint_threshold=1e-6):
        super().__init__(problem, pop_size, sub_pop_size)
        self.constraint_threshold = constraint_threshold
        self.adaptive_mutation_rate = 0.1
        
    def log_progress(self, generation, scv):
        """改进的进度记录"""
        if generation % 10 == 0:
            print(f"Generation {generation}:")
            scv_sum = np.sum(scv, axis=1)
            count_feasible = np.sum(scv_sum <= self.constraint_threshold)
            print(f"结构可行子种群数量: {count_feasible}/{self.pop_size}")
            
            # 输出最优解的约束违反情况
            best_idx = np.argmin(scv_sum)
            print(f"最优解结构约束违反: {scv_sum[best_idx]:.8f}")
            # 这里需要根据具体问题调整x的索引
            # if scv_sum[best_idx] < 0.1:
            #     print(f"最优解x_1均值: {np.mean(x[best_idx, :, 1]):.8f}")
        
    def mutation(self, offspring_pop, generation=None, max_gen=None):
        """精确变异操作"""
        if generation is not None and max_gen is not None:
            mutation_rate = (1 - generation / max_gen) * 0.5
        else:
            mutation_rate = 0.1
        
        for i in range(self.pop_size):
            # 计算当前子种群的结构约束违反
        
            current_scv = self.problem.evaluate_scv(offspring_pop[i:i+1])[0]
            probability = np.random.rand()
            
            if probability > mutation_rate:
        
                change_scale = 0.1
                scv_mean = np.mean(current_scv)
                
                while scv_mean < change_scale:
                    change_scale *= 0.1
                
                idx = np.random.randint(0, self.problem.n_var)
                diff = (np.random.randint(0, 2) - 0.5) * np.random.rand() * change_scale
                offspring_pop[i, :, idx] += diff
            else:
                prob = np.random.rand(self.pop_size, self.sub_pop_size, self.problem.sub_prob.n_var)
                offspring_pop += (np.random.randint(0, 1) - 0.5) * prob * (self.problem.sub_prob.xu - self.problem.sub_prob.xl) * 0.1
        
        # 处理越界
        offspring_pop = np.clip(offspring_pop, self.problem.sub_prob.xl, self.problem.sub_prob.xu)
        return offspring_pop   

class ImprovedNSGA(BaseNSGA):
    """改进的NSGA算法实现"""
    
    def __init__(self, problem, pop_size=100, sub_pop_size=100, constraint_threshold=1e-6):
        super().__init__(problem, pop_size, sub_pop_size)
        self.constraint_threshold = constraint_threshold
        self.adaptive_mutation_rate = 0.1
    
    def selection(self, hv, cv, scv, n, generation=None, max_gen=None, k=3):
        """自适应选择策略"""
        cv_sum = np.sum(cv, axis=(1, 2))
        scv_sum = np.sum(scv, axis=1)
        selected = np.zeros(n, dtype=int)
        
        # 计算结构可行解的比例
        feasible_ratio = np.sum(scv_sum <= self.constraint_threshold) / len(scv_sum)
        
        # 自适应调整约束选择概率
        if generation is not None and max_gen is not None:
            constraint_prob = max(0.8 - 0.5 * (generation / max_gen), 0.3)
        else:
            constraint_prob = 0.5
        
        # 根据可行解比例调整
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
                    if (scv_sum[candidates[j]] < scv_sum[best]) or \
                       (abs(scv_sum[candidates[j]] - scv_sum[best]) < 1e-10 and 
                        cv_sum[candidates[j]] < cv_sum[best]):
                        best = candidates[j]
                selected[i] = best
            else:
                # 目标优先选择（但仍考虑约束）
                best = candidates[0]
                for j in range(1, k):
                    if (scv_sum[candidates[j]] <= self.constraint_threshold and 
                        scv_sum[best] <= self.constraint_threshold):
                        if hv[candidates[j]] > hv[best]:
                            best = candidates[j]
                    elif scv_sum[candidates[j]] < scv_sum[best]:
                        best = candidates[j]
                selected[i] = best
                
        return selected
    
    def mutation(self, offspring_pop, generation=None, max_gen=None):
        """精确变异操作"""
        if generation is not None and max_gen is not None:
            mutation_strength = self.adaptive_mutation_rate * (1 - generation / max_gen) * 0.5
        else:
            mutation_strength = self.adaptive_mutation_rate * 0.5
        
        for i in range(self.pop_size):
            # 计算当前子种群的结构约束违反
            current_scv = self.problem.evaluate_scv(offspring_pop[i:i+1])[0]
            
            # 对于接近可行的解，使用更精确的变异
            if np.mean(current_scv) < 0.1:
                # 针对结构约束的定向变异
                target_val = self.problem.same_val[0]  # 假设为0.2
                
                for j in range(self.sub_pop_size):
                    current_x1 = offspring_pop[i, j, 1]
                    error = current_x1 - target_val
                    
                    if abs(error) > self.constraint_threshold:
                        correction = -error * 0.5 + np.random.normal(0, mutation_strength)
                        offspring_pop[i, j, 1] += correction
                        offspring_pop[i, j, 1] = np.clip(offspring_pop[i, j, 1], 
                                                       self.problem.sub_prob.xl[1], 
                                                       self.problem.sub_prob.xu[1])
            else:
                # 常规变异
                for j in range(self.sub_pop_size):
                    if np.random.rand() < 0.3:
                        idx = np.random.randint(0, self.problem.n_var)
                        diff = np.random.normal(0, mutation_strength) * \
                               (self.problem.sub_prob.xu[idx] - self.problem.sub_prob.xl[idx])
                        offspring_pop[i, j, idx] += diff
        
        # 处理越界
        offspring_pop = np.clip(offspring_pop, self.problem.sub_prob.xl, self.problem.sub_prob.xu)
        return offspring_pop
    
    def environmental_selection(self, combined_hv, combined_cv, combined_scv):
        """增强的排序策略"""
        rank = np.array(range(len(combined_hv)))
        
        def sort_key(x):
            idx, hv_val, cv_val, scv_val = x
            scv_total = np.sum(scv_val)
            cv_total = np.sum(cv_val)
            return (scv_total, cv_total, -hv_val)
        
        comb = list(zip(rank, combined_hv, combined_cv, combined_scv))
        sorted_comb = sorted(comb, key=sort_key)
        return np.array([x[0] for x in sorted_comb])[:self.pop_size]
    
    def log_progress(self, generation, scv):
        """改进的进度记录"""
        if generation % 10 == 0:
            print(f"Generation {generation}:")
            scv_sum = np.sum(scv, axis=1)
            count_feasible = np.sum(scv_sum <= self.constraint_threshold)
            print(f"结构可行子种群数量: {count_feasible}/{self.pop_size}")
            
            # 输出最优解的约束违反情况
            best_idx = np.argmin(scv_sum)
            print(f"最优解结构约束违反: {scv_sum[best_idx]:.8f}")
            # 这里需要根据具体问题调整x的索引
            # if scv_sum[best_idx] < 0.1:
            #     print(f"最优解x_1均值: {np.mean(x[best_idx, :, 1]):.8f}")

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
    
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV

class UnifiedNSGA(NSGA):
    """
    统一的高精度多样性NSGA算法
    整合了精确约束优化和多样性保持的所有策略
    """
    
    def __init__(self, problem, pop_size=100, sub_pop_size=100, 
                 constraint_threshold=1e-6, diversity_weight=0.3, 
                 niching_radius=0.1, archive_size_ratio=2.0):
        super().__init__(problem, pop_size, sub_pop_size)
        
        # 约束优化参数
        self.constraint_threshold = constraint_threshold
        self.adaptive_mutation_rate = 0.1
        
        # 多样性保持参数  
        self.diversity_weight = diversity_weight
        self.niching_radius = niching_radius
        self.min_distance_threshold = 0.05
        
        # 精英存档参数
        self.archive = []
        self.max_archive_size = int(pop_size * archive_size_ratio)
        self.archive_usage_prob = 0.3
        
        # 演化阶段参数
        self.exploration_phase_ratio = 0.3  # 前30%代数为探索阶段
        self.exploitation_phase_ratio = 0.4  # 中40%代数为开发阶段  
        self.refinement_phase_ratio = 0.3   # 后30%代数为精化阶段
        
        # 统计信息
        self.diversity_history = []
        self.constraint_history = []
        
    def get_evolution_phase(self, generation, max_generations):
        """确定当前演化阶段"""
        progress = generation / max_generations
        
        if progress <= self.exploration_phase_ratio:
            return "exploration"  # 探索阶段：注重多样性
        elif progress <= self.exploration_phase_ratio + self.exploitation_phase_ratio:
            return "exploitation"  # 开发阶段：平衡约束和多样性
        else:
            return "refinement"   # 精化阶段：注重精确约束满足
    
    def calculate_diversity_metrics(self, pop):
        """计算种群多样性指标"""
        if len(pop.shape) == 3:
            pop_repr = np.mean(pop, axis=1)  # 子种群代表解
        else:
            pop_repr = pop
            
        if len(pop_repr) < 2:
            return 0.0, 0.0, 0.0
            
        distances = pdist(pop_repr)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        return np.mean(distances), np.min(distances), np.std(distances)
    
    def adaptive_selection_strategy(self, hv, cv, scv, n, generation, max_gen):
        """自适应选择策略 - 根据演化阶段调整"""
        phase = self.get_evolution_phase(generation, max_gen)
        cv_sum = np.sum(cv, axis=(1, 2))
        scv_sum = np.sum(scv, axis=1)
        
        # 计算可行解比例
        feasible_ratio = np.sum(scv_sum <= self.constraint_threshold) / len(scv_sum)
        
        # 根据阶段和可行解比例调整策略
        if phase == "exploration":
            # 探索阶段：高多样性，中等约束压力
            constraint_prob = 0.4 if feasible_ratio < 0.1 else 0.2
            tournament_size = 2
            
        elif phase == "exploitation":  
            # 开发阶段：平衡约束和多样性
            constraint_prob = 0.7 if feasible_ratio < 0.3 else 0.5
            tournament_size = 3
            
        else:  # refinement
            # 精化阶段：高约束压力，保持必要多样性
            constraint_prob = 0.9 if feasible_ratio < 0.8 else 0.6
            tournament_size = 4
        
        return self.tournament_selection_unified(hv, cv_sum, scv_sum, n, 
                                               constraint_prob, tournament_size)
    
    def tournament_selection_unified(self, hv, cv_sum, scv_sum, n, 
                                   constraint_prob, tournament_size):
        """统一的锦标赛选择"""
        selected = np.zeros(n, dtype=int)
        
        for i in range(n):
            candidates = np.random.randint(0, len(hv), tournament_size)
            
            if np.random.rand() < constraint_prob:
                # 约束优先选择
                best = candidates[0]
                for j in range(1, tournament_size):
                    if (scv_sum[candidates[j]] < scv_sum[best]) or \
                       (abs(scv_sum[candidates[j]] - scv_sum[best]) < 1e-10 and 
                        cv_sum[candidates[j]] < cv_sum[best]) or \
                       (abs(scv_sum[candidates[j]] - scv_sum[best]) < 1e-10 and 
                        abs(cv_sum[candidates[j]] - cv_sum[best]) < 1e-10 and
                        hv[candidates[j]] > hv[best]):
                        best = candidates[j]
                selected[i] = best
            else:
                # 多样性和质量平衡选择
                best = candidates[0]
                for j in range(1, tournament_size):
                    # 可行解优先，然后考虑HV
                    if ((scv_sum[candidates[j]] <= self.constraint_threshold and 
                         scv_sum[best] > self.constraint_threshold) or
                        (scv_sum[candidates[j]] <= self.constraint_threshold and 
                         scv_sum[best] <= self.constraint_threshold and
                         hv[candidates[j]] > hv[best]) or
                        (scv_sum[candidates[j]] > self.constraint_threshold and 
                         scv_sum[best] > self.constraint_threshold and
                         scv_sum[candidates[j]] < scv_sum[best])):
                        best = candidates[j]
                selected[i] = best
                
        return selected
    
    def maintain_elite_archive(self, pop, f, cv, scv, hv):
        """维护精英存档"""
        scv_sum = np.sum(scv, axis=1)
        cv_sum = np.sum(cv, axis=(1, 2))
        
        # 添加优秀解到存档
        for i in range(len(pop)):
            repr_sol = np.mean(pop[i], axis=0)
            
            # 计算质量分数
            quality_score = hv[i] - scv_sum[i] - cv_sum[i]
            
            # 计算与存档中解的最小距离
            min_dist = float('inf')
            if len(self.archive) > 0:
                archive_repr = np.array([arch['repr'] for arch in self.archive])
                distances = np.linalg.norm(archive_repr - repr_sol, axis=1)
                min_dist = np.min(distances)
            
            # 添加解的条件：质量好 且 多样性足够
            should_add = False
            if len(self.archive) < self.max_archive_size:
                if min_dist > self.niching_radius * 0.3:  # 多样性要求
                    should_add = True
            else:
                # 存档已满，只添加比最差解更好的
                worst_idx = np.argmin([arch['quality'] for arch in self.archive])
                if (quality_score > self.archive[worst_idx]['quality'] and 
                    min_dist > self.niching_radius * 0.3):
                    self.archive.pop(worst_idx)
                    should_add = True
            
            if should_add:
                self.archive.append({
                    'pop': pop[i].copy(),
                    'f': f[i].copy(), 
                    'cv': cv[i].copy(),
                    'scv': scv[i].copy(),
                    'hv': hv[i],
                    'repr': repr_sol,
                    'quality': quality_score,
                    'diversity': min_dist
                })
    
    def unified_crossover(self, x, f, cv, selected, generation, max_gen):
        """统一的交叉操作"""
        phase = self.get_evolution_phase(generation, max_gen)
        offspring_pop = []
        selected_len = int(len(selected) / 2)
        
        # 根据阶段调整存档使用概率
        if phase == "exploration":
            archive_prob = 0.1  # 少用存档，保持多样性
        elif phase == "exploitation":
            archive_prob = 0.3  # 适度使用存档
        else:  # refinement  
            archive_prob = 0.5  # 多用存档，利用精英
        
        for i in range(selected_len):
            parent1_idx = selected[i]
            parent2_idx = selected[i + selected_len]
            
            # 决定是否使用存档
            if (len(self.archive) > 0 and 
                np.random.rand() < archive_prob):
                
                # 从存档中选择一个多样化的精英解
                archive_qualities = [arch['quality'] for arch in self.archive]
                archive_probs = np.exp(np.array(archive_qualities) - np.max(archive_qualities))
                archive_probs /= np.sum(archive_probs)
                archive_idx = np.random.choice(len(self.archive), p=archive_probs)
                
                if np.random.rand() < 0.5:
                    parent1 = self.archive[archive_idx]['pop']
                    parent2 = x[parent2_idx]
                    p1_f, p1_cv = self.archive[archive_idx]['f'], self.archive[archive_idx]['cv']
                    p2_f, p2_cv = f[parent2_idx], cv[parent2_idx]
                else:
                    parent1 = x[parent1_idx] 
                    parent2 = self.archive[archive_idx]['pop']
                    p1_f, p1_cv = f[parent1_idx], cv[parent1_idx]
                    p2_f, p2_cv = self.archive[archive_idx]['f'], self.archive[archive_idx]['cv']
                    
                tmp_f = np.vstack((p1_f, p2_f))
                tmp_cv = np.vstack((p1_cv, p2_cv))
                tmp_x = np.vstack((parent1, parent2))
            else:
                # 常规交叉
                tmp_f = np.vstack((f[parent1_idx], f[parent2_idx]))
                tmp_cv = np.vstack((cv[parent1_idx], cv[parent2_idx]))
                tmp_x = np.vstack((x[parent1_idx], x[parent2_idx]))
            
            # 子种群内选择
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
    
    def unified_mutation(self, offspring_pop, generation, max_gen):
        """统一的变异操作"""
        phase = self.get_evolution_phase(generation, max_gen)
        
        # 根据阶段调整变异参数
        if phase == "exploration":
            base_mutation_rate = 0.3
            diversity_mutation_prob = 0.8
            constraint_mutation_strength = 0.02
        elif phase == "exploitation":
            base_mutation_rate = 0.2  
            diversity_mutation_prob = 0.5
            constraint_mutation_strength = 0.01
        else:  # refinement
            base_mutation_rate = 0.1
            diversity_mutation_prob = 0.2
            constraint_mutation_strength = 0.005
        
        # 计算当前种群多样性
        avg_dist, min_dist, _ = self.calculate_diversity_metrics(offspring_pop)
        need_diversity_boost = min_dist < self.min_distance_threshold
        
        for i in range(self.pop_size):
            # 计算当前解的约束状态
            current_scv = self.problem.evaluate_scv(offspring_pop[i:i+1])[0]
            scv_mean = np.mean(current_scv)
            
            # 检查解的相似度
            current_repr = np.mean(offspring_pop[i], axis=0)
            is_similar = False
            if i > 0:
                other_repr = [np.mean(offspring_pop[j], axis=0) for j in range(i)]
                min_similarity = min(np.linalg.norm(current_repr - other) for other in other_repr)
                is_similar = min_similarity < self.min_distance_threshold
            
            # 决定变异类型
            if (need_diversity_boost or is_similar) and np.random.rand() < diversity_mutation_prob:
                # 多样性导向变异
                self._diversity_mutation(offspring_pop[i], base_mutation_rate * 2)
            elif scv_mean < 0.1:  
                # 精确约束变异
                self._constraint_mutation(offspring_pop[i], constraint_mutation_strength)
            elif np.random.rand() < base_mutation_rate:
                # 常规变异
                self._regular_mutation(offspring_pop[i], base_mutation_rate)
        
        # 处理越界
        offspring_pop = np.clip(offspring_pop, self.problem.sub_prob.xl, self.problem.sub_prob.xu)
        return offspring_pop
    
    def _diversity_mutation(self, individual, mutation_rate):
        """多样性导向变异"""
        for j in range(len(individual)):
            n_mutate = np.random.randint(1, max(2, self.problem.n_var // 2))
            mutate_vars = np.random.choice(self.problem.n_var, n_mutate, replace=False)
            
            for var_idx in mutate_vars:
                if var_idx == 1:  # 结构约束变量
                    noise = np.random.normal(0, 0.02)
                else:
                    noise = np.random.normal(0, 0.15) * \
                           (self.problem.sub_prob.xu[var_idx] - self.problem.sub_prob.xl[var_idx])
                individual[j, var_idx] += noise
    
    def _constraint_mutation(self, individual, strength):
        """精确约束变异"""
        target_val = self.problem.same_val[0]
        for j in range(len(individual)):
            current_x1 = individual[j, 1]
            error = current_x1 - target_val
            if abs(error) > self.constraint_threshold:
                correction = -error * 0.5 + np.random.normal(0, strength)
                individual[j, 1] += correction
    
    def _regular_mutation(self, individual, mutation_rate):
        """常规变异"""
        for j in range(len(individual)):
            if np.random.rand() < mutation_rate:
                var_idx = np.random.randint(0, self.problem.n_var)
                noise = np.random.normal(0, 0.05) * \
                       (self.problem.sub_prob.xu[var_idx] - self.problem.sub_prob.xl[var_idx])
                individual[j, var_idx] += noise
    
    def niching_selection(self, combined_pop, combined_f, combined_cv, combined_scv, combined_hv):
        """小生境选择"""
        n_total = len(combined_pop)
        repr_solutions = np.mean(combined_pop, axis=1)
        distances = squareform(pdist(repr_solutions))
        
        scv_sum = np.sum(combined_scv, axis=1)
        cv_sum = np.sum(combined_cv, axis=(1, 2))
        
        # 综合适应度计算
        fitness = combined_hv - scv_sum - cv_sum
        
        selected_indices = []
        remaining_indices = list(range(n_total))
        
        while len(selected_indices) < self.pop_size and remaining_indices:
            # 选择适应度最高的
            remaining_fitness = [fitness[i] for i in remaining_indices]
            best_idx_in_remaining = np.argmax(remaining_fitness)
            best_idx = remaining_indices[best_idx_in_remaining]
            
            selected_indices.append(best_idx)
            
            # 移除小生境内的解
            to_remove = []
            for i, idx in enumerate(remaining_indices):
                if distances[best_idx, idx] < self.niching_radius:
                    to_remove.append(i)
            
            for i in sorted(to_remove, reverse=True):
                remaining_indices.pop(i)
        
        # 补充剩余解
        while len(selected_indices) < self.pop_size and remaining_indices:
            remaining_fitness = [fitness[i] for i in remaining_indices]
            best_idx_in_remaining = np.argmax(remaining_fitness)
            best_idx = remaining_indices.pop(best_idx_in_remaining)
            selected_indices.append(best_idx)
        
        return np.array(selected_indices[:self.pop_size])
    
    def run(self, generations=500):
        """统一算法主流程"""
        print("=== 统一NSGA算法开始运行 ===")
        print(f"种群大小: {self.pop_size}, 子种群大小: {self.sub_pop_size}")
        print(f"总代数: {generations}")
        print(f"约束阈值: {self.constraint_threshold}")
        print(f"小生境半径: {self.niching_radius}")
        
        # 初始化
        x, f, cv, scv = self.initialize_population()
        
        f_max = [np.max(f[:, :, i]) for i in range(self.problem.sub_prob.n_obj)]
        f_max = np.array(f_max)
        f_history_max = f_max
        hv_c = HV(ref_point=f_history_max)
        hv = [hv_c.do(sub_f) for sub_f in f]
        hv = np.array(hv)
        
        # 主进化循环
        for gen in range(generations):
            current_phase = self.get_evolution_phase(gen, generations)
            
            if gen % 25 == 0:
                self._print_progress(gen, current_phase, scv, x, hv)
            
            # 维护精英存档
            self.maintain_elite_archive(x, f, cv, scv, hv)
            
            # 自适应选择
            selected = self.adaptive_selection_strategy(hv, cv, scv, 
                                                      self.pop_size*2, gen, generations)
            
            # 统一交叉
            offspring_pop = self.unified_crossover(x, f, cv, selected, gen, generations)
            
            # 统一变异
            offspring_pop = self.unified_mutation(offspring_pop, gen, generations)
            
            # 评估子代
            offspring_f, offspring_cv, offspring_scv = self.problem.evaluate(offspring_pop)
            
            # 合并种群
            combined_pop = np.concatenate((x, offspring_pop), axis=0)
            combined_f = np.concatenate((f, offspring_f), axis=0)
            combined_cv = np.concatenate((cv, offspring_cv), axis=0)
            combined_scv = np.concatenate((scv, offspring_scv), axis=0)
            
            # 更新参考点
            f_max = [np.max(combined_f[:, :, i]) for i in range(self.problem.sub_prob.n_obj)]
            f_max = np.array(f_max)
            f_history_max = np.maximum(f_max, f_history_max)
            hv_c = HV(ref_point=f_max)
            combined_hv = [hv_c.do(sub_f) for sub_f in combined_f]
            combined_hv = np.array(combined_hv)
            
            # 小生境选择
            selected_indices = self.niching_selection(combined_pop, combined_f, 
                                                    combined_cv, combined_scv, combined_hv)
            
            # 更新种群
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
            
            # 记录统计信息
            if gen % 10 == 0:
                avg_dist, min_dist, std_dist = self.calculate_diversity_metrics(x)
                self.diversity_history.append((gen, avg_dist, min_dist, std_dist))
                
                scv_sum = np.sum(scv, axis=1)
                feasible_count = np.sum(scv_sum <= self.constraint_threshold)
                self.constraint_history.append((gen, feasible_count, np.min(scv_sum), np.mean(scv_sum)))
        
        self._print_final_results(x, scv)
        return f, cv, scv
    
    def _print_progress(self, gen, phase, scv, x, hv):
        """打印进度信息"""
        print(f"\n--- 第 {gen} 代 ({phase.upper()} 阶段) ---")
        
        scv_sum = np.sum(scv, axis=1)
        feasible_count = np.sum(scv_sum <= self.constraint_threshold)
        print(f"可行解数量: {feasible_count}/{self.pop_size}")
        
        if feasible_count > 0:
            feasible_mask = scv_sum <= self.constraint_threshold
            best_feasible_idx = np.where(feasible_mask)[0][np.argmax(hv[feasible_mask])]
            best_x1_values = x[best_feasible_idx, :, 1]
            print(f"最佳可行解x_1: 均值={np.mean(best_x1_values):.8f}, "
                  f"标准差={np.std(best_x1_values):.8f}")
        
        best_idx = np.argmin(scv_sum)
        print(f"最优解约束违反: {scv_sum[best_idx]:.8f}")
        
        avg_dist, min_dist, _ = self.calculate_diversity_metrics(x)
        print(f"种群多样性: 平均距离={avg_dist:.6f}, 最小距离={min_dist:.6f}")
        print(f"存档大小: {len(self.archive)}")
    
    def _print_final_results(self, x, scv):
        """打印最终结果"""
        print(f"\n{'='*50}")
        print("最终结果分析")
        print(f"{'='*50}")
        
        scv_sum = np.sum(scv, axis=1)
        feasible_mask = scv_sum <= self.constraint_threshold
        feasible_count = np.sum(feasible_mask)
        
        print(f"完全可行解数量: {feasible_count}/{self.pop_size}")
        print(f"可行解比例: {feasible_count/self.pop_size*100:.1f}%")
        
        if feasible_count > 0:
            feasible_x1 = x[feasible_mask, :, 1]
            print(f"\n可行解x_1统计:")
            print(f"  目标值: {self.problem.same_val[0]}")
            print(f"  实际均值: {np.mean(feasible_x1):.10f}")
            print(f"  误差: {abs(np.mean(feasible_x1) - self.problem.same_val[0]):.10f}")
            print(f"  标准差: {np.std(feasible_x1):.10f}")
            print(f"  范围: [{np.min(feasible_x1):.8f}, {np.max(feasible_x1):.8f}]")
        
        # 最优解分析
        best_idx = np.argmin(scv_sum)
        print(f"\n最优解分析:")
        print(f"  约束违反量: {scv_sum[best_idx]:.10f}")
        print(f"  x_1均值: {np.mean(x[best_idx, :, 1]):.10f}")
        
        # 多样性分析
        final_diversity = self.calculate_diversity_metrics(x)
        print(f"\n最终多样性:")
        print(f"  平均距离: {final_diversity[0]:.6f}")
        print(f"  最小距离: {final_diversity[1]:.6f}")
        print(f"  距离标准差: {final_diversity[2]:.6f}")
        
        print(f"\n存档统计:")
        print(f"  存档大小: {len(self.archive)}")
        if len(self.archive) > 0:
            archive_qualities = [arch['quality'] for arch in self.archive]
            print(f"  存档质量范围: [{np.min(archive_qualities):.4f}, {np.max(archive_qualities):.4f}]")

# ============================================================================
# 算法流程说明
# ============================================================================

def print_algorithm_workflow():
    """打印算法的详细流程"""
    workflow = """
    统一NSGA算法流程 (UnifiedNSGA)
    ==========================================
    
    1. 初始化阶段:
       - 生成初始种群 (pop_size × sub_pop_size × n_var)
       - 评估初始种群的目标函数、约束和结构约束
       - 计算初始超体积值
       - 初始化精英存档
    
    2. 主进化循环 (每代):
       
       A. 演化阶段判断:
          - 探索阶段 (前30%): 重视多样性，适度约束压力
          - 开发阶段 (中40%): 平衡约束满足和多样性  
          - 精化阶段 (后30%): 重视精确约束满足
       
       B. 选择操作:
          - 根据当前阶段自适应调整选择策略
          - 锦标赛大小和约束优先概率动态变化
          - 平衡约束违反、超体积和多样性
       
       C. 交叉操作:
          - 常规交叉: 父代种群内选择
          - 存档交叉: 利用历史精英解 (概率随阶段变化)
          - 子种群内非支配排序选择最优后代
       
       D. 变异操作:
          - 多样性变异: 解相似时大幅变异多个变量
          - 精确约束变异: 接近可行时向目标值微调
          - 常规变异: 随机扰动保持搜索能力
       
       E. 环境选择:
          - 小生境选择: 确保解在空间中均匀分布
          - 距离阈值控制: 避免过度相似的解
          - 综合适应度: 结合HV、约束违反和多样性
       
       F. 存档维护:
          - 添加高质量且多样化的解
          - 存档大小控制和质量更新
          - 为后续交叉提供精英基因
    
    3. 自适应机制:
       - 参数随演化阶段和种群状态动态调整
       - 可行解比例影响选择策略
       - 多样性指标影响变异强度
    
    4. 多样性保持:
       - 小生境选择确保空间分布
       - 距离监控防止早熟收敛  
       - 存档机制保持历史多样性
    
    5. 精确约束优化:
       - 分阶段增强约束选择压力
       - 定向变异推动解向约束边界
       - 约束阈值控制精度要求
    
    关键创新点:
    - 三阶段自适应演化策略
    - 统一的选择-交叉-变异框架
    - 精英存档与小生境选择结合
    - 多层次多样性保持机制
    - 精确约束满足技术
    """
    print(workflow)

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
    generations = 200
    pop_size = 50
    sub_pop_size = 10
    n_var = 6
    x_idx = [2]
    y_idx = [0.2]
    PROBLEM_CLASS = ZDT1NoCV
    WRAPPER_CLASS = SharedX_1
    suffix = 'MyImprovedMutation'

    prob = WRAPPER_CLASS(PROBLEM_CLASS, n_var=n_var, share_idx=x_idx, same_val=y_idx)
    nsga = ImprovedMutation(prob, pop_size=pop_size, sub_pop_size=sub_pop_size)
    # np.set_printoptions(threshold=np.inf)
    f, cv, scv = nsga.run(generations=generations)
    
    filename = generate_filename(PROBLEM_CLASS, WRAPPER_CLASS, generations, pop_size, sub_pop_size, n_var, x_idx, y_idx, suffix=suffix)
    
    save_history_array(nsga.history, "x", filename)
    
    print("目标函数值:", f)
    print("约束违反量:", cv < 1e-9)
    print("结构约束违反量:", scv < 1e-9)
    
    pf = prob.sub_prob.pareto_front(100)
    plot_objectives(f, pf, WRAPPER_CLASS.__name__ + "_" + PROBLEM_CLASS.__name__, filename + ".png")