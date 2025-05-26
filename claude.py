import numpy as np
import matplotlib.pyplot as plt

# 问题定义: ZDT1
class ZDT1:
    def __init__(self, n_var=30):
        self.n_var = n_var
        self.n_obj = 2
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
    
    def evaluate(self, x):
        """计算目标函数值"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        f = np.zeros((x.shape[0], 2))
        
        # 第一个目标
        f[:, 0] = x[:, 0]
        
        # 第二个目标
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
        f[:, 1] = g * (1 - np.sqrt(x[:, 0] / g))
        
        return f


# NSGA-II的实现
class NSGA2:
    def __init__(self, problem, pop_size=100, crossover_prob=0.9, crossover_eta=20, 
                 mutation_prob=None, mutation_eta=20):
        self.problem = problem
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_prob = mutation_prob if mutation_prob is not None else 1.0 / problem.n_var
        self.mutation_eta = mutation_eta
    
    def initialize_population(self):
        """初始化种群"""
        x = np.random.random((self.pop_size, self.problem.n_var))
        # x shape(100, 30)
        x = self.problem.xl + x * (self.problem.xu - self.problem.xl)
        f = self.problem.evaluate(x)
        return x, f
    
    def fast_non_dominated_sort(self, f):
        """快速非支配排序 fronts, rank存的都是索引"""
        n_points = f.shape[0]
        S = [[] for _ in range(n_points)]  # 每个个体支配的解集合
        n = np.zeros(n_points)  # 支配每个个体的解的数量
        rank = np.zeros(n_points, dtype=int)  # 每个个体的支配等级
        
        # 对每对个体比较支配关系
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # 如果i支配j
                    if np.all(f[i] <= f[j]) and np.any(f[i] < f[j]):
                        S[i].append(j)
                    # 如果j支配i
                    elif np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
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
    
    def crowding_distance(self, f, front):
        """计算拥挤度距离"""
        n_points = len(front)
        if n_points <= 2:
            return np.full(n_points, np.inf)
        
        dist = np.zeros(n_points)
        
        for obj in range(self.problem.n_obj):
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
    
    def tournament_selection(self, rank, crowding_dist, k=2):
        """锦标赛选择"""
        selected = np.zeros(self.pop_size, dtype=int)
        
        for i in range(self.pop_size):
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
    
    def simulated_binary_crossover(self, x1, x2):
        """模拟二进制交叉"""
        if np.random.random() > self.crossover_prob:
            return x1.copy(), x2.copy()
        
        x1, x2 = x1.copy(), x2.copy()
        
        # 对每个变量执行交叉
        for i in range(self.problem.n_var):
            if np.random.random() <= 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if x1[i] < x2[i]:
                        y1, y2 = x1[i], x2[i]
                    else:
                        y1, y2 = x2[i], x1[i]
                    
                    xl, xu = self.problem.xl[i], self.problem.xu[i]
                    
                    # 计算beta
                    beta = 1.0 + (2.0 * (y1 - xl) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(self.crossover_eta + 1.0))
                    rand = np.random.random()
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (self.crossover_eta + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.crossover_eta + 1.0))
                    
                    # 计算子代
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    # 边界处理
                    c1 = max(min(c1, xu), xl)
                    c2 = max(min(c2, xu), xl)
                    
                    # 交换回原来的顺序
                    if x1[i] > x2[i]:
                        x1[i], x2[i] = c2, c1
                    else:
                        x1[i], x2[i] = c1, c2
        
        return x1, x2
    
    def polynomial_mutation(self, x):
        """多项式变异"""
        y = x.copy()
        
        for i in range(self.problem.n_var):
            if np.random.random() <= self.mutation_prob:
                xl, xu = self.problem.xl[i], self.problem.xu[i]
                delta1 = (y[i] - xl) / (xu - xl)
                delta2 = (xu - y[i]) / (xu - xl)
                
                rand = np.random.random()
                mut_pow = 1.0 / (self.mutation_eta + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.mutation_eta + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.mutation_eta + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                
                y[i] += delta_q * (xu - xl)
                y[i] = max(min(y[i], xu), xl)
        
        return y
    
    def run(self, n_gen=200, verbose=True):
        """运行NSGA-II算法"""
        # 初始化种群
        x, f = self.initialize_population()
        history = {'x': [x.copy()], 'f': [f.copy()], 'fronts': []}
        
        # 主循环
        for gen in range(n_gen):
            if verbose and gen % 10 == 0:
                print(f"Generation {gen}")
            
            # 非支配排序
            fronts, rank = self.fast_non_dominated_sort(f)
            
            # 计算拥挤度
            crowding_dist = np.zeros(self.pop_size)
            for front in fronts:
                if len(front) > 0:  # 确保front不为空
                    crowding_dist[front] = self.crowding_distance(f, front)
            
            # 选择父代 parents_x shape(100, 30) = x
            selected = self.tournament_selection(rank, crowding_dist)
            parents_x = x[selected]
            
            # 通过交叉和变异生成子代
            offspring_x = np.zeros_like(parents_x)
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    x1, x2 = self.simulated_binary_crossover(parents_x[i], parents_x[i+1])
                    offspring_x[i] = self.polynomial_mutation(x1)
                    offspring_x[i+1] = self.polynomial_mutation(x2)
                else:
                    offspring_x[i] = self.polynomial_mutation(parents_x[i])
            
            # 评估子代
            offspring_f = self.problem.evaluate(offspring_x)
            
            # 合并父代和子代
            combined_x = np.vstack([x, offspring_x])
            combined_f = np.vstack([f, offspring_f])
            
            # 非支配排序
            fronts, rank = self.fast_non_dominated_sort(combined_f)
            
            # 选择下一代种群
            new_x = np.zeros((self.pop_size, self.problem.n_var))
            new_f = np.zeros((self.pop_size, self.problem.n_obj))
            
            # 按非支配等级和拥挤度选择
            count = 0
            front_index = 0
            
            # 按支配级别从前往后填充新种群
            while front_index < len(fronts) and count + len(fronts[front_index]) <= self.pop_size:
                for j in fronts[front_index]:
                    new_x[count] = combined_x[j]
                    new_f[count] = combined_f[j]
                    count += 1
                front_index += 1
            
            # 如果还有空位，按拥挤度填充最后一个前沿的部分解
            if count < self.pop_size and front_index < len(fronts):
                # 计算最后一个front的拥挤度
                last_front = fronts[front_index]
                last_front_crowding = self.crowding_distance(combined_f, last_front)
                
                # 按拥挤度降序排序
                last_front = np.array(last_front)
                idx = np.argsort(-last_front_crowding)
                sorted_front = last_front[idx]
                
                # 选择拥挤度最大的个体
                remaining = self.pop_size - count
                for j in sorted_front[:remaining]:
                    new_x[count] = combined_x[j]
                    new_f[count] = combined_f[j]
                    count += 1
            
            # 更新当前种群
            x = new_x
            f = new_f
            
            # 保存历史
            history['x'].append(x.copy())
            history['f'].append(f.copy())
            
            # 保存非支配解集
            current_fronts, _ = self.fast_non_dominated_sort(f)
            if len(current_fronts) > 0 and len(current_fronts[0]) > 0:
                history['fronts'].append(current_fronts[0])
            else:
                history['fronts'].append([])
        
        # 最终的非支配排序
        final_fronts, _ = self.fast_non_dominated_sort(f)
        
        # 确保有有效的非支配解
        if len(final_fronts) > 0 and len(final_fronts[0]) > 0:
            pareto_x = x[final_fronts[0]]
            pareto_f = f[final_fronts[0]]
        else:
            # 如果没有找到非支配解，返回整个种群
            pareto_x = x
            pareto_f = f
        
        return {
            'x': pareto_x,  # 非支配解的决策变量
            'f': pareto_f,  # 非支配解的目标函数值
            'history': history        # 优化历史
        }


def plot_pareto_front(f, title="Pareto Front"):
    """绘制Pareto前沿"""
    plt.figure(figsize=(10, 6))
    plt.scatter(f[:, 0], f[:, 1], s=30, facecolors='none', edgecolors='blue', label="NSGA-II")
    
    # 添加真实的Pareto前沿（理论值）
    x = np.linspace(0, 1, 100)
    y = 1 - np.sqrt(x)
    plt.plot(x, y, 'r-', label="True Pareto Front")
    
    plt.xlabel("$f_1$")
    plt.ylabel("$f_2$")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt


# 主函数: 解决ZDT1问题
def main():
    # 设置随机种子
    np.random.seed(42)
    
    # 创建问题实例
    problem = ZDT1(n_var=30)
    
    # 创建NSGA-II算法实例
    nsga2 = NSGA2(
        problem=problem,
        pop_size=100,
        crossover_prob=0.9,
        crossover_eta=20,
        mutation_eta=20
    )
    
    # 运行算法
    print("开始优化...")
    try:
        result = nsga2.run(n_gen=200, verbose=True)
        print("优化完成！")
        
        # 输出结果信息
        f = result['f']
        print(f"非支配解数量: {f.shape[0]}")
        print(f"目标函数值范围 f1: [{min(f[:, 0])}, {max(f[:, 0])}]")
        print(f"目标函数值范围 f2: [{min(f[:, 1])}, {max(f[:, 1])}]")
        
        # 绘制Pareto前沿
        plt = plot_pareto_front(f, title="ZDT1 - NSGA-II")
        plt.savefig("zdt1_pareto_front_custom.png")
        plt.show()
        
        return result
    except Exception as e:
        print(f"优化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()