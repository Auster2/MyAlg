import numpy as np

class ZDT1:
    def __init__(self, n_var=30):
        self.n_var = n_var
        self.n_obj = 2
        self.n_constr = 2  # 基本约束数量（不包括结构约束）
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
    
    def evaluate(self, x):
        """计算目标函数值和约束违反量"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        f = np.zeros((x.shape[0], 2))
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
        
        f[:, 0] = x[:, 0]
        f[:, 1] = g * (1 - np.sqrt(x[:, 0] / g))
        
        # 原始两个约束
        cv = np.zeros((x.shape[0], self.n_constr))
        cv[:, 0] = 0.2 - x[:, 0] - x[:, 1]  # 约束1
        cv[:, 1] = f[:, 0] + f[:, 1] - 1.2  # 约束2
        
        return f, cv 
    
    def pareto_front(self, n_points=100):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return np.vstack([f1, f2]).T
    
class ZDT1NoCV:
    def __init__(self, n_var=30):
        self.n_var = n_var
        self.n_obj = 2
        self.n_constr = 0  # 基本约束数量（不包括结构约束）
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
    
    def evaluate(self, x):
        """计算目标函数值和约束违反量"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        f = np.zeros((x.shape[0], 2))
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
        
        f[:, 0] = x[:, 0]
        f[:, 1] = g * (1 - np.sqrt(x[:, 0] / g))
        
        return f, np.zeros((x.shape[0], 1))
    
    def pareto_front(self, n_points=100):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return np.vstack([f1, f2]).T
    
class DTLZ2:
    def __init__(self, n_var=12, n_obj=3):
        self.n_var = n_var
        self.n_obj = n_obj
        self.k = n_var - n_obj + 1
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)

    def evaluate(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        g = np.sum((x[:, -self.k:] - 0.5) ** 2, axis=1)

        f = np.ones((x.shape[0], self.n_obj))
        for i in range(self.n_obj):
            for j in range(self.n_obj - i - 1):
                f[:, i] *= np.cos(x[:, j] * 0.5 * np.pi)
            if i > 0:
                f[:, i] *= np.sin(x[:, self.n_obj - i - 1] * 0.5 * np.pi)
            f[:, i] *= (1 + g)
        return f, None

    def pareto_front(self, n_points=100):
        from scipy.stats import qmc
        # 均匀采样单位球面上的点
        sampler = qmc.Sobol(d=self.n_obj, scramble=True)
        points = sampler.random_base2(m=int(np.log2(n_points)))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms