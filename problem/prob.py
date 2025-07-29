import numpy as np

class CSMOP1:
    def __init__(self, n_var=12):
        self.n_var = n_var          # 非共享变量维度
        self.n_obj = 2
        self.n_constr = 0
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self.nadir_point = np.array([1.0 + 1.5, 10.0 + 1.0])
        self.ideal_point = np.array([0.0 - 1.5, 0.0 - 1.0])
    
    def T(self, s):
        """共享变量s的变换函数T(s)"""
        s = np.atleast_2d(s)
        return np.mean(np.sqrt(s), axis=1)  # shape: (batch_size,)
    
    def g(self, x):
        return 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
    
    def evaluate(self, x):
        """计算目标函数值"""
        s = x[:, 6:]  # 共享变量
        x = x[:, :6]
        x = np.atleast_2d(x)
        s = np.atleast_2d(s)
        N = x.shape[0]

        t_vals = self.T(s)  # shape: (N,)
        g_vals = self.g(x)

        f1 = x[:, 0] + 1.5 * np.cos(0.5 * np.pi * t_vals)
        f2 = g_vals * (1 - np.sqrt(x[:, 0] / g_vals)) + np.sin(0.5 * np.pi * t_vals)

        f = np.stack([f1, f2], axis=1)
        cv = np.zeros((N, self.n_constr))
        return f, cv
    
    def pareto_front(self, n_points=100, s_val=None):
        """获取某个s下的Pareto前沿近似解（近似RF）"""
        f1 = np.linspace(0, 1, n_points)
        if s_val is None:
            t_val = 0  # 默认T(s)=0
        else:
            t_val = np.mean(np.sqrt(s_val))
        
        f2 = 1 - np.sqrt(f1) + np.sin(0.5 * np.pi * t_val)
        f1 = f1 + 1.5 * np.cos(0.5 * np.pi * t_val)
        return np.vstack([f1, f2]).T

class CSMOP2:
    def __init__(self, n_var=12):
        self.n_var = n_var
        self.n_obj = 2
        self.n_constr = 0
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self.nadir_point = np.array([1.0 + 1.0, 10.0 + 1.0])
        self.ideal_point = np.array([0.0 - 1.0, 0.0 - 1.0])
    
    def T(self, s):
        s = np.atleast_2d(s)
        return np.mean(s * (50 * s**2 - 65 * s + 22), axis=1) / 7
    
    def g(self, x):
        return 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (x.shape[1] - 1)

    def evaluate(self, x):
        s = x[:, 6:]
        x = x[:, :6]
        x = np.atleast_2d(x)
        s = np.atleast_2d(s)
        N = x.shape[0]
        
        t_vals = self.T(s)
        g_vals = self.g(x)

        f1 = x[:, 0] + 1.0 - t_vals
        f2 = g_vals * (1 - (x[:, 0] / g_vals)**2) + 0.7 * t_vals

        f = np.stack([f1, f2], axis=1)
        cv = np.zeros((N, self.n_constr))
        return f, cv

    def pareto_front(self, n_points=100, s_val=None):
        f1 = np.linspace(0, 1, n_points)
        if s_val is None:
            t_val = 0
        else:
            t_val = np.mean(s_val * (50 * s_val**2 - 65 * s_val + 22)) / 7

        f2 = 1 - f1**2 + 0.7 * t_val
        f1 = f1 + 1.0 - t_val
        return np.vstack([f1, f2]).T

class CSMOP3:
    def __init__(self, n_var=12):
        self.n_var = n_var
        self.n_obj = 2
        self.n_constr = 0
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self.nadir_point = np.array([1.0 + 0.5, 2.0 + 0.9])
        self.ideal_point = np.array([0.0 - 0.5, 0.0 - 0.9])

    def T(self, s):
        s = np.atleast_2d(s)
        return np.mean(np.sqrt(s), axis=1)

    def g(self, x):
        return 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (x.shape[1] - 1)

    def evaluate(self, x):
        s = x[:, 6:]
        x = x[:, :6]
        x = np.atleast_2d(x)
        s = np.atleast_2d(s)
        N = x.shape[0]

        t_vals = self.T(s)
        g_vals = self.g(x)

        f1 = x[:, 0] + 0.5 * t_vals
        f2 = 1.5 * g_vals * (1 - np.sqrt(x[:, 0] / g_vals) - (x[:, 0] / g_vals) * np.sin(8.5 * np.pi * x[:, 0])) + 0.9 - 0.9 * t_vals

        f = np.stack([f1, f2], axis=1)
        cv = np.zeros((N, self.n_constr))
        return f, cv

    def pareto_front(self, n_points=100, s_val=None):
        f1 = np.linspace(0, 1, n_points)
        if s_val is None:
            t_val = 0
        else:
            t_val = np.mean(np.sqrt(s_val))

        f2 = 1.5 * (1 - np.sqrt(f1) - f1 * np.sin(8.5 * np.pi * f1)) + 0.9 - 0.9 * t_val
        f1 = f1 + 0.5 * t_val
        return np.vstack([f1, f2]).T

class CSMOP4:
    def __init__(self, n_var=12):
        self.n_var = n_var
        self.n_obj = 2
        self.n_constr = 0
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self.nadir_point = np.array([3.0 + 0.7, 3.0 + 1.0])
        self.ideal_point = np.array([0.0 - 0.7, 0.0 - 1.0])

    def T(self, s):
        s = np.atleast_2d(s)
        return np.mean(s * (50 * s**2 - 65 * s + 22), axis=1) / 7

    def evaluate(self, x):
        s = x[:, 6:]
        x = x[:, :6]
        x = np.atleast_2d(x)
        s = np.atleast_2d(s)
        N = x.shape[0]

        J1 = np.arange(1, x.shape[1], 2)
        J2 = np.arange(2, x.shape[1], 2)

        term1 = np.sum((x[:, J1] - (0.5 + 3 * (J1 - 1) / (x.shape[1] - 2)))**2, axis=1)
        term2 = np.sum((x[:, J2] - (0.5 + 3 * (J2 - 1) / (x.shape[1] - 2)))**2, axis=1)

        t_vals = self.T(s)

        f1 = x[:, 0] + 2 / len(J1) * term1 + 0.7 * t_vals
        f2 = 3 - 2 * x[:, 0] + 2 / len(J2) * term2 - t_vals

        f = np.stack([f1, f2], axis=1)
        cv = np.zeros((N, self.n_constr))
        return f, cv

    def pareto_front(self, n_points=100, s_val=None):
        x1 = np.linspace(0, 1.5, n_points)
        if s_val is None:
            t_val = 0
        else:
            t_val = np.mean(s_val * (50 * s_val**2 - 65 * s_val + 22)) / 7

        f1 = x1 + 0.7 * t_val
        f2 = 3 - 2 * x1 - t_val
        return np.vstack([f1, f2]).T


class ZDT1:
    def __init__(self, n_var=30):
        self.n_var = n_var
        self.n_obj = 2
        self.n_constr = 2  # 基本约束数量（不包括结构约束）
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self.nadir_point = np.array([1.0, 1.0])
        self.ideal_point = np.array([0.0, 0.0])
    
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
        self.nadir_point = np.array([1.0, 1.0])
        self.ideal_point = np.array([0.0, 0.0])
    
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
        self.n_constr = 0  # 基本约束数量（不包括结构约束）
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
        return f, np.zeros((x.shape[0], 1))

    def pareto_front(self, n_points=100):
        from scipy.stats import qmc
        # 均匀采样单位球面上的点
        sampler = qmc.Sobol(d=self.n_obj, scramble=True)
        points = sampler.random_base2(m=int(np.log2(n_points)))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms