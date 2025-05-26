
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
    
    def evaluate_scv(self, pop):
        """计算所有子群个体的结构约束违反量"""
        whole_scv = []
        
        for sub_pop in pop:
            n = len(sub_pop)
            cv_struct = np.zeros((n,))
            
            for i in range(n):
                for j in range(i + 1, n):
                    diff = abs(sub_pop[i][2] - sub_pop[j][2]) + abs(sub_pop[i][3] - sub_pop[j][3])
                    cv_struct[i] += diff
                    cv_struct[j] += diff
        
            whole_scv.append(cv_struct)
        
        whole_scv = np.array(whole_scv)
        return whole_scv