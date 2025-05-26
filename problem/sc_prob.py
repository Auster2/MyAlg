

import numpy as np

class SharedComponents:
    def __init__(self, sub_prob, n_var=30, same_idx=[2, 3]):
        self.sub_prob = sub_prob(n_var)
        self.n_constr = self.sub_prob.n_constr + 1  # 添加结构约束
        self.n_var = n_var
    
    def evaluate(self, pop):
        """计算所有子群个体的目标函数和约束"""
        whole_f = []
        whole_cv = []
        whole_scv = []
        
        for sub_pop in pop:  # 每个子种群
            f, cv = self.sub_prob.evaluate(sub_pop)  # cv_base: (n, 2)
            n = len(sub_pop)
            cv_struct = np.zeros((n,))  # 每个个体对应一个结构约束值
            
            # 结构约束：对子群中每对个体 (i, j)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = abs(sub_pop[i][2] - sub_pop[j][2]) + abs(sub_pop[i][3] - sub_pop[j][3])
                    cv_struct[i] += diff
                    cv_struct[j] += diff  # 共享惩罚
            
            whole_f.append(f)
            whole_cv.append(cv)
            whole_scv.append(cv_struct)
        
        whole_f = np.array(whole_f)
        whole_cv = np.array(whole_cv)
        whole_scv = np.array(whole_scv)
        
        return whole_f, whole_cv, whole_scv