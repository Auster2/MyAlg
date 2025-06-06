import numpy as np

class SharedComponents:
    def __init__(self, sub_prob, n_var=30, same_idx=[2, 3]):
        self.sub_prob = sub_prob(n_var)
        self.n_constr = self.sub_prob.n_constr + 1  # 添加结构约束
        self.n_var = n_var
        self.same_idx = same_idx
    
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
                    # diff = abs(sub_pop[i][2] - sub_pop[j][2]) + abs(sub_pop[i][3] - sub_pop[j][3])
                    diff = np.sum(np.abs(sub_pop[i][self.same_idx] - sub_pop[j][self.same_idx]))
                    cv_struct[i] += diff
                    cv_struct[j] += diff  # 共享惩罚
            
            whole_f.append(f)
            whole_cv.append(cv)
            whole_scv.append(cv_struct)
        
        whole_f = np.array(whole_f)
        whole_cv = np.array(whole_cv)
        whole_scv = np.array(whole_scv)
        
        return whole_f, whole_cv, whole_scv
    
    def evaluate_scv(self, pop):
        """计算所有子群个体的结构约束违反量"""
        whole_scv = []
        
        for sub_pop in pop:
            n = len(sub_pop)
            cv_struct = np.zeros((n,))
            
            for i in range(n):
                for j in range(i + 1, n):
                    diff = np.sum(np.abs(sub_pop[i][self.same_idx] - sub_pop[j][self.same_idx]))
                    cv_struct[i] += diff
                    cv_struct[j] += diff
        
            whole_scv.append(cv_struct)
        
        whole_scv = np.array(whole_scv)
        return whole_scv
    
class SharedX_0:
    def __init__(self, sub_prob, n_var=30, same_val=[0.5]):
        self.sub_prob = sub_prob(n_var)
        self.n_constr = self.sub_prob.n_constr + 1  # 添加结构约束
        self.n_var = n_var
        self.same_val = same_val
    
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
                cv_struct[i] = np.abs(sub_pop[i, 0] - self.same_val[0])
            
            whole_f.append(f)
            whole_cv.append(cv)
            whole_scv.append(cv_struct)
        
        whole_f = np.array(whole_f)
        whole_cv = np.array(whole_cv)
        whole_scv = np.array(whole_scv)
        
        return whole_f, whole_cv, whole_scv
    
    def evaluate_scv(self, pop):
        """计算所有子群个体的结构约束违反量"""
        whole_scv = []
        
        for sub_pop in pop:
            n = len(sub_pop)
            cv_struct = np.zeros((n,))
            for i in range(n):
                cv_struct[i] = np.abs(sub_pop[i, 0] - self.same_val[0])
            whole_scv.append(cv_struct)
        
        whole_scv = np.array(whole_scv)
        return whole_scv
        
class SharedX_1:
    def __init__(self, sub_prob, n_var=30, same_val=[0.5]):
        self.sub_prob = sub_prob(n_var)
        self.n_constr = self.sub_prob.n_constr + 1  # 添加结构约束
        self.n_var = n_var
        self.same_val = same_val
    
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
                cv_struct[i] = np.abs(sub_pop[i, 1] - self.same_val[0])
            
            whole_f.append(f)
            whole_cv.append(cv)
            whole_scv.append(cv_struct)
        
        whole_f = np.array(whole_f)
        whole_cv = np.array(whole_cv)
        whole_scv = np.array(whole_scv)
        
        return whole_f, whole_cv, whole_scv
    
    def evaluate_scv(self, pop):
        """计算所有子群个体的结构约束违反量"""
        whole_scv = []
        
        for sub_pop in pop:
            n = len(sub_pop)
            cv_struct = np.zeros((n,))
            for i in range(n):
                cv_struct[i] = np.abs(sub_pop[i, 1] - self.same_val[0])
            whole_scv.append(cv_struct)
        
        whole_scv = np.array(whole_scv)
        return whole_scv
    
class VarialbeRelationship:
    def __init__(self, sub_prob, n_var=30, x_idx=2, y_idx=1):
        self.sub_prob = sub_prob(n_var)
        self.n_constr = self.sub_prob.n_constr + 1  # 添加结构约束
        self.n_var = n_var
        self.x_idx = x_idx
        self.y_idx = y_idx
    
    def evaluate(self, pop):
        """计算所有子群个体的目标函数和约束"""
        whole_f = []
        whole_cv = []
        whole_scv = []
        
        for sub_pop in pop:  # 每个子种群
            f, cv = self.sub_prob.evaluate(sub_pop)  # cv_base: (n, 2)
            n = len(sub_pop)
            cv_struct = np.zeros((n,))  # 每个个体对应一个结构约束值
            
            scv_list = sub_pop[:, self.y_idx] / sub_pop[:, self.x_idx]    
            # 结构约束：对子群中每对个体 (i, j)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = np.abs(scv_list[i] - scv_list[j])
                    cv_struct[i] += diff
                    cv_struct[j] += diff  # 共享惩罚
            
            whole_f.append(f)
            whole_cv.append(cv)
            whole_scv.append(cv_struct)
        
        whole_f = np.array(whole_f)
        whole_cv = np.array(whole_cv)
        whole_scv = np.array(whole_scv)
        
        return whole_f, whole_cv, whole_scv
    
    def evaluate_scv(self, pop):
        """计算所有子群个体的结构约束违反量"""
        whole_scv = []
        
        for sub_pop in pop:  # 每个子种群
            n = len(sub_pop)
            cv_struct = np.zeros((n,))
            scv_list = sub_pop[:, self.y_idx] / sub_pop[:, self.x_idx]   
            
            for i in range(n):
                for j in range(i + 1, n):
                    diff = np.abs(scv_list[i] - scv_list[j])
                    cv_struct[i] += diff
                    cv_struct[j] += diff  # 共享惩罚
            whole_scv.append(cv_struct)

        whole_scv = np.array(whole_scv)
        return whole_scv
    