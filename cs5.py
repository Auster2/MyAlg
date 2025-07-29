# compare

# epsl
from problem import CSMOP1, CSMOP2
import pandas as pd
from alg.epsl_run import EPSLRunner
import matplotlib.pyplot as plt
from alg.nsga import NSGA2
from pymoo.algorithms.moo.nsga2 import NSGA2 as PymooNSGA2
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体

hv_value_true = 0.87661645  # 真实的超体积值
n_steps = 200
n_sample = 5
n_pref_update = 10
PROB = CSMOP1  
pop_size = 300  # 设置种群大小
n_gen = 200  # 设置迭代次数

prob = PROB(n_var=12)

file_name = 'nsga2_' + PROB.__name__ + f'_{pop_size}_{n_gen}'
nsga2 = NSGA2(problem=prob, pop_size=pop_size, n_gen=n_gen, crossover_prob=0.9, crossover_eta=20, mutation_prob=0.1, mutation_eta=20)
res = nsga2.run()

plt.plot(res['f'][:, 0], res['f'][:, 1], 'o', label='NSGA-II Result')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('NSGA-II Result Visualization')
plt.legend()
plt.xlim(prob.ideal_point[0], prob.nadir_point[0])
plt.ylim(prob.ideal_point[1], prob.nadir_point[1])
plt.savefig(f"data/{file_name}_f.png")

# epsl_runner = EPSLRunner(problem=prob, hv_value_true=hv_value_true, n_steps=n_steps, n_sample=n_sample, n_pref_update=n_pref_update, device='cpu', n_run=1)

# file_name = "epsl" + PROB.__name__ +  "_{n_steps}_{n_sample}_{n_pref_update}".format(prob=prob, n_steps=n_steps, n_sample=n_sample, n_pref_update=n_pref_update)
# x, f1, hv = epsl_runner.run_once()
# print("Result shape:", x.shape)
# pd.DataFrame(x, columns=[f"x{i+1}" for i in range(x.shape[1])]).to_csv(f"data/{file_name}_x.csv", index=False)
# # 绘制原pf

# plt.plot(f1[:, 0], f1[:, 1], 'o', label='EPSL Result')
# plt.xlabel('Objective 1')
# plt.ylabel('Objective 2')
# plt.title('EPSL Result Visualization')
# plt.legend()
# plt.xlim(prob.ideal_point[0], prob.nadir_point[0])
# plt.ylim(prob.ideal_point[1], prob.nadir_point[1])
# # plt.show()
# plt.savefig(f"data/{file_name}_f.png")
# print("EPSLRunner test completed.")

# my_alg

# from alg import NSGA
# from problem import SharedComponents

# wrapper = SharedComponents(sub_prob=CSMOP1, same_idx=[6, 7, 8, 9, 10, 11], n_var=12)
# my_alg = NSGA(problem=wrapper, pop_size=30, sub_pop_size=30)
# f2, _, hv = my_alg.run(generations=400)


# print("My Algorithm Result shape:", f2.shape)
# x0 = f2[0]
# plt.plot(x0[:, 0], x0[:, 1], 'o', label='My Algorithm Result')
# plt.xlabel('Objective 1')
# plt.ylabel('Objective 2')
# plt.title('My Algorithm Result Visualization')
# plt.legend()
# plt.show()
# plt.savefig("data/my_alg_csmop1.png")