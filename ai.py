import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.misc import stack
from pymoo.optimize import minimize


# -----------------------------
# Custom Problem: x1, x2, x3 structure
# -----------------------------
class MyProblem(Problem):
    def __init__(self, len_x1=2, len_x2=2, len_x3=2):
        self.len_x1 = len_x1
        self.len_x2 = len_x2
        self.len_x3 = len_x3
        n_var = len_x1 + len_x2 + len_x3
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        x1 = X[:, :self.len_x1]
        x2 = X[:, self.len_x1:self.len_x1+self.len_x2]
        x3 = X[:, -self.len_x3:]

        f1 = np.sum((x1 - 0.5)**2, axis=1) + np.sum(x3, axis=1)
        f2 = np.sum((x2 - 0.2)**2, axis=1) + np.sum((x3 - 1.0)**2, axis=1)

        out["F"] = np.column_stack([f1, f2])


# -----------------------------
# SubPopulation Class
# -----------------------------
class SubPopulation:
    def __init__(self, problem, x3_shared, pop_size=20):
        self.problem = problem
        self.len_x3 = len(x3_shared)
        self.x3_shared = x3_shared
        self.pop_size = pop_size

        self.sampling = FloatRandomSampling()
        self.crossover = SBX(prob=1.0, eta=15)
        self.mutation = PM(eta=20)

        self.partial_pop = self.sampling.do(self.problem, self.pop_size)
        self.update_full_pop()

    def update_full_pop(self):
        # if self.partial_pop.ndim == 1:
        #     self.partial_pop = self.partial_pop[np.newaxis, :]
        # x3 = np.tile(self.x3_shared, (self.pop_size, 1))
        # self.full_pop = np.hstack([self.partial_pop, x3])
        self.full_pop = self.partial_pop.copy()
        for _,ind in enumerate(self.partial_pop):
            # ind.X = np.hstack([ind.X[0:4], self.x3_shared])
            ind.X = ind.X[:self.problem.n_var - self.len_x3]
            
        for _, ind in enumerate(self.full_pop):
            ind.X = np.hstack([ind.X, self.x3_shared])
        
        self.problem.evaluate([self.full_pop[i].X for i in range(self.full_pop.size)], out={"F": None})

    def evolve(self, n_gen=1):
        for _ in range(n_gen):
            parents = self.partial_pop[np.random.randint(0, self.pop_size, (self.pop_size, 2))]
            off = self.crossover.do(self.problem, parents)
            off = self.mutation.do(self.problem, off)
            off_full = np.hstack([off, np.tile(self.x3_shared, (len(off), 1))])
            self.problem.evaluate(off_full, out={"F": None})

            combined_partial = np.vstack([self.partial_pop, off])
            combined_full = np.vstack([self.full_pop, off_full])
            F = self.problem.evaluate(combined_full, return_values_of=["F"])
            fronts = NonDominatedSorting().do(F, self.pop_size)
            I = fronts[0][:self.pop_size]

            self.partial_pop = combined_partial[I]
            self.full_pop = combined_full[I]

    def mutate_shared_x3(self, prob=0.2):
        mask = np.random.rand(self.len_x3) < prob
        self.x3_shared[mask] += np.random.normal(0, 0.1, mask.sum())
        self.x3_shared = np.clip(self.x3_shared, 0, 1)
        self.update_full_pop()

    def get_representative(self):
        F = self.problem.evaluate(self.full_pop, return_values_of=["F"])
        best_idx = np.lexsort((F[:, 1], F[:, 0]))[0]  # sort by f1 then f2
        return self.full_pop[best_idx], F[best_idx]


# -----------------------------
# MultiPopulationNSGA2 Algorithm
# -----------------------------
class MultiPopulationNSGA2(Algorithm):
    def __init__(self, n_subpops, subpop_args, n_survive=None):
        super().__init__()
        self.n_subpops = n_subpops
        self.subpop_args = subpop_args
        self.n_survive = n_survive or n_subpops

    def _setup(self, problem, **kwargs):
        self.subpops = [
            SubPopulation(problem, x3_shared=np.random.rand(problem.len_x3), **self.subpop_args)
            for _ in range(self.n_subpops)
        ]

    def _step(self):
        for sub in self.subpops:
            sub.evolve(n_gen=1)
            sub.mutate_shared_x3()

        reps = [sub.get_representative() for sub in self.subpops]
        X = np.array([r[0] for r in reps])
        F = np.array([r[1] for r in reps])

        fronts = NonDominatedSorting().do(F, self.n_survive)
        selected_indices = fronts[0][:self.n_survive]
        self.subpops = [self.subpops[i] for i in selected_indices]

    def _finalize(self):
        all_individuals = [ind for sub in self.subpops for ind in sub.full_pop]
        return np.array(all_individuals)


# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    problem = MyProblem(len_x1=2, len_x2=2, len_x3=2)
    algo = MultiPopulationNSGA2(
        n_subpops=10,
        subpop_args={"pop_size": 20},
        n_survive=5
    )
    algo.setup(problem)

    # for gen in range(50):
    #     algo.step()
    #     print(f"Generation {gen+1} done, surviving subpopulations: {len(algo.subpops)}")

    # final_pop = algo.finalize()
    # print("Final population shape:", final_pop.shape)
    
    res = minimize(
        algo,
        problem,
        termination=('n_gen', 50),
        seed=1,
        verbose=True
    )
    
    print(res.X)
