import os
import numpy as np
import torch
from pymoo.indicators.hv import HV

class EPSLRunner:
    def __init__(self, problem, hv_value_true, n_steps=4000, n_sample=10, n_pref_update=10,
                 sampling_method='Bernoulli-Shrinkage', device='cpu', n_run=1):
        self.n_steps = n_steps
        self.n_sample = n_sample
        self.n_pref_update = n_pref_update
        self.sampling_method = sampling_method
        self.device = device
        self.n_run = n_run
        self.problem = problem
        self.n_dim = self.problem.n_var
        self.n_obj = self.problem.n_obj
        self.hv_value_true = hv_value_true

        from model.model_syn_shared_component import ParetoSetModel
        self.ParetoSetModel = ParetoSetModel

        self.ref_point = torch.Tensor([1.1 * x for x in self.problem.nadir_point]).to(self.device)

    def run_once(self):
        psmodel = self.ParetoSetModel(self.n_dim, self.n_obj).to(self.device)
        optimizer = torch.optim.Adam(psmodel.parameters(), lr=0.0025)
        z = torch.ones(self.n_obj).to(self.device)
        ideal_point_tensor = torch.tensor(self.problem.ideal_point).to(self.device)
        ref_point = self.ref_point

        for t in range(self.n_steps):
            psmodel.train()
            optimizer.zero_grad()

            pref_vec = torch.tensor(np.random.dirichlet(np.ones(self.n_obj), self.n_pref_update)).float().to(self.device)
            x = psmodel(pref_vec)
            grad_es_list = []

            for k in range(pref_vec.shape[0]):
                delta = self._sample_delta()
                x_plus = x[k] + 0.01 * delta
                x_plus = torch.clamp(x_plus, 0, 1)
                x_plus = x_plus.detach().numpy()

                f_x_plus, _ = self.problem.evaluate(x_plus)
                f_x_plus = torch.tensor(f_x_plus).to(self.device)
                f_x_plus = (f_x_plus - ideal_point_tensor) / (ref_point - ideal_point_tensor)
                z = torch.min(torch.cat([z[None], f_x_plus - 0.1]), dim=0).values.data

                tch = self._compute_tch(f_x_plus, pref_vec[k], z)
                rank = torch.argsort(tch)
                tch_rank = torch.ones(len(tch)).to(self.device)
                tch_rank[rank] = torch.linspace(-0.5, 0.5, len(tch)).to(self.device)

                grad = 1.0 / (self.n_sample * 0.01) * torch.sum(tch_rank[:, None] * delta, dim=0)
                grad_es_list.append(grad)

            grad_es = torch.stack(grad_es_list)
            psmodel(pref_vec).backward(grad_es)
            optimizer.step()

        return self._evaluate_model(psmodel)

    def _sample_delta(self):
        if self.sampling_method == 'Gaussian':
            return torch.randn(self.n_sample, self.n_dim).to(self.device).double()
        if self.sampling_method == 'Bernoulli-Shrinkage':
            m = np.sqrt((self.n_sample + self.n_dim - 1) / (4 * self.n_sample))
            return ((torch.bernoulli(0.5 * torch.ones(self.n_sample, self.n_dim)) - 0.5) / m).to(self.device).double()
        raise ValueError(f"Unknown sampling method: {self.sampling_method}")

    def _compute_tch(self, f, pref_vec_k, z):
        u = 0.1
        return u * torch.logsumexp((1 / pref_vec_k) * torch.abs(f - z) / u, dim=1)

    def _evaluate_model(self, model):
        model.eval()
        with torch.no_grad():
            pref = torch.tensor(np.stack([np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)]).T).float().to(self.device)
            x = model(pref)
            x = x.detach().numpy()
            f, _ = self.problem.evaluate(x)
            f = torch.tensor(f)
            results_F_norm = (f - torch.tensor(self.problem.ideal_point).to(self.device)) / \
                             (torch.tensor(self.problem.nadir_point).to(self.device) - torch.tensor(self.problem.ideal_point).to(self.device))
            hv = HV(ref_point=self.ref_point.cpu().numpy())
            hv_val = hv(results_F_norm.cpu().numpy())
            print(f"Hypervolume: {hv_val:.4f}")
            return x, f.numpy(), hv_val
