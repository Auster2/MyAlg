import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class StructureMatrixLearner:
    def __init__(
        self,
        X,                        # 输入矩阵 X: shape (n, d)
        structure_fn,            # 结构约束函数: structure_fn(Y) → scalar
        A_shape=(3, 3),          # A 的形状: (input_dim, output_dim)
        lr=0.01,                 # 学习率
        optimizer="adam",        # 优化器选择: 'sgd', 'adam', 'adamw'
        weight_decay=0.0,        # L2 正则
        verbose=True,            # 是否输出日志
        device="cpu"             # 可切换为 'cuda'
    ):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.structure_fn = structure_fn
        self.device = device
        self.verbose = verbose

        # 可学习参数 A
        self.A = nn.Parameter(torch.randn(*A_shape, dtype=torch.float32).to(device))

        # 优化器选择
        self.optimizer = self._build_optimizer(optimizer, lr, weight_decay)

        # 日志记录
        self.loss_history = []

    def _build_optimizer(self, opt_name, lr, wd):
        if opt_name.lower() == "sgd":
            return optim.SGD([self.A], lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name.lower() == "adam":
            return optim.Adam([self.A], lr=lr, weight_decay=wd)
        elif opt_name.lower() == "adamw":
            return optim.AdamW([self.A], lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def step(self):
        """执行一次训练步骤"""
        Y = self.X @ self.A
        loss = self.structure_fn(Y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())
        return loss.item()

    def train(self, epochs=10000, log_interval=100):
        for epoch in range(epochs):
            loss = self.step()
            if self.verbose and epoch % log_interval == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
            if loss < 1e-6:
                print(f"Converged at epoch {epoch}, Loss: {loss:.6f}")
                break

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Structure Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.show()

    def get_A(self):
        return self.A.detach().cpu().numpy()

    def get_Y(self):
        return (self.X @ self.A).detach().cpu().numpy()
