import numpy as np
import torch
from structure_matrix_learner import StructureMatrixLearner

# 输入 X: n × 3
X = np.random.rand(100, 3)

# 定义结构损失: 希望 Y[:, 2] ≈ 2 * Y[:, 0]
def structure_loss(Y):
    return torch.var((Y[:, 2] / 2 * Y[:, 0]) ** 2, dim=0).sum()

# def structure_loss(Y):
#     return torch.mean((Y[:, 2] - 2 * Y[:, 0]) ** 2)

# def structure_loss(Y):
#     return torch.var(Y, dim=0).sum()

# 实例化学习器
learner = StructureMatrixLearner(
    X=X,
    structure_fn=structure_loss,
    A_shape=(3, 3),
    lr=0.01,
    optimizer="adamw",
    weight_decay=1e-3,
    verbose=True
)

# 训练
learner.train(epochs=1000)

# 可视化 loss
learner.plot_loss()

# 查看 A 和 Y
print("Learned A:")
print(learner.get_A())

print("Learned Y:")
print(X @ learner.get_A())

print("Structure residual:", structure_loss(torch.tensor(learner.get_Y())).item())
