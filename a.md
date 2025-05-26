$$
\begin{bmatrix}
    x_{11} & x_{12} & x_{13} \\
    x_{21} & x_{22} & x_{23} \\
    x_{31} & x_{32} & x_{33} \\
    \vdots & \vdots & \vdots \\
    x_{n1} & x_{n2} & x_{n3}
\end{bmatrix} \cdot 
\begin{bmatrix}
    a_1 & a_2 & a_3 \\
    b_1 & b_2 & b_3 \\
    c_1 & c_2 & c_3
\end{bmatrix} = 
\begin{bmatrix}
    y_{11} & y_{12} & y_{13} \\
    y_{21} & y_{22} & y_{23} \\
    y_{31} & y_{32} & y_{33} \\
    \vdots & \vdots & \vdots \\
    y_{n1} & y_{n2} & y_{n3}
\end{bmatrix} \\
X \cdot A = Y
$$

我能否训练一个模型 $A$ , 通过Y的structure constraints value，训练出一个适合的A

```py
def evaluate_loss(self, Ys):
    loss = []
    
    for Y in Ys:
        n = len(Y)
        sub_loss = np.zeros((n,))

        for i in range(n):
            for j in range(i + 1, n):
                diff = abs(Y[i][1] - Y[j][1]) + abs(Y[i][2] - Y[j][2])
                sub_loss[i] += diff
                sub_loss[j] += diff
    
        loss.append(sub_loss)
    
    loss = np.array(loss)
    return loss
```

我发现上面的代码是通过无限缩小A来减小方差，这样是无效的，有没有办法避免？
而且我把问题改成
def structure_constraint(Y):
    return torch.sum(torch.abs(Y[:, 2] - 2 * Y[:, 0]))
后，按道理来说是有解的，但是A总是在2.几不会继续减小，是为什么呢
Epoch 99100, Structure Loss: 2.3343
Epoch 99200, Structure Loss: 2.3446
Epoch 99300, Structure Loss: 2.3278
Epoch 99400, Structure Loss: 2.3578
Epoch 99500, Structure Loss: 2.3665
Epoch 99600, Structure Loss: 2.3404
Epoch 99700, Structure Loss: 2.3400
Epoch 99800, Structure Loss: 2.3397
Epoch 99900, Structure Loss: 2.3400

# A = torch.nn.Parameter(torch.tensor([[1.0, 1.0, 2.0],
#                                     [1.0, 1.0, 2.0],
#                                     [1.0, 1.0, 2.0]], dtype=torch.float32))