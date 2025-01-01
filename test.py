import torch
import numpy as np

a = torch.tensor(np.array([[1, 2, 3],[4, 5, 6]]))
b = torch.tensor(np.array([[4, 5, 6]]))
print(torch.concat((a, b), dim=0))