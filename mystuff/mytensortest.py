#%%
import torch
import seaborn as sns
import numpy as np

#%%
x = torch.tensor(4., requires_grad=True)
z = torch.tensor(2., requires_grad=True)

def function(X):
    return (X ** 2) - (3 * X)  + 4

x_range = np.linspace(0, 100, 101)
y_range = [function(i) for i in x_range]


sns.lineplot(x=x_range, y=y_range)



y = (2 * x) **3 - (5 * z) **2 


   
y.backward()
print(x.grad)
print(z.grad)


# %%
