#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% convert data to tensor
X = torch.tensor(cars['wt'].values, dtype=torch.float32).reshape(-1,1)
Y = torch.tensor(cars['mpg'].values, dtype=torch.float32)

w = torch.randn(1, requires_grad=True, dtype=torch.float32)
b = torch.randn(1, requires_grad=True, dtype=torch.float32)

sns.regplot(x='wt', y='mpg', data=cars)
print(f'x: {X.shape}, y: {Y.shape}`)')
print(X[0:5])
print(Y[0:5])
print(w)
print(b)


# %%

#%% training
lr = 0.002
numEpochs = 1000
netLosses = []
for i in range(numEpochs):
    losssum = 0
    # create prediction
    for index, x in enumerate(X):
    
        hidden = (x * w) + b
        y_pred = hidden

        # calculate loss
        loss = torch.pow(y_pred - Y[index], 2)
        # backward pass
        loss.backward()
        # update weights and biases
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            # print(w)
            # print(b)
            w.grad.zero_()
            
            b.grad.zero_()
        losssum += loss.item()
    netLosses.append(losssum / len(X))
    if i % 100 == 0:
        print(f'Epoch {i}, Loss: {losssum / len(X)}')

#%% check results
testpredictions = ((w * X) + b).detach().numpy().reshape(-1)

print(testpredictions)
print(X.detach().numpy().reshape(-1))

sns.lineplot(x=X.detach().numpy().reshape(-1), y=testpredictions, color='red')



sns.scatterplot(x='wt', y='mpg', data=cars)

# %%

# %%

# %%
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X=X, y=Y)
print(f'slope {reg.coef_}, bias {reg.intercept_}')

# %%

# %%

# %% (Statistical) Linear Regression


# %% create graph visualisation
# make sure GraphViz is installed (https://graphviz.org/download/)
# if not computer restarted, append directly to PATH variable
# import os
# from torchviz import make_dot
# os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'
# make_dot(loss_tensor)
# %%
