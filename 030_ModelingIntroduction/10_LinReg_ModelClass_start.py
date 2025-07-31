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

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% model class
class Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
    
 
    def forward(self,x):
        y = self.fc1(x)
    
        return y
model = Model(1, 1)
lr = 0.02
epochs = 1000
optim = torch.optim.SGD(model.parameters(), lr=0.02)
loss_fn = nn.MSELoss()
biases = []
weights = []
for epoch in range(epochs):
    for i, dat_x in enumerate(X):
        optim.zero_grad()
        pred = model(dat_x)
        loss = loss_fn(pred, y_true[i])
        loss.backward()
        optim.step()
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = param.data.detach().numpy()[0][0]
            weights.append(weight)
            print(f'{name.replace("fc1.","")}: {weight}')
        elif 'bias' in name:
            bias = param.data.detach().numpy()[0]
            biases.append(bias)
            print(f'{name.replace("fc1.","")}: {bias}')
            
#%%
print(biases)
print(weights)
print(list(range(epochs)))
#%%
sns.scatterplot(x=list(range(epochs)), y=weights)

#%%
sns.scatterplot(x=list(range(epochs)), y=biases)

# %%
sns.scatterplot(x=X_list, y=y_list)

sns.scatterplot(x=X_list, y=model(X).detach().numpy().reshape(-1) )
# %%

# %%

# %%

# %%

# %%
