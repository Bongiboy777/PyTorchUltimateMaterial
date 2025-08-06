#%% packages
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
os.getcwd()

# %% transform and load data
# TODO: set up image transforms
size = 64
transformer = transforms.Compose([
    transforms.Grayscale(),

    transforms.Resize((size,size)),
    transforms.CenterCrop(size),
    transforms.ToTensor()
])

#%%

# TODO: set up train and test datasets
testSet = torchvision.datasets.ImageFolder('./test', transformer)
trainSet = torchvision.datasets.ImageFolder('./train', transformer)
#%%
#  TODO: set up data loaders
testLoader = DataLoader(testSet, 4, True)
trainLoader = DataLoader(trainSet, 4, True)

dataIter = iter(trainLoader)
images, labels = dataIter._next_data()
plt.imshow(np.transpose(images[-1].numpy(), (1,2,0)))

# %%
CLASSES = ['affenpinscher', 'akita', 'corgi']
NUM_CLASSES = len(CLASSES)

# TODO: set up model class
class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self, image_size, lastConvOut: int) -> None:
        super().__init__()
        self.imageSize = image_size
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, lastConvOut, 3)
        self.conv1outSize = image_size - 2
        self.pool1size = self.conv1outSize / 2
        self.conv2outsize = self.pool1size - 2
        self.pool2size = int(np.floor(self.conv2outsize / 2))
        # print(self.pool2size)

        self.fc1 = nn.Linear(self.pool2size * self.pool2size * lastConvOut, 64)
        self.fc2 = nn.Linear(64,16)
        self.fc3 = nn.Linear(16,3)
        self.softMax = nn.Softmax()

    

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softMax(x)
        return x


# input = torch.rand(1, 1, 50, 50) # BS, C, H, W
model = ImageMulticlassClassificationNet(size, 16)      
# model(input).shape


# TODO: set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.002)
\
NUM_EPOCHS = 100
losses = []
for epoch in range(NUM_EPOCHS):
    loss_epoch = 0
    for i, data in enumerate(trainLoader, 0):
    
        inputs, labels = data
        # TODO: define training loop
        optimizer.zero_grad()
        # print(inputs)
        preds = model(inputs)
        loss = loss_fn(preds, labels)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    losses.append(loss_epoch)
   

    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {losses[-1]:.4f}')


# %% test
y_test = []
y_test_hat = []
for i, data in enumerate(testLoader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    

    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f} %')
# %% confusion matrix
confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
# %%


