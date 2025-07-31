#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

os.getcwd()
#%% transform, load data
Images = torchvision.datasets.Country211

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    
    # transforms.Normalize()

])

trainingSet = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
testSet = torchvision.datasets.ImageFolder(root='./data/test', transform=transform)
#%% 

# %% visualize images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
BATCHSIZE = 4
trainloader = DataLoader(trainingSet, batch_size=BATCHSIZE, shuffle=True)
dataiter = iter(trainloader)
images, labels = dataiter._next_data()
imshow(torchvision.utils.make_grid(images, nrow=2))
print(labels)
# %%
#  n * w * h * elements 
print(trainloader.dataset[999][0][0][0][31])
print(trainloader.dataset[0])
# %% Neural Network setup
class ImageClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # kernel size is size of filter, padding is size outside of input array, in channels, 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=0, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(6 * 6 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fc4 = nn.Linear(32, 1)

        pass
    
    def forward(self, x):
        # Conv1: input (batch_size, 1, 32, 32), kernel=3, stride=1, padding=0
        x = self.conv1(x) 
        # Output shape: (batch_size, 6, 32 - 3 + 1, 32 - 3 + 1) = (batch_size, 6, 30, 30)
        # print(f'After conv1: {x.shape}, expected: (batch_size, 6, {32 - 3 + 1}, {32 - 3 + 1})')
        
        x = self.relu(x)
        # ReLU does not change shape
        # print(f'After relu1: {x.shape}, expected: same as previous')
        
        x = self.pool(x)                                           
        # MaxPool2d: kernel=2, stride=2
        # Output shape: (prev_output - pool_kernel) // pool_stride + 1
        # (30 - 2) // 2 + 1 = 15
        # print(f'After pool1: {x.shape}, expected: (batch_size, 6, {(32 - 3 + 1)//2}, {(32 - 3 + 1)//2})')
        
        x = self.conv2(x)
        # Conv2: kernel=3, stride=1, padding=0
        # Output shape: (15 - 3 + 1) = 13
        # print(f'After conv2: {x.shape}, expected: (batch_size, 16, {((32 - 3 + 1)//2) - 3 + 1}, {((32 - 3 + 1)//2) - 3 + 1})')
        
        x = self.relu(x)
        # ReLU does not change shape
        # print(f'After relu2: {x.shape}, expected: same as previous')
        
        x = self.pool(x)
        # MaxPool2d: kernel=2, stride=2
        # Output shape: (13 - 2) // 2 + 1 = 6
        # print(f'After pool2: {x.shape}, expected: (batch_size, 16, {(((32 - 3 + 1)//2) - 3 + 1)//2}, {(((32 - 3 + 1)//2) - 3 + 1)//2})')
        
        x = torch.flatten(x, start_dim=1) # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
     
        # x = self.relu(x)
        # For binary classification, use sigmoid instead of softmax
        x = torch.sigmoid(x)
        return x
    



model = ImageClassificationNet()      
loss_fn = nn.MSELoss()
LR = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

NUM_EPOCHS = 1000
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # zero gradients
        optimizer.zero_grad()
        
        
        # forward pass
        predictions = model(data[0].float())
        
        # calc losses
        loss = loss_fn(predictions, data[1].reshape(-1,1).float())
        # print(loss)
        # print(predictions)
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{NUM_EPOCHS}, Step {i+1}/{len(trainloader)},'
                f'Loss: {loss.item():.4f}')
        print(f'predictions: {predictions}')
        print(f'values: {labels}')

        
            # print(predictions)
            # print(data[1])
            # print(data[1].reshape(-1,1))
 # %%

# %% test
testloader = DataLoader(testSet, batch_size=4, shuffle=True)
y_test = []
y_test_pred = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_pred.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, y_test_pred)
print(f'Accuracy: {acc*100:.2f} %')
# %%
# We know that data is balanced, so baseline classifier has accuracy of 50 %.

# %%
