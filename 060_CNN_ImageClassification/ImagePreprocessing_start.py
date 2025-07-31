#%%
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
# %% import image
img = Image.open('kiki.jpg')
img

# %% compose a series of steps
transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Grayscale(),
    transforms.RandomRotation(15),
    
    transforms.ToTensor()
])
tensorImg = transform(img)


# %%

# %%

# %%

# %%
# %%

# %%

# %%

# %%

#%%

# %%
