#!/usr/bin/env python
# coding: utf-8

# ### PyTorch MNIST Example similar to TensorFlow Tutorial

# Refer to https://www.tensorflow.org/versions/master/tutorials/mnist/pros/

# PyTorch 0.1.6 and torchvision 0.1.7

# In[1]:


# get_ipython().run_line_magic('pylab', '')
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'watermark')
# get_ipython().run_line_magic('watermark', '-v -p numpy,torch,torchvision')


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# - MnistNet inherit from nn.Module class.
# - MnistNet must implement forward method.
# - layers instantiated in \__init__, but actual graph is created in forward method.
# - functional method like pooling, activation are in torch.nn.functional
# - explicitly specify padding and stride, otherwise default padding=0, stride=1 applied
# - Conv2d(in_channels, out_channels, kernel_size), weight/bias is uniform dist. with -1/sqrt(n_units)~+1/sqrt(n_units)
# - Linear(in_features, out_features), weight/bias is uniform dist. with -1/sqrt(n_units)~+1/sqrt(n_units)
# - model.conv1.weight, moedel.conv1.bias, .. etc
# - dropout default prob. p=0.5 applied when training=True

# In[3]:


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*7*7)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
model = MnistModel()
#model


# - dataset.MNIST load train-images when train=True
# - dataset.MNIST load t10k-images when train=False
# - pixel value divided by 255 with tranform.ToTensor()

# In[4]:


batch_size = 50
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)


# In[5]:


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=1000)


# - model.parameters return all weights and biases

# In[6]:


for p in model.parameters():
    print(p.size())


# In[7]:


optimizer = optim.Adam(model.parameters(), lr=0.0001)


# - model.train() makes training=True, model.eval() makes traingin=False

# In[8]:


model.train()
train_loss = []
train_accu = []
i = 0
for epoch in range(15):
    for data, target in train_loader:
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()    # calc gradients
        train_loss.append(loss.data)
        optimizer.step()   # update gradients
        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = prediction.eq(target.data).sum()/batch_size*100
        train_accu.append(accuracy)
        if i % 1000 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data, accuracy))
        i += 1


# In[9]:


plt.plot(np.arange(len(train_loss)), train_loss)


# In[10]:


plt.plot(np.arange(len(train_accu)), train_accu)


# - volatile=True is used in inference mode. it makes stopping histoy recording for backward().

# In[11]:


model.eval()
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('\nTest set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

