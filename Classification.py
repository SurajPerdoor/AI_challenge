#!/usr/bin/env python
# coding: utf-8

# In[21]:


import torch
import numpy as np
from glob import glob


# In[22]:


files = np.array(glob("data/*/*/*"))
print('There are %d total images.' % len(files))


# In[23]:


import os
from torchvision import datasets

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as transforms


from torch.utils.data.sampler import SubsetRandomSampler


# In[24]:


num_workers = 0
batch_size = 32

transform_train = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.RandomRotation(20),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

transform_test = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

data_dir = 'data'

#loading images in train validate and test folders into dictionary loaders_scratch with 
# train, valid, test as keys

train_data = datasets.ImageFolder(os.path.join(data_dir, 'train/'), transform=transform_train)
test_data = datasets.ImageFolder(os.path.join(data_dir, 'test/'), transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

loaders_scratch = {'train': train_loader,
                   'test': test_loader}

#classes = ['apple', 'orange', 'banana', 'cardboard', 'bottle', 'can', 'mixed']


# In[25]:


import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        
        '''using stride 2 in place of maxpooling layers'''
        
        # in : 3*224*224
        self.conv1 = nn.Conv2d(3, 16, 4, stride = 2, padding = 1)
        
        # in :16* 112 *112
        self.conv2 = nn.Conv2d(16, 32, 4, stride = 2, padding = 1)
        
        # in : 32 * 56 * 56
        self.conv3 = nn.Conv2d(32, 64, 4, stride = 2, padding = 1)
        # out : 64*28*28
        
        self.fc1 = nn.Linear(64*7*7, 784)
        self.fc2 = nn.Linear(784, 7)
        
        self.dropout = nn.Dropout(p = 0.2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        ## Define forward behavior
        
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.batch_norm1(self.conv2(x)), 0.2)
        x = self.maxpool(x)
        x = F.leaky_relu(self.batch_norm2(self.conv3(x)), 0.2)
        x = self.maxpool(x)
        
        x = x.view(-1, 64*7*7)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # No relu here, we use CELoss
        x = self.fc2(x)
        
        
        
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()


# In[26]:


import torch.optim as optim

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr = 0.1)


# In[27]:


def train(n_epochs, loaders, model, optimizer, criterion, save_path):
    """returns trained model"""
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            optimizer.zero_grad()
            
            #typical train step:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            
        # print training statistics 
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        
        torch.save(model.state_dict(), save_path) 
    # return trained model
    return model


# In[ ]:


model_scratch = train(40, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, 'model_scratch.pt')


# In[ ]:


def test(loaders, model, criterion):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders_scratch, model_scratch, criterion_scratch)


# ## Trasfer Learning

# In[36]:


loaders_transfer = loaders_scratch


# In[37]:


import torchvision.models as models
import torch.nn as nn


model_transfer = models.resnet50(pretrained = True)


# In[38]:


model_transfer


# In[39]:


#freeze pretrained model parameters
for param in model_transfer.parameters():
    param.requires_grad = False
    
# resnet classifies 1000 categories, but we want only 133:

model_transfer.fc = nn.Linear(2048, 133)


# In[40]:


criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.Adam(model_transfer.fc.parameters(), lr = 0.01)


# In[41]:


# train the model
model_transfer = train(10, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))


# In[43]:


test(loaders_transfer, model_transfer, criterion_transfer)


# In[ ]:




