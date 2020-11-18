#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import json
from os import listdir
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
import skimage.io
import cv2
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
import torch.nn.functional as F
import random


# In[2]:


#define the input size and transform function
TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# In[5]:


class VGG16(nn.Module):
    
    
    def __init__(self):
        super(VGG16, self).__init__()
        
        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3) # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1)) # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 64 * 112 * 112
        
        self.conv2_1 = nn.Conv2d(64, 128, 3) # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1)) # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 128 * 56 * 56
        
        self.conv3_1 = nn.Conv2d(128, 256, 3) # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 256 * 28 * 28
        
        self.conv4_1 = nn.Conv2d(256, 512, 3) # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 14 * 14
        
        self.conv5_1 = nn.Conv2d(512, 512, 3) # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 7 * 7
        
        # view
        
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        # softmax 1 * 1 * 1000
        
    def forward_once(self, x):
        
        # x.size(0) is batch_size
        in_size = x.size(0)
        
        out = self.conv1_1(x) # 222
        out = F.relu(out)
        
        out = self.conv1_2(out) # 222
        out = F.relu(out)
        out = self.maxpool1(out) # 112
        
        out = self.conv2_1(out) # 110
        out = F.relu(out)
        out = self.conv2_2(out) # 110
        out = F.relu(out)
        out = self.maxpool2(out) # 56
        
        out = self.conv3_1(out) # 54
        out = F.relu(out)
        out = self.conv3_2(out) # 54
        out = F.relu(out)
        out = self.conv3_3(out) # 54
        out = F.relu(out)
        out = self.maxpool3(out) # 28
        
        out = self.conv4_1(out) # 26
        out = F.relu(out)
        out = self.conv4_2(out) # 26
        out = F.relu(out)
        out = self.conv4_3(out) # 26
        out = F.relu(out)
        out = self.maxpool4(out) # 14
        
        out = self.conv5_1(out) # 12
        out = F.relu(out)
        out = self.conv5_2(out) # 12
        out = F.relu(out)
        out = self.conv5_3(out) # 12
        out = F.relu(out)
        out = self.maxpool5(out) # 7
        
        # reshape as 2D
        out = out.view(in_size, -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        #out = F.log_softmax(out, dim=1)
        return out
                
    def forward(self, input1, input2=None, input3=None):
        if input2 is None and input3 is None:
            output1 = self.forward_once(input1)
            return output1
        else:
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            output3 = self.forward_once(input3)
            return output1, output2, output3


# In[4]:


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class SiameseNetwork(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(SiameseNetwork,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 9, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(2048,num_classes)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x
    
    def forward(self, input1, input2=None, input3=None):
        if input2 is None and input3 is None:
            output1 = self.forward_once(input1)
            return output1
        else:
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            output3 = self.forward_once(input3)
            return output1, output2, output3


# In[6]:


class Net(torch.nn.Module): #This is a linear claffification model used to classify each person
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x


# In[7]:


class ContrastiveLoss(torch.nn.Module):
    """
    Triplet loss function based on Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, output3, label):
        euclidean_distance = torch.mean(F.pairwise_distance(output1, output2, keepdim = True))
        euclidean_distance1 = torch.mean(F.pairwise_distance(output1, output3, keepdim = True))
        loss_contrastive = torch.mean((1-label) * (torch.pow(torch.clamp(self.margin + euclidean_distance1 - euclidean_distance, min=0.0), 2)) + (label) * (torch.pow(torch.clamp(self.margin + euclidean_distance - euclidean_distance1, min=0.0), 2)))
        return loss_contrastive


# In[8]:


class MyDataset(torch.utils.data.Dataset): #Create my dataset which inherits from torch.utils.data.Dataset
    def __init__(self,txt, transform=None, target_transform=None): 
        super(MyDataset,self).__init__()
        path=txt
        file_list=open(path,'r')
        imgs = []
        for line in file_list:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1], words[2], words[3])) 
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        an_name, anchor, po_name, postive = self.imgs[index] 
        an_path = '/home/molan/Desktop/lfw/'+an_name+'/'+an_name+'_'+str(anchor).zfill(4)+'.jpg'
        po_path = '/home/molan/Desktop/lfw/'+po_name+'/'+po_name+'_'+str(postive).zfill(4)+'.jpg'
        if an_name == po_name :
            label = 1
        else:
            label = 0
            
        file_lis=listdir('/home/molan/Desktop/lfw/')
        file_lis.remove(po_name)
        ne_name = random.choice(file_lis)
        ne_file=listdir('/home/molan/Desktop/lfw/'+ne_name+'/')
        ne_img = random.choice(ne_file)
        ne_path = '/home/molan/Desktop/lfw/'+ne_name+'/'+ne_img
        img_an = cv2.imread(an_path)
        img_an = cv2.resize(img_an,(224,224))
        img_po = cv2.imread(po_path)
        img_po = cv2.resize(img_po,(224,224))
        img_ne = cv2.imread(ne_path)
        img_ne = cv2.resize(img_ne,(224,224))
        
        if self.transform is not None:
            img = self.transform(img_an) #transform images as we defined before
            exp_po = self.transform(img_po)
            exp_ne = self.transform(img_ne)
        return img,exp_po,exp_ne,label,an_name
 
    def __len__(self): 
        return len(self.imgs)
    


# In[41]:


def extract_feature(net):
    net.eval()
    feature = []
    label_num = 0
    label = []
    label_name = {}
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        fp = open('/home/molan/Desktop/project/people.txt','r')
        for line in fp:
            line = line.rstrip()
            words = line.split()
            if len(words) != 2:
                continue
            name = words[0]
            for i in range(1,int(words[1])+1):
                path = '/home/molan/Desktop/lfw/'+name+'/'+name+'_'+str(i).zfill(4)+'.jpg'
                img = cv2.imread(path)
                img = cv2.resize(img,(224,224))
                img = transform(img)
                img = torch.unsqueeze(img, dim=0)
                img = img.cuda()
                output = net(img)
                feature.append(list(output[0].cpu()))
                label.append(label_num)
            label_name[name]=label_num
            label_num += 1
    return feature,label,label_name    


# In[10]:


#load training and testing data
train_data=MyDataset(txt='/home/molan/Desktop/project/train_new.txt', transform=transform)
test_data=MyDataset(txt='/home/molan/Desktop/project/test_new.txt', transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=16)


# In[35]:


net = VGG16().cuda()
#net = SiameseNetwork([3, 4, 6, 3]).cuda()
classification = Net(n_feature=2048, n_hidden=10, n_output=5749).cuda()

criterion =ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.003 )
optimizer_class = torch.optim.SGD(classification.parameters(), lr=0.02)
loss_class = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

loss_history = [] 
acc = []
iteration_number= 0


# In[26]:


for epoch in range(10):
    net.train()
    for i, data in enumerate(train_loader,0):
        img, img_o , img_p, label, an_name = data
        img, img_o , img_p, label = img.cuda(), img_o.cuda() , img_p.cuda(), label.cuda()
        optimizer.zero_grad()
        output1,output2, output3 = net(img,img_o,img_p)
        loss_contrastive = criterion(output1,output2,output3,label)
        loss_contrastive.backward()
        optimizer.step()
        print("train times: {}\nEpoch number {}\n Current loss {}\n".format(i,epoch,loss_contrastive.item()))
    else:
        feature,label,label_name = extract_feature(net)
        label_dict = dict(zip(label_name.values(), label_name.keys()))
        feature = torch.Tensor(feature).cuda()
        label = torch.Tensor(label).type(torch.LongTensor).cuda()
        feature, label = Variable(feature), Variable(label)
        for t in range(100):
            out = classification(feature)                 # input x and predict based on x
            loss = loss_class(out, label)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
            optimizer_class.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer_class.step()        # apply gradients 
            loss_history.append(loss.cpu().item())

        net.eval()
        accuracy = 0
        count = 0
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for i, data in enumerate(test_loader,0):
               img, img_o , img_p, label, an_name = data
               img = img.cuda()
               feature = net(img)
               out = classification(feature)
               prediction = torch.max(F.softmax(out), 1)[1].cpu()
               pred = prediction.data.numpy().squeeze()
               for j in range(len(an_name)):
                   count += 1
                   if pred[j] == label_name[an_name[j]] :
                       accuracy += 1
            acc.append(accuracy/count)
    torch.save(net,'/home/molan/Desktop/project/feature_extraction_model_triple.h5')
    torch.save(classification,'/home/molan/Desktop/project/classification_model_triple.h5')
plt.plot(loss_history)
plt.title("training loss")
plt.savefig("/home/molan/Desktop/project/loss_training.png")
plt.plot(acc)
plt.title("testing accuracy")
plt.savefig("/home/molan/Desktop/project/accuracy.png")


# In[ ]:




