#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 04:35:53 2022

@author: lucas
"""
import torchvision as T
import torch
from torch import nn
import types


class binaryResNet(nn.Module):
    def __init__(self):
        super(binaryResNet, self).__init__()
        self.model =  T.models.resnet18(weights=None)
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(512 , 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256 , 128),
            nn.Linear(128 , 1)
            #nn.Sigmoid()
        )
        self.model = modify_resnets(self.model)
        
def modify_resnets(model):
    # Modify attributs
    model.last_linear = model.fc
    model.fc = None
    
    def features(self,input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def logits(self,features):
        x = self.avgpool(features)
        x = x.view(x.size(0),-1)
        x = self.last_linear(x)
        return x
    
    def forward(self, x):
        batch_size ,_,_,_ = x.shape #taking out batch_size from input image
        x = self.model.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x,1).reshape(batch_size,-1) # then reshaping the batch_size
        x = self.classifier_layer(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model
