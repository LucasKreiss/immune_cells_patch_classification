#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:42:39 2022

@author: lucas

loads x as 2x64x64 in 16.bit images and y as 1x64x64 in 16 bit
"""

import torch
import tifffile as tiff
import numpy as np
import torchvision.transforms as T
from PIL import Image

class ClassificationDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 patch_size=int
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float
        self.targets_dtype = torch.float
        self.path_size = patch_size


    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target images
        x = tiff.imread(input_ID)
              
        
        
        # convert to float
        x = np.float32(x)

        # Preprocessing
        if self.transform is not None:
            x = self.transform(x)

        # Typecasting from numpy to torch tensor
        x = torch.from_numpy(x).type(self.inputs_dtype)
        
        #x = T.Resize(size=(299,299))(x)
        
        # add zeros as 3rd channel since inception expects RBG
        #print(x.shape)
        #x = torch.cat((x,torch.zeros([1,x.shape[1],x.shape[2]])),0)
        #print(x.shape)
        
        # cast from floattensor to longtensor
        #y = y.type(torch.LongTensor)
        
        # get class of cell
        class_idx = str(target_ID).find('class')
        class_str = str(target_ID)[class_idx+5]
        
        label = np.float32(class_str)
        #print(class_str)
        #print(label)
        # 1 = t-cell, 0 = Neutrophil
        #print(type(x))
        
        #x, y = torch.x.type(self.inputs_dtype), torch.y.type(self.targets_dtype)
        return x, label
    

class SegmentationDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 patch_size=int
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32 # change to int64if using crossentropy
        self.path_size = patch_size


    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target images
        x, y = tiff.imread(input_ID), tiff.imread(target_ID)
                
        
        # convert to int16
        x, y = np.float32(x), np.float32(y)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting from numpy to torch tensor
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        
        # cast from floattensor to longtensor
        #y = y.type(torch.LongTensor)
        
        # get class of cell, # 1 = t-cell, 2 = Neutrophil
        class_idx = str(target_ID).find('class')
        cell_class = str(target_ID)[class_idx+5]
        


        #x, y = torch.x.type(self.inputs_dtype), torch.y.type(self.targets_dtype)
        return x, y, cell_class

    
class SegmentationDataSet_sanityCheck(torch.utils.data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 patch_size=int
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32 # change to int64if using crossentropy
        self.path_size = patch_size


    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target images
        x, y = tiff.imread(input_ID), tiff.imread(target_ID)
                
        
        # convert to int16
        x, y = np.float32(x), np.float32(y)
        
        x_ = np.expand_dims(x[0,:,:],0)
        y = x[1,:,:]
        x = x_

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting from numpy to torch tensor
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        
        # cast from floattensor to longtensor
        #y = y.type(torch.LongTensor)
        
        # get class of cell, # 1 = t-cell, 2 = Neutrophil
        class_idx = str(target_ID).find('class')
        cell_class = str(target_ID)[class_idx+5]

                
        # sanity check to predict ch2 with ch1
        return x, y, cell_class