#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:29:07 2022

@author: lucas
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def mean_std(loader):
    
  x,y,cell_class = loader[0]
  nChannels = x.shape[0]
    
  sum_inp, squared_sum_inp = np.zeros(nChannels),np.zeros(nChannels)
  sum_inp_Ch, squared_sum_inp_Ch= np.zeros(nChannels),np.zeros(nChannels)
  sum_tar, squared_sum_tar, num_batches = 0, 0, 0
  for x,y,cell_class in loader:
      for i_ch in range(nChannels):
          sum_inp_Ch[i_ch] += torch.mean(x[i_ch,:,:])
          #sum_inp_Ch2 += torch.mean(x[1,:,:])
          squared_sum_inp_Ch[i_ch] += torch.mean(x[i_ch,:,:]**2)
          #squared_sum_inp_Ch2 += torch.mean(x[1,:,:]**2)
      y = torch.squeeze(y)
      y = y.type(torch.FloatTensor)
      sum_tar += torch.mean(y[:,:])
      squared_sum_tar += torch.mean(y[:,:]**2)
      num_batches += 1

  sum_inp = sum_inp_Ch#[sum_inp_Ch1, sum_inp_Ch2]
  squared_sum_inp = squared_sum_inp_Ch#[squared_sum_inp_Ch1,squared_sum_inp_Ch2]
  mean_inp = np.divide(sum_inp,num_batches)
  std_inp = np.sqrt(np.subtract(np.divide(squared_sum_inp,num_batches), np.square(mean_inp)))
  mean_tar = np.divide(sum_tar,num_batches)
  std_tar = (squared_sum_tar/num_batches - mean_tar**2)**0.5
  return mean_inp, std_inp,mean_tar.numpy(),std_tar.numpy()

def mean_std_input(loader):
    
  x,y,cell_class = loader[0]
  nChannels = x.shape[0]
  
  # revert one hot encoding of y
  y = torch.argmax(y, dim=2)

    
  sum_inp, squared_sum_inp = np.zeros(nChannels),np.zeros(nChannels)
  sum_inp_Ch, squared_sum_inp_Ch= np.zeros(nChannels),np.zeros(nChannels)
  num_batches = 0
  for x,y,cell_class in loader:
      for i_ch in range(nChannels):
          sum_inp_Ch[i_ch] += torch.mean(x[i_ch,:,:])
          #sum_inp_Ch2 += torch.mean(x[1,:,:])
          squared_sum_inp_Ch[i_ch] += torch.mean(x[i_ch,:,:]**2)
          #squared_sum_inp_Ch2 += torch.mean(x[1,:,:]**2)

      num_batches += 1

  sum_inp = sum_inp_Ch#[sum_inp_Ch1, sum_inp_Ch2]
  squared_sum_inp = squared_sum_inp_Ch#[squared_sum_inp_Ch1,squared_sum_inp_Ch2]
  mean_inp = np.divide(sum_inp,num_batches)
  std_inp = np.sqrt(np.subtract(np.divide(squared_sum_inp,num_batches), np.square(mean_inp)))

  return mean_inp, std_inp

def min_max(loader):
    
  x,y,cell_class = loader[0]
  nChannels = x.shape[0]
    
  min_inp, max_inp = 1000*np.ones([nChannels,1]),np.zeros([nChannels,1])
  min_tar, max_tar = 1000, 0
  for x,y,cell_class in loader:
      for i_ch in range(nChannels):
          # new min
          if min_inp[i_ch] > torch.min(x[i_ch,:,:]).numpy():
              min_inp[i_ch] = torch.min(x[i_ch,:,:]).numpy()
          # new max
          if max_inp[i_ch] < torch.max(x[i_ch,:,:]).numpy():
              max_inp[i_ch] = torch.max(x[i_ch,:,:]).numpy()
          
      y = torch.squeeze(y)
      # new min
      if min_tar > torch.min(y).numpy():
          min_tar = torch.min(y).numpy()
      # new max
      if max_tar < torch.max(y).numpy():
          max_tar = torch.max(y).numpy()


  return min_inp, max_inp,min_tar, max_tar

def min_max_input(loader):
    
  x,y,cell_class = loader[0]
  nChannels = x.shape[0]
    
  min_inp, max_inp = 1000*np.ones([nChannels,1]),np.zeros([nChannels,1])

  for x,y,cell_class in loader:
      for i_ch in range(nChannels):
          # new min
          if min_inp[i_ch] > torch.min(x[i_ch,:,:]).numpy():
              min_inp[i_ch] = torch.min(x[i_ch,:,:]).numpy()
          # new max
          if max_inp[i_ch] < torch.max(x[i_ch,:,:]).numpy():
              max_inp[i_ch] = torch.max(x[i_ch,:,:]).numpy()

  return min_inp, max_inp