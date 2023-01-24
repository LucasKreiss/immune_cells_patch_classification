#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:51:24 2022

@author: lucas
"""

# =============================================================================
# test code to use dataLoader
# =============================================================================
from customDataSet import SegmentationDataSet
from customDataSet import ClassificationDataSet
#from torch.utils import data
import torch
import albumentations
import transformations


#from skimage.transform import resize
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from datetime import date
#import albumentations
import torchvision as T
#import timm # for classification model (inception v3)
#from torchsummary import summary
import shap
from trainer_classify import Trainer_classify
import torch.nn as nn
import plot_training_curves 
import os
import pandas as pd 
import time
import Support as sup
# =============================================================================
# control parameters
# =============================================================================
st = time.time()
# size of median filter of images
median_filter_size = 8
# random seed
random_seed = 42
#  ratio of train to val data size 
train_size = 0.7  # 70:30 split
# batch size
batch_size = 64
# probabilty of flipped images as percentage
prob_flipped=0.8
# umber of down/up blocks in U-net
depth_unet = 4
# learning rate
learning_rate=0.0005
# number of epochs
Nepochs = 50
# size of cropped patches
patch_size = 64;
# index of image in total set to show as example
index_example_imag = 99 
# plot n number of val examples
n_val_examples = 5
# range of int values to display examples
range_hist=[-0.1, 2.1]
# specifiy if you want to use all GPUs
use_all_GPUs = True
# specify to save the trained model
save_model = False
# =============================================================================
# set up GPU
# =============================================================================

torch.cuda.empty_cache()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

use_gpu_num = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# device
if  use_all_GPUs:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda:"+use_gpu_num if torch.cuda.is_available() else "cpu")
   
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(device)

# =============================================================================
# load input and labels paired images and split to train and validation
# =============================================================================

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

root = pathlib.Path.cwd() 
dir_data = root.parent.absolute()/'Data'


# input and targets: choose from unpatched 1024x1024 (Input_AF), patches with 512x512 (Input_AF_8patches) or patches with 256x256 (Input_AF_16patches) 
#inputs = get_filenames_of_path(root / 'Data/Input_AF_16patches')
#targets = get_filenames_of_path(root / 'Data/Target_APC_16patches')

#inputs = get_filenames_of_path(dir_data / 'Input_AF_cropped_single_cells_64x64')
inputs = get_filenames_of_path(dir_data / 'Input_AF_cropped_single_cells_64x64_reduced')
targets = get_filenames_of_path(dir_data / 'Target_APC_cropped_single_cells_labeled_reduced')


# only use smaller subset of data for test
#inputs = inputs[25*16:35*16]
#targets = targets[25*16:35*16]
#inputs = ['Data/Input_AF/210517_CD4_M1_Neutrophils_M1_Pos1_.tif', 'Data/Input_AF/210517_CD4_M1_Neutrophils_M1_Pos2_.tif']
#targets = ['Data/Target_APC/210517_CD4_M1_Neutrophils_M1_Pos1_.tif', 'Data/Target_APC/210517_CD4_M1_Neutrophils_M1_Pos2_.tif']

inputs_train, inputs_valid = train_test_split(
    inputs,
    random_state = random_seed,
    train_size=train_size,
    shuffle = False)

targets_train, targets_valid = train_test_split(
    targets,
    random_state = random_seed,
    train_size=train_size,
    shuffle = False)

# =============================================================get_conv_layer================
# prepare folder for saving and documentation
# =============================================================================
today = date.today()
save_folder = root/today.strftime("%d-%m-%Y")

folder_name = input('Please provide a folder name for this experiment:')
folder_name='Classification_'+folder_name

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

os.makedirs(root/today.strftime("%d-%m-%Y")/folder_name)
save_folder = root/today.strftime("%d-%m-%Y")/folder_name



# =============================================================================
# carry out transformations
# =============================================================================

# get mean and std of all train data
loader = SegmentationDataSet(inputs=inputs_train,targets=targets_train)
loader_classify = ClassificationDataSet(inputs=inputs_train,targets=targets_train)
mean_all_inputs,std_all_inputs,mean_all_targets,std_all_targets = sup.mean_std(loader)

min_all_inputs,max_all_inputs,min_all_targets,max_all_targets = sup.min_max(loader)


print(mean_all_inputs,std_all_inputs,mean_all_targets,std_all_targets)

# =============================================================================
#  transformations
# =============================================================================

# The x,y should have a shape of [N, C, H, W]

transforms_classify_train = transformations.ComposeSingle([
  #transformations.AlbuSeg2d_Single(albumentations.HorizontalFlip(p=prob_flipped)),
  #transformations.AlbuSeg2d_Single(albumentations.VerticalFlip(p=prob_flipped)),
  #transformations.AlbuSeg2d(albumentations.augmentations.geometric.rotate.RandomRotate90(p=prob_rot)),
  transformations.FunctionWrapperSingle(transformations.median_input,median_filter_size=median_filter_size),
  #transformations.FunctionWrapperSingle(transformations.median_target,median_filter_size=median_filter_size),
  #transformations.FunctionWrapperDouble(transformations.normalize_input_quantization,input=True,target=False,bitdepth=16),
  #transformations.FunctionWrapperDouble(transformations.normalize_target_quantization,input=False,target=True,bitdepth=8),
  #transformations.FunctionWrapperDouble(transformations.normalize_input_z_score,input=True,target=False,mean=mean_all_inputs,std=std_all_inputs),
  #transformations.FunctionWrapperDouble(transformations.normalize_target_z_score,input=False,target=True,mean=mean_all_targets,std =std_all_targets)
  #transformations.FunctionWrapperDouble(transformations.random_crop, patch_size)
  #transformations.FunctionWrapperDouble(transformations.normalize_input_min_max,input=True,target=False),
  #transformations.FunctionWrapperDouble(transformations.normalize_target_min_max,input=False,target=True)
  transformations.FunctionWrapperSingle(transformations.normalize_input_global_min_max,min_global=min_all_inputs,max_global=max_all_inputs),
  #transformations.FunctionWrapperSingle(transformations.normalize_target_global_min_max,min_global=min_all_targets,max_global=mean_all_targets)
])

transforms_classify_val = transformations.ComposeSingle([
   # transformations.AlbuSeg2d(albumentations.HorizontalFlip(p=prob_flipped)),
   transformations.FunctionWrapperSingle(transformations.median_input,median_filter_size=median_filter_size),
   #transformations.FunctionWrapperSingle(transformations.median_target,median_filter_size=median_filter_size),
   #transformations.FunctionWrapperDouble(transformations.normalize_input_quantization,input=True,target=False,bitdepth=16),
   #transformations.FunctionWrapperDouble(transformations.normalize_target_quantization,input=False,target=True,bitdepth=8),
   #transformations.FunctionWrapperDouble(transformations.normalize_input_z_score,input=True,target=False,mean=mean_all_inputs,std=std_all_inputs),
   #transformations.FunctionWrapperDouble(transformations.normalize_target_z_score,input=False,target=True,mean=mean_all_targets,std =std_all_targets)
   #transformations.FunctionWrapperDouble(transformations.random_crop, patch_size)
   #transformations.FunctionWrapperDouble(transformations.normalize_input_min_max,input=True,target=False),
   #transformations.FunctionWrapperDouble(transformations.normalize_target_min_max,input=False,target=True)
   transformations.FunctionWrapperSingle(transformations.normalize_input_global_min_max,min_global=min_all_inputs,max_global=max_all_inputs),
   #transformations.FunctionWrapperSingle(transformations.normalize_target_global_min_max,min_global=min_all_targets,max_global=mean_all_targets)
])



# =============================================================================
#  data loaders
# =============================================================================

train_classify_data = ClassificationDataSet(inputs=inputs_train,
                                       targets=targets_train,
                                       transform=transforms_classify_train,
                                       patch_size = patch_size)

val_classify_data = ClassificationDataSet(inputs=inputs_valid,
                                       targets=targets_valid,
                                       transform=transforms_classify_val,
                                       patch_size = patch_size)

train_classify_dataloader = torch.utils.data.DataLoader(dataset=train_classify_data,
                                      batch_size=8,
                                      shuffle=True,
                                      num_workers=8,
                                      prefetch_factor=8) 

val_classify_dataloader = torch.utils.data.DataLoader(dataset=val_classify_data,
                                      batch_size=8,
                                      shuffle=True,
                                      num_workers=8,
                                      prefetch_factor=8) 


# =============================================================================
# show example data
# =============================================================================


batch = train_classify_data[index_example_imag]
x, y = batch  # 1 = t-cell, 0 = Neutrophil


nChannels_inp = x.shape[0]

# =============================================================================
# full classification block
# =============================================================================

print('RUNNING PATCH CLASSIFICATION')
#model_classification = timm.create_model('inception_v4', pretrained=False)
#model_classification = binaryResNet()
model_classification =  T.models.resnet18(weights='ResNet18_Weights.DEFAULT') # 'ResNet18_Weights.DEFAULT'

# create new conv layer with info about original  1st conv layer, only with different input channels
new_conv_layer1 = nn.Conv2d(in_channels = nChannels_inp,
                   out_channels = model_classification.conv1.out_channels,
                   kernel_size = model_classification.conv1.kernel_size,
                   stride = model_classification.conv1.stride, 
                   padding = model_classification.conv1.padding, 
                   bias = model_classification.conv1.bias)

# exchange 1st conv layer in resnet
model_classification.conv1 = new_conv_layer1

# add dense layer to final layer for binary output
lin = model_classification.fc
new_lin = nn.Sequential(
    #nn.Linear(lin.in_features, lin.in_features),
    #nn.ReLU(),
    lin,
    nn.ReLU(),
    nn.Linear(lin.out_features,1), #512 for restnet18
    #nn.Softmax()
)
# exhange final layer in model
model_classification.fc = new_lin
print(model_classification)

# activate parallel use of all GPUs
if  use_all_GPUs:
    model_classification= nn.DataParallel(model_classification,device_ids=[0,1,3])
    
# send model to available device:
model_classification.to(device)
model_classification.float()


# define loss function

l1_loss = torch.nn.L1Loss()
MSE_loss = torch.nn.MSELoss()
crossEntropy_loss = torch.nn.CrossEntropyLoss()
loss_metric = 'MAE'
# is optimization criterion pos or neg (e.g., ACC vs Loss)
criterion_pos = False;


# optimizer
optimizer = torch.optim.Adam(model_classification.parameters(), lr=learning_rate)

# trainer
trainer_classify = Trainer_classify(model=model_classification,
                  device=device,
                  criterion=l1_loss,
                  criterion_pos=criterion_pos,
                  optimizer=optimizer,
                  training_DataLoader=train_classify_dataloader,
                  validation_DataLoader=val_classify_dataloader,
                  lr_scheduler=None,
                  epochs=Nepochs,
                  epoch=0,
                  notebook=True)

st_train = time.time()
# training
(training_losses_mean,
    training_losses_std, 
    training_losses_min, 
    training_losses_max, 
    validation_losses_mean,
    validation_losses_std, 
    validation_losses_min, 
    validation_losses_max,
    lr_rates) = trainer_classify.run_trainer()
    
et_train = time.time()

# training curves min max area
plot_training_curves.plot_loss_curves_min_max(np.array(training_losses_mean),np.array(training_losses_min),np.array(training_losses_max), np.array(validation_losses_mean),np.array(validation_losses_min),np.array(validation_losses_max),loss_metric+'std_errorbar',save_folder = save_folder)

# training curves std errorbar
plot_training_curves.plot_loss_curves_std(np.array(training_losses_mean),np.array(training_losses_std), np.array(validation_losses_mean),np.array(validation_losses_std),loss_metric,save_folder = save_folder)


# training curves log loss
plot_training_curves.plot_loss_curves_std(np.log(np.array(training_losses_mean)),np.zeros(len(training_losses_std)), np.log(np.array(validation_losses_mean)),np.zeros(len(validation_losses_std)),loss_metric+'_log',save_folder = save_folder)

# plot val example images
X1 = np.random.randint(low=0, high=len(val_classify_dataloader)-1, size=(n_val_examples,))

# overall training acc:
correct = 0.0
for i, (x,label) in enumerate(train_classify_dataloader):
    #x = x[None,:]
    y_pred_prob = model_classification(x.cuda())
    y_pred_prob = torch.squeeze(y_pred_prob).cpu().detach().numpy()
    label  = torch.squeeze(label).cpu().detach().numpy()
    y_pred = (y_pred_prob>0.5).astype(float)
    correct += (y_pred == label).astype(float).sum()
    
    if i in X1:
        plot_training_curves.plot_example_classification(x,label,y_pred_prob,save_folder,i,range_hist,'train')
    
train_acc = 100*correct/len(train_classify_data)
print('train acc = '+str(train_acc)+'%')


# overall validation acc:
correct = 0.0
for i, (x,label) in enumerate(val_classify_dataloader):
    #x = x[None,:]
    y_pred_prob = model_classification(x.cuda())
    y_pred_prob = torch.squeeze(y_pred_prob).cpu().detach().numpy()
    label  = torch.squeeze(label).cpu().detach().numpy()
    y_pred = (y_pred_prob>0.5).astype(float)
    correct += (y_pred == label).astype(float).sum()
    
    if i in X1:
        plot_training_curves.plot_example_classification(x,label,y_pred_prob,save_folder,i,range_hist,'val')
    
validation_acc = 100*correct/len(val_classify_data)
print('validation acc = '+str(validation_acc)+'%')

# save model
if save_model:
    torch.save(model_classification.state_dict(),save_folder/"Trained_Classification_Model")

# save losses
losses_pd = pd.DataFrame(np.transpose([np.array(training_losses_mean),np.array(training_losses_std),np.array(training_losses_min),np.array(training_losses_max), np.array(validation_losses_mean),np.array(validation_losses_std),np.array(validation_losses_min),np.array(validation_losses_max)]),columns=['train loss mean','train loss std','train loss min','train loss max','val loss mean','val loss std','val loss min','val loss max'])
losses_pd.to_csv(save_folder/"Losses_classification.csv",sep='\t')
#np.savetxt("Losses.csv", [np.array(training_losses_mean),np.array(training_losses_std), np.array(validation_losses_mean),np.array(validation_losses_std)], delimiter=",")

et = time.time()
elapsed_time = et - st
print('total execution time: ',elapsed_time)
elapsed_time_train = et_train - st_train
print('training execution time: ',elapsed_time_train)


# save specifications
file = open(save_folder/"Specs_Classification.txt", "w")
file.write("Mean train ACC = "+str(np.mean(train_acc)) + "\n"+
           "Mean val ACC = "+str(np.mean(va_acc)) + "\n"+
           "Model = " + str(model_classification) + "\n" +
           "Training loss metric = " + str(trainer_classify.criterion) + "\n" +
           "Optimizer = "+str(trainer_classify.optimizer) + "\n"+
           "LR scheduler: "+str(trainer_classify.lr_scheduler)+"\n"+
           "Learning rate (starting lr, if scheduler isused)= "+str(learning_rate) + "\n"+
           "Median filter = "+str(median_filter_size)+ "\n"+
           "Train-Val-Split: "+str(train_size)+"-"+str(round(1-train_size,2))+ "\n"+
           "batch size = "+str(batch_size)+ "\n"+
           "Prob for flipping = "+ str(prob_flipped)+"\n"+
           "Device = "+str(device)+ "\n"+
           "Number of epochs = "+str(Nepochs)+ "\n"+
           "Image size = "+str(x.shape[2])+"x"+str(x.shape[2])+ "\n"+
           "Images in training: "+str(len(inputs_train))+ "\n"+
           "Images in validation: "+str(len(inputs_valid))+ "\n"+
           "Depth of Unet: "+str(depth_unet)+ "\n"+
           "Time for of training: "+str(elapsed_time_train)+ "\n"+
           "total execution time: "+str(elapsed_time)+ "\n"+
           "transforms: "+str(transforms_classify_train))
file.close()
    
# save model
torch.save(model_classification.state_dict(),save_folder/"Trained_Classification_Model")

# save losses
losses_pd = pd.DataFrame(np.transpose([np.array(training_losses_mean),np.array(training_losses_std),np.array(training_losses_min),np.array(training_losses_max), np.array(validation_losses_mean),np.array(validation_losses_std),np.array(validation_losses_min),np.array(validation_losses_max)]),columns=['train loss mean','train loss std','train loss min','train loss max','val loss mean','val loss std','val loss min','val loss max'])
losses_pd.to_csv(save_folder/"Losses_classification.csv",sep='\t')
#np.savetxt("Losses.csv", [np.array(training_losses_mean),np.array(training_losses_std), np.array(validation_losses_mean),np.array(validation_losses_std)], delimiter=",")


stop

# =============================================================================
# DeepShap explainer
# =============================================================================
# select a set of background examples to take an expectation over
background = train_data[np.random.choice(train_data, 100, replace=False)]

# explain predictions of the model on four images
e = shap.DeepExplainer(model, background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(val_data[1:5])

# plot the feature attributions
shap.image_plot(shap_values, -val_data[1:5])