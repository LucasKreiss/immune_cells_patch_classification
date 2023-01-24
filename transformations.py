#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:16:48 2022

@author: lucas
"""

from typing import List, Callable, Tuple

import numpy as np
import albumentations as A
#from sklearn.externals._pilutil import bytescale

from skimage.util import crop
from scipy import ndimage
import torchvision
#import torch


def random_crop(patch_size: int):
    torchvision.transforms.RandomCrop(size=(patch_size,patch_size), padding=None, pad_if_needed=False, fill=0, padding_mode='constant')

   # crops = [cropper(orig_img) for _ in range(4)]
    

def median_input(inp: np.ndarray, median_filter_size: int):
    """median filter of image channel separately"""
    inp_out=np.squeeze(inp)


    if (len(inp_out.shape)>2):
        nChannels = inp_out.shape[0]
        for i_ch in range(nChannels):
            inp_out[i_ch,:,:] = ndimage.median_filter(inp_out[i_ch,:,:],size=median_filter_size)
            
    else: 
        inp_out = ndimage.median_filter(inp,size=median_filter_size)
 
    return inp_out

def median_target(tar: np.ndarray, median_filter_size: int):
    """median filter of image channel separately"""
    tar_out=np.squeeze(tar)
    if (len(tar_out.shape)>2):
        nChannels = tar_out.shape[0]
        for i_ch in range(nChannels):
            tar_out[i_ch,:,:] = ndimage.median_filter(tar_out[i_ch,:,:],size=median_filter_size)
            
    else: 
        tar_out = ndimage.median_filter(tar,size=median_filter_size)
        
    return tar_out

def normalize_input_min_max(inp: np.ndarray):
    """Squash image input to the value range [0, 1] of image channel separately according to [min max] (no clipping)"""
    inp_out=np.squeeze(inp)
    
    if (len(inp_out.shape)>2):
        nChannels = inp_out.shape[0]
        for i_ch in range(nChannels):
            inp_out[i_ch,:,:] = (inp_out[i_ch,:,:] - np.min(inp_out[i_ch,:,:]))
            inp_out[i_ch,:,:] = np.divide(inp_out[i_ch,:,:], np.max(inp_out[i_ch,:,:]))
    else: 
        inp_out = (inp - np.min(inp))
        inp_out = inp_out / np.max(inp_out)
    
    return inp_out



def normalize_input_global_min_max(inp: np.ndarray, min_global: float, max_global: float):
    """Normalize of image channel separately based on low and high value."""

    inp_out=np.squeeze(inp) 
    
    if (len(inp_out.shape)>2):
        nChannels = inp_out.shape[0]
        #print(inp_out.shape)
        for i_ch in range(nChannels):
            inp_out[i_ch,:,:] = (inp_out[i_ch,:,:] - min_global[i_ch]) / (max_global[i_ch]-min_global[i_ch])
            
    else: 
        inp_out = (inp - min_global) / (max_global-min_global)

    return inp_out

def normalize_target_global_min_max(tar: np.ndarray, min_global: float, max_global: float):
    """Normalize of image channel separately based on mean and standard deviation."""
    
    tar_out=np.squeeze(tar)
    
    if (len(tar_out.shape)>2):
        nChannels = tar_out.shape[0]
        for i_ch in range(nChannels):
            tar_out[i_ch,:,:] = (tar_out[i_ch,:,:] - min_global[i_ch]) / (max_global[i_ch]-min_global[i_ch])
            
    else: 
        tar_out = (tar - min_global) / (max_global-min_global)

    return tar_out

def normalize_input_z_score(inp: np.ndarray, mean: float, std: float):
    """Normalize of image channel separately based on mean and standard deviation."""
    
    inp_out=np.squeeze(inp)
    
    if (len(inp_out.shape)>2):
        nChannels = inp_out.shape[0]
        for i_ch in range(nChannels):
            inp_out[i_ch,:,:] = (inp_out[i_ch,:,:] - mean[i_ch]) / std[i_ch]
            
    else: 
        inp_out = (inp - mean) / std

    return inp_out

def normalize_target_min_max(tar: np.ndarray):
    """Squash image input to the value range [0, 1] of image channel separately according to [min max] (no clipping)"""
    tar_out=np.squeeze(tar)
    
    if (len(tar_out.shape)>2):
        nChannels = tar_out.shape[0]
        for i_ch in range(nChannels):
            tar_out[i_ch,:,:] = (tar_out[i_ch,:,:]- np.min(tar_out[i_ch,:,:]))
            tar_out[i_ch,:,:] = tar_out[i_ch,:,:] / np.max(tar_out[i_ch,:,:])

    else: 
        tar_out = (tar - np.min(tar))
        tar_out = tar_out / np.max(tar_out)
    return tar_out


def normalize_target_z_score(tar: np.ndarray, mean: float, std: float):
    """Normalize of image channel separately based on mean and standard deviation."""
    tar_out=np.squeeze(tar)
    
    if (len(tar_out.shape)>2):
        nChannels = tar_out.shape[0]
        for i_ch in range(nChannels):
            tar_out[i_ch,:,:] = (tar_out[i_ch,:,:] - mean[i_ch]) / std[i_ch]
            
    else: 
        tar_out = (tar_out - mean) / std

    return tar_out

def normalize_input_quantization(inp: np.ndarray, bitdepth: int):
    """Normalize of image channel separately based on pixel bit depth"""
    
    inp_out=np.squeeze(inp)
    norm = (2**bitdepth)/20
    
    
    if (len(inp_out.shape)>2):
        nChannels = inp_out.shape[0]
        for i_ch in range(nChannels):
            inp_out[i_ch,:,:] = inp_out[i_ch,:,:] / norm
            
    else: 
        inp_out = inp_out  / norm

    return inp_out

def normalize_target_quantization(tar: np.ndarray, bitdepth: int):
    """Normalize of image channel separately based on pixel bit depth"""
    
    tar_out=np.squeeze(tar)
    norm = (2**bitdepth)/20
    
    if (len(tar_out.shape)>2):
        nChannels = tar_out.shape[0]
        for i_ch in range(nChannels):
            tar_out[i_ch,:,:] = tar_out[i_ch,:,:] / norm
            
    else: 
        tar_out = tar_out  / norm

    return tar_out


def create_dense_target(tar: np.ndarray):
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    for idx, value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx

    return dummy


def center_crop_to_size(x: np.ndarray,
                        size: Tuple,
                        copy: bool = False,
                        ) -> np.ndarray:
    """
    Center crops a given array x to the size passed in the function.
    Expects even spatial dimensions!
    """
    x_shape = np.array(x.shape)
    size = np.array(size)
    params_list = ((x_shape - size) / 2).astype(np.int).tolist()
    params_tuple = tuple([(i, i) for i in params_list])
    cropped_image = crop(x, crop_width=params_tuple, copy=copy)
    return cropped_image


def random_flip(inp: np.ndarray, tar: np.ndarray, ndim_spatial: int):
    flip_dims = [np.random.randint(low=0, high=2) for dim in range(ndim_spatial)]

    flip_dims_inp = tuple([i + 1 for i, element in enumerate(flip_dims) if element == 1])
    flip_dims_tar = tuple([i for i, element in enumerate(flip_dims) if element == 1])

    inp_flipped = np.flip(inp, axis=flip_dims_inp)
    tar_flipped = np.flip(tar, axis=flip_dims_tar)

    return inp_flipped, tar_flipped


class Repr:
    """Evaluable string representation of an object"""

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'


class FunctionWrapperSingle(Repr):
    """A function wrapper that returns a partial for input only."""

    def __init__(self, function: Callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray): return self.function(inp)


class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(self, function: Callable, input: bool = True, target: bool = False, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target
        #self.patch_size = patch_size

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        if self.input: inp = self.function(inp)
        if self.target: tar = self.function(tar)
        return inp, tar


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp: np.ndarray, target: np.ndarray):
        for t in self.transforms:
            inp, target = t(inp, target)
            
        return inp, target


class ComposeSingle(Compose):
    """Composes transforms for input only."""

    def __call__(self, inp: np.ndarray):
        for t in self.transforms:
            inp = t(inp)
        return inp

class AlbuSeg2d_Single(Repr):
    """
    Wrapper for albumentations' segmentation-compatible 2D augmentations.
    Wraps an augmentation so it can be used within the provided transform pipeline.
    See https://github.com/albu/albumentations for more information.
    Expected input: (C, spatial_dims)
    Expected target: (spatial_dims) -> No (C)hannel dimension
    """
    def __init__(self, albumentation: Callable):
        self.albumentation = albumentation

    def __call__(self, inp: np.ndarray,):
        # input, target
        out_dict = self.albumentation(image=inp)
        input_out = out_dict['image']
        

        return input_out
    
class AlbuSeg2d(Repr):
    """
    Wrapper for albumentations' segmentation-compatible 2D augmentations.
    Wraps an augmentation so it can be used within the provided transform pipeline.
    See https://github.com/albu/albumentations for more information.
    Expected input: (C, spatial_dims)
    Expected target: (spatial_dims) -> No (C)hannel dimension
    """
    def __init__(self, albumentation: Callable):
        self.albumentation = albumentation

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        # input, target
        out_dict = self.albumentation(image=inp, mask=tar)
        input_out = out_dict['image']
        target_out = out_dict['mask']

        return input_out, target_out


class AlbuSeg3d(Repr):
    """
    Wrapper for albumentations' segmentation-compatible 2D augmentations.
    Wraps an augmentation so it can be used within the provided transform pipeline.
    See https://github.com/albu/albumentations for more information.
    Expected input: (spatial_dims)  -> No (C)hannel dimension
    Expected target: (spatial_dims) -> No (C)hannel dimension
    Iterates over the slices of a input-target pair stack and performs the same albumentation function.
    """

    def __init__(self, albumentation: Callable):
        self.albumentation = A.ReplayCompose([albumentation])

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        # input, target
        tar = tar.astype(np.uint8)  # target has to be in uint8

        input_copy = np.copy(inp)
        target_copy = np.copy(tar)

        replay_dict = self.albumentation(image=inp[0])['replay']  # perform an albu on one slice and access the replay dict

        # TODO: consider cases with RGB 3D or multimodal 3D input

        # only if input_shape == target_shape
        for index, (input_slice, target_slice) in enumerate(zip(inp, tar)):
            result = A.ReplayCompose.replay(replay_dict, image=input_slice, mask=target_slice)
            input_copy[index] = result['image']
            target_copy[index] = result['mask']

        return input_copy, target_copy