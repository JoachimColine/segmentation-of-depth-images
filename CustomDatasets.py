# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:43:35 2020

@author: Joachim
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
"""
Requirements:
    - all scans are located in the root directory provided in the constructor
    - they have the name '"scan_x.npy" where x is a positive integer
    - the first scan (x=0) is background only.
    - the first row of each file is depth data, the second is ground truth 
      segmentation.
"""

def scale(x):
    return (x - np.mean(x))/np.std(x)


class BlensorDataset(Dataset):
    """Custom dataset generated with BlenSor."""

    def __init__(self,  mode, rootdir='./dataset_100_000', width=224, height=171):
        """
        Args:
            rootdir: str, directory containing the scans.
            width, height: int, dimensions of the scans.
            mode: str, either 'train', 'val' or 'test' 
        """
        self.rootdir = rootdir 
        self.width = width
        self.height = height
        self.mode = mode
        # Determine the value assigned to background for segmentation.
        # The first scan is assumed to contain this value only.
        firstScan = np.load(os.path.join(self.rootdir, "scan_0.npy"))
        self.background = np.unique(firstScan[1,:])
        self.preprocess_input = transforms.Compose([
            transforms.Lambda(scale),
            transforms.ToTensor(),
            ])
        # Compute dataset size. It depends on self.mode.
        # The first 70% of scans in rootdir are train, next 20% are val 
        # and last 10% are test data.
        if self.mode == 'train':
            self.length = int(0.7*len(os.listdir(self.rootdir)))
        elif self.mode == 'val':
            self.length = int(0.2*len(os.listdir(self.rootdir)))
        elif self.mode == 'test':
            self.length = int(0.1*len(os.listdir(self.rootdir)))   
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Variable idx corresponds to scan number -> apply offset according to
        # self.mode for accessing different files:
        if self.mode == 'train':
            offset = 0
        elif self.mode == 'val':
            offset = int(0.7*len(os.listdir(self.rootdir)))
        elif self.mode == 'test':
            offset = int(0.9*len(os.listdir(self.rootdir)))
        idx = idx + offset
        filename = os.path.join(self.rootdir, "scan_"+str(idx)+".npy")
        scan = np.load(filename)
        depth_data =  scan[0,:]
        segmentation = scan[1,:]
        segmentation = np.array((segmentation != self.background) * 1)
        img = depth_data.reshape((self.width,self.height))
        msk = segmentation.reshape((self.width,self.height))        

        # Variable img is required to be repeated 3 times along third dimension 
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
        # Variable msk is required to be added a new third dimension
        msk = np.expand_dims(msk,axis=0)
        
        # apply preprocessing to img
        img = self.preprocess_input(img)

        # add zeros along dimension that is not multiple of 32
        padding1 = (32 - self.width % 32) % 32
        padding2 = (32 - self.height % 32) % 32
        img = np.pad(img, ((0,0),(0,padding1),(0,padding2)), 'constant')
        msk = np.pad(msk, ((0,0),(0,padding1),(0,padding2)), 'constant')
        sample = (torch.from_numpy(img).float(), torch.from_numpy(msk).float())
        return sample
