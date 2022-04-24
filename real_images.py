# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:47:20 2020

@author: Joachim
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from CustomModels import FCN_8s
import torch

if __name__ == "__main__":
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    
    print("-----------------------------------------------")
    print('Using CUDA:', use_cuda)
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
	    print(torch.cuda.get_device_name(0))
	    print('Memory Usage:')
	    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
	    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    print("-----------------------------------------------")    
    
    # Import model
    model = FCN_8s()
    model.to(device)
    model.load_state_dict(torch.load('./results/vgg16-big/best_model.pth'))
    
    # Read a .ply file
    input_file = "./real_data/raw/royale_20200513_203605_0.ply"
    pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
    
    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format. 
    data = np.asarray(pcd.points).reshape((171,224,3)) 
    depthmap = np.sqrt(np.power(data[:,:,0],2) + np.power(data[:,:,1],2) + np.power(data[:,:,2],2))
    nonzeros = depthmap[depthmap != 0]
    depthmap[depthmap != 0] = (depthmap[depthmap != 0] - np.mean(nonzeros)) / np.std(nonzeros)   
    # add zero padding
    depthmap = np.pad(depthmap, ((0,21),(0,0)), 'constant')
    plt.imshow(depthmap[:171,:])
    plt.axis('off')
    plt.show()
    
    x = torch.Tensor([[depthmap.transpose(), depthmap.transpose(), depthmap.transpose()]])
    threshold = 0.99999
    pred = model.predict(x.float().to(device)).to("cpu")
    pred_img = (pred[0][0,:,:171].numpy().transpose() > threshold) * 1.0
    plt.imshow(pred_img)
    plt.axis('off')
    plt.show()