# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:34:41 2020

@author: Joachim
"""

from CustomDatasets import BlensorDataset
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
    
    # Import data
    trainingDataset = BlensorDataset('train')
    valDataset = BlensorDataset('val')
    testDataset = BlensorDataset('test')
    
    # Import model
    model = FCN_8s()
    model.to(device)
    
    # Prepare loader
    bs = 16 
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size = bs, shuffle=False, num_workers=12)
    
    # load weights
    model.load_state_dict(torch.load('./results/vgg16-big/best_model.pth'))
    
    # Assess model on test data
    mpa, iou, precision, recall = model.assess(testLoader, device)
    print('mpa       = '+str(mpa))
    print('iou       = '+str(iou))
    print('precision = '+str(precision))
    print('recall    = '+str(recall))
