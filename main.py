# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:41:44 2020

@author: Joachim
"""

from CustomDatasets import BlensorDataset
from CustomModels import FCN_8s
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_training_report(trainL, valL):
    fontsize = 12
    train_points = [i/(int(len(trainL)/len(valL))) for i in range(len(trainL))]
    # smoothen training signal
    window = 50
    trainL = np.convolve(np.ones(window)/window, trainL, mode='same')
    limit = window * 2 # removes boundary effects of the smoothening
    lw = 1.5 # linewidth
    plt.figure(figsize=(3,4))
    plt.plot(train_points[limit:-limit], trainL[limit:-limit],'darkseagreen', label="Training set", linewidth= lw)
    val_points = [i + 1 for i in range(len(valL))]
    plt.plot(val_points, valL, 'r', label="Validation set",linewidth=lw*1)
    plt.rc('font', size=fontsize) 
    plt.legend()
    plt.xlabel("Number of epochs", size=fontsize)
    plt.ylabel("Mean loss", size=fontsize)
    plt.show()
    
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
    
    # Prepare training
    bs = 16 
    trainLoader = torch.utils.data.DataLoader(trainingDataset, batch_size = bs, shuffle=True, num_workers=8)
    valLoader = torch.utils.data.DataLoader(valDataset, batch_size = bs, shuffle=False, num_workers=12)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size = bs, shuffle=False, num_workers=12)
    n_epochs = 100
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Train
    trainL, valL = model.fit(trainLoader, valLoader, n_epochs, criterion, optimizer, device)

    # Save variables (may be useful for later)
    np.save('trainLosses',trainL)
    np.save('valLosses',valL)    

    # Print prediction example
    index = 2 # pick a number
    model.print_example(testDataset, index, device)
    
    # Plot a report
    plot_training_report(trainL, valL)

    # Assess model on test data
    mpa, iou, precision, recall = model.assess(testLoader, device)
    print('--- Test ---')
    print('mpa       = '+str(mpa))
    print('iou       = '+str(iou))
    print('precision = '+str(precision))
    print('recall    = '+str(recall))
    
