# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:15:04 2020

@author: Joachim
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:34:41 2020

@author: Joachim
"""

from CustomModels import FCN_8s
import numpy as np
import torch
import time

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

    # Perform prediction of random tensor, measure the mean time it takes
    # for a forward pass.
    n_runs = 10000
    times = []
    for run in range(n_runs):
        x = torch.rand(1,3,224,192).to(device)
        start_time = time.time()
        pred = model.predict(x)
        times.append(time.time()-start_time)
    mean_ms = np.mean(times) * 1000
    print('Mean time required for a forward pass: '+str(mean_ms)+' ms')
