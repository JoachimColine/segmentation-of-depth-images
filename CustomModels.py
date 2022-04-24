# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:58:15 2020

@author: Joachim
"""

import torch
import torch.nn as nn
from torchvision import models
from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time

class FCN_8s(nn.Module):
    
    def __init__(self):
        super(FCN_8s, self).__init__()
        # Encoder part is loaded from torchvision:
        self.encoder = models.vgg16(pretrained=True).features
		#self.encoder = models.vgg11(pretrained=True).features
		#self.encoder = models.vgg19(pretrained=True).features
        # Feature extraction is fixed, meaning gradient computation is not required:
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Decoder operations:
        # - ReLU:
        self.relu = nn.ReLU(inplace=True)
        # - Deconvolutions:
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # - Batch normalizations:
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        # - Final layer
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)
        # - Sigmoid used for prediction
        self.sigmoid = torch.sigmoid
            
    def forward(self, x):
        # ENCODER PART
        # Forward pass through the layers of the encoder.
        # Feature maps are stored after each MaxPool so they can be used 
        # in the decoder.
        features = []
        for _, layer in self.encoder._modules.items():
            x = layer(x)
            if str(layer)[:7] == 'MaxPool' and len(features) < 4:
                features.append(x)
        # DECODER PART
        # Forward pass through operations declared in the constructor
        # Feature maps from encoder are also used
        x = self.deconv1(x)
        x = self.relu(x)
        x = x + features[-1]
        x = self.bn1(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = x + features[-2]
        x = self.bn2(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = x + features[-3]
        x = self.bn3(x)
        x = self.deconv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.deconv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y = self.sigmoid(self.forward(x))
        return y
    
    def fit(self, trainLoader, valLoader, n_epochs, criterion, optimizer, device):
        train_batch_losses = []
        val_epoch_losses = []
        print("Training started.\n")
        start_time = time.time()
        for epoch in range(n_epochs):
            # TRAINING
            self.train()
            progress = 1
            for (inputs, masks) in trainLoader:
                stdout.write("\r"+str(progress)+" out of "+str(len(trainLoader))+" training batches")
                stdout.flush()
                progress += 1
                optimizer.zero_grad()
                predictions = self.forward(inputs.to(device)).to("cpu")
                loss = criterion(predictions[:,:,:,:171], masks[:,:,:,:171])
                loss.backward()
                optimizer.step()
                train_batch_losses.append(loss.item())
                del loss
            cumul_loss = 0
            stdout.write("\n")
            # VALIDATION
            self.eval()
            progress = 1
            with torch.no_grad():
                for (inputs, masks) in valLoader:
                    stdout.write("\r"+str(progress)+" out of "+str(len(valLoader))+" val. batches")
                    stdout.flush()
                    progress += 1
                    predictions = self.forward(inputs.to(device)).to("cpu")
                    loss = criterion(predictions[:,:,:,:171], masks[:,:,:,:171])
                    cumul_loss += loss.item()
                    del loss
            val_epoch_losses.append(cumul_loss / len(valLoader))
            stdout.write("\n")
            # If validation loss is minimal, then the current model is stored:
            if val_epoch_losses[-1] <= min(val_epoch_losses):
                torch.save(self.state_dict(), './best_model.pth')
                print("Model saved.")
            print("Epoch "+str(epoch+1)+"/"+str(n_epochs)+" | Train loss: "+
                  str(np.mean(train_batch_losses[-len(trainLoader):-1]))+ " | Val. loss: "+str(val_epoch_losses[-1])+".\n")
            print("Time ellapsed: " + str(time.time()-start_time) + " seconds.\n")
        # The best model is finally reloaded
        self.load_state_dict(torch.load('./best_model.pth'))
        return train_batch_losses, val_epoch_losses
    
    def assess(self, dataLoader, device, threshold=0.5):
        print("Assessing model...")
        mean_pixel_accuracies = []
        ious = []
        precisions = []
        recalls = []
        for (inputs, masks) in dataLoader:
            with torch.no_grad():
                predictions = self.predict(inputs.to(device)).to("cpu") > threshold 
            masks = masks[:,:,:,:171].squeeze(1)
            masks = masks == 1
            predictions = predictions[:,:,:,:171].squeeze(1)
            # IoU            
            i = (masks & predictions).float().sum((1,2))
            u = (masks | predictions).float().sum((1,2))
            iou = (i + 1e-6)/(u+ 1e-6) # 1e-6 is to avoid dividing by zero
            iou = iou.mean()
            # Precision
            predicted_p = predictions.float().sum((1,2))
            precision =  (i + 1e-6)/(predicted_p + 1e-6) # 1e-6 is to avoid dividing by zero
            precision = precision.mean()
            # Recall
            all_p = masks.float().sum((1,2))
            recall =  (i + 1e-6)/(all_p + 1e-6) # 1e-6 is to avoid dividing by zero
            recall = recall.mean()
            # Append results
            mean_pixel_accuracies.append((masks == predictions).float().mean())
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            
        return np.mean(mean_pixel_accuracies), np.mean(ious), np.mean(precisions), np.mean(recalls)
        
    def print_example(self, dataset, index, device, threshold=0.5):
        (x, y) = dataset[index]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        # raw input:
        x_input = x[0][0,:,:171].numpy().transpose()
        plt.imshow(x_input)
        plt.axis('off')
        plt.show()
        # prediction:
        pred = self.predict(x.to(device)).to("cpu")
        pred_img = (pred[0][0,:,:171].numpy().transpose() > threshold) * 1.0
        plt.imshow(pred_img)
        plt.axis('off')
        plt.show()
        # truth: 
        truth_img = (y[0][0,:,:171].numpy().transpose()) * 1.0
        plt.imshow(truth_img)
        plt.axis('off')
        plt.show()
        # false positives
        FP_img = (np.logical_and(pred_img, np.logical_not(truth_img))) * 1.0
        FP_img[FP_img == 0] = np.nan
        # false negatives
        FN_img = (np.logical_and(np.logical_not(pred_img), truth_img)) * 1.0
        FN_img[FN_img == 0] = np.nan
        # illustrate both on one image:
        plt.imshow(truth_img, 'gray', alpha=0.2)
        cmap1 = colors.ListedColormap(['red'])
        bounds=[0,2]
        norm = colors.BoundaryNorm(bounds, cmap1.N)
        plt.imshow(FP_img, cmap=cmap1, norm=norm)
        cmap2 = colors.ListedColormap(['blue'])
        bounds=[0,2]
        norm = colors.BoundaryNorm(bounds, cmap2.N)
        plt.imshow(FN_img, cmap=cmap2, norm=norm)
        plt.axis('off')
        plt.show()