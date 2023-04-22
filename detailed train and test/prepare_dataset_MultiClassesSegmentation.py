# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:54:50 2023

@author: FaresBougourzi
"""


import os
import numpy as np
import cv2
import torch

import nibabel as nib

val_pathsave = "./Save MCCT-scans2/" 
if not os.path.exists(val_pathsave):
    os.makedirs(val_pathsave) 


idximg = -1

percentage = []
Images_names = []
subj = []
sub_percentage = []

database_path = './9 CT scans/rp_im'
database_pathh = './9 CT scans/'
Ct_scans = os.listdir(database_path)
sub = -1

Training_data = []
Training_mask = []
Training_inf = []

for Ct_scan in Ct_scans:
    
    data_name = 'rp_im'
    slice_samples = os.path.join(database_pathh, data_name, Ct_scan)

    mask_name = 'rp_lung_msk'
    mask_samples = os.path.join(database_pathh, mask_name, Ct_scan)
    
    inf_name = 'rp_msk'
    inf_samples = os.path.join(database_pathh, inf_name, Ct_scan)   
    
    slices = nib.load(slice_samples)
    masks = nib.load(mask_samples)
    infs = nib.load(inf_samples)    
    

    slices = slices.get_fdata()
    masks = masks.get_fdata()
    infs = infs.get_fdata()
    sub += 1

    sub_lung_area = 0
    sub_inf_area = 0
    for i in range(infs.shape[2]): 
        
        idximg += 1
       
        slice1 = slices[:,:,i]
        slice1 = cv2.rotate(slice1, cv2.ROTATE_90_CLOCKWISE)
        mask1 = masks[:,:,i]

        mask1 = np.uint8(cv2.rotate(mask1, cv2.ROTATE_90_CLOCKWISE))
        inf1 = infs[:,:,i]

        inf1 =  np.uint8(cv2.rotate(inf1, cv2.ROTATE_90_CLOCKWISE))
        
        img = cv2.normalize(src=slice1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        infected_area = np.sum(inf1 > 0)
        if infected_area >0:
            idximg += 1        
            Training_data.append(img)
            Training_mask.append(mask1)
            Training_inf.append(inf1)
        
################################################

data_path = './100 slices'
data_name = 'tr_im.nii.gz'
slice_samples = os.path.join(data_path, data_name)

mask_name = 'tr_lungmasks_updated.nii.gz'
mask_samples = os.path.join(data_path, mask_name)

inf_name = 'tr_mask.nii.gz'
inf_samples = os.path.join(data_path, inf_name)    

import nibabel as nib
slices = nib.load(slice_samples)
masks = nib.load(mask_samples)
infs = nib.load(inf_samples)


percentage = []
Images_names = []
slices = slices.get_fdata()
masks = masks.get_fdata()
infs = infs.get_fdata()

for i in range(50): 
    idximg += 1
    
    slice1 = slices[:,:,i]
    slice1 = cv2.rotate(slice1, cv2.ROTATE_90_CLOCKWISE)
    
    img = cv2.normalize(src=slice1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    
    mask1 = masks[:,:,i]
    mask1 = np.uint8(cv2.rotate(mask1, cv2.ROTATE_90_CLOCKWISE))
    inf1 = infs[:,:,i]
    inf1 = np.uint8(cv2.rotate(inf1, cv2.ROTATE_90_CLOCKWISE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    infected_area = np.sum(inf1 > 0)
    if infected_area >0:
        idximg += 1        
        Training_data.append(img)
        Training_mask.append(mask1)
        Training_inf.append(inf1)
    
######
          
X = [i for i in Training_data]
y = [i for i in Training_mask] 
y2 = [i for i in Training_inf]
training= (X, y, y2)
torch.save(training,'./Datas/Train_MC.pt') 


################################################
          
Training_data=[]
Training_mask=[]
Training_inf=[]

for i in range(50): 
    idximg += 1
    i += 50
    
    slice1 = slices[:,:,i]
    slice1 = cv2.rotate(slice1, cv2.ROTATE_90_CLOCKWISE)
    
    img = cv2.normalize(src=slice1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    
    mask1 = masks[:,:,i]
    mask1 = np.uint8(cv2.rotate(mask1, cv2.ROTATE_90_CLOCKWISE))
    inf1 = infs[:,:,i]
    inf1 = np.uint8(cv2.rotate(inf1, cv2.ROTATE_90_CLOCKWISE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if infected_area >0:
        idximg += 1        
        Training_data.append(img)
        Training_mask.append(mask1)
        Training_inf.append(inf1)
 
########         
X = [i for i in Training_data]
y = [i for i in Training_mask] 
y2 = [i for i in Training_inf]
training= (X, y, y2)
torch.save(training,'./Datas/Test_MC.pt')  
