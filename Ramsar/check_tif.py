import json
import os
import time
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv, dotenv_values
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio as rio

#from wetlands 
#import utils
#from wetlands.jaccard_similarity import calculate_intersection_over_union

#image_path = "/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/'Orebro lan_mosaic_2018-07-04_sar_VH.tif'"
#mask_path = "/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/'Orebro lan_mosaic_2018-07-04_ndwi_mask.tif'"

load_dotenv()
data_dir = os.getenv('DATA_DIR') + '/'
###sar_dir = os.getenv('SAR_DIR')
###ndwi_dir = os.getenv('NDWI_MASK_DIR')
sar_dir = os.getenv('SAR_ANN_VAL')
ndwi_dir = os.getenv('MASK_ANN_VAL')
###sar_dir = os.getenv('FOLDER_DATA')
###ndwi_dir= os.getenv('FOLDER_MASK')


###image_file = 'Orebro lan_mosaic_2018-07-04_sar_VH.tif'
###mask_file = 'Orebro lan_mosaic_2018-07-04_ndwi_mask.tif'

#image_path = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/"Orebro lan_mosaic_2018-07-04_sar_VH.tif"'
#mask_path = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/"Orebro lan_mosaic_2018-07-04_ndwi_mask.tif"'

sar_folder = sar_dir # +'/'
ndwi_folder = ndwi_dir # + '/'

"""
sar_folder = data_dir + '/'+ image_file
ndwi_folder = data_dir + '/' + mask_file

image = rio.open(sar_folder).read()
#filtered_image = image[np.logical_not(np.isnan(image))]

filtered_image = image.flatten()
        #filtered_image = image[np.logical_not(np.isnan(image))]

        #max_img = np.max(filtered_image)
        #min_img = np.min(filtered_image)

max_img = np.max(filtered_image)
min_img = np.min(filtered_image)

       # print("NUMBER", i)
print("IMG MAX & MIN", max_img, min_img)
print("image size", image.shape)
print("filtered image size", filtered_image.shape)

image = rio.open(ndwi_folder).read()
#filtered_image = image[np.logical_not(np.isnan(image))]

filtered_image = image.flatten()
        #filtered_image = image[np.logical_not(np.isnan(image))]

        #max_img = np.max(filtered_image)
        #min_img = np.min(filtered_image)

max_img = np.max(filtered_image)
min_img = np.min(filtered_image)

       # print("NUMBER", i)
print("IMG MAX & MIN", max_img, min_img)
print("image size", image.shape)
print("filtered image size", filtered_image.shape)
"""

for filename in os.listdir(sar_folder):
    for i in range(10):
        path = os.path.join(sar_folder, filename)
        ###image = rio.open(path).read()
        image_vh_vv = rio.open(path).read()
        #image = image_vh_vv[0]
        #image = torch.unsqueeze(image_vh_vv[0], dim=0)
        image = np.expand_dims(image_vh_vv[0], axis=0)
        filtered_image = image[np.logical_not(np.isnan(image))]
        
    
        with rio.open(path) as src:
            ###dataset_array = src.read()
            dataset_array_vh_vv = src.read()
            dataset_array = dataset_array_vh_vv [0]
            minValue = np.nanpercentile(dataset_array, 1)
            maxValue = np.nanpercentile(dataset_array, 99)
        
        filtered_image[filtered_image > maxValue] = maxValue
        filtered_image[filtered_image < minValue] = minValue
        filtered_image = (filtered_image - minValue) / (maxValue - minValue)
        
        ##cm = plt.get_cmap('gray')
        ##colored_image = cm(filtered_image[0])

        ##print("TEST", colored_image.shape)

        ##test_im= colored_image[:, :, :3] * 255


        
        #filtered_image = test_im.flatten()
        filtered_image = filtered_image.flatten()
        #filtered_image = image[np.logical_not(np.isnan(image))]
        
        #max_img = np.max(filtered_image)
        #min_img = np.min(filtered_image)

        max_img = np.max(filtered_image)
        min_img = np.min(filtered_image)

        print("NUMBER", i)
        print("IMG MAX & MIN", max_img, min_img)
        print("image size", image.shape)
        print("filtered image size", filtered_image.shape)

    break

print()
print()

for filename in os.listdir(ndwi_folder):
    for i in range(10):
        path = os.path.join(ndwi_folder, filename)
        image = rio.open(path).read()
        filtered_image = image[np.logical_not(np.isnan(image))].flatten()
        max_img = np.max(filtered_image)
        min_img = np.min(filtered_image)

        print("NUMBER", i)
        print("MASK  MAX & MIN", max_img, min_img)
        print("MASK size", image.shape)
        print("filtered MASK size", filtered_image.shape)
    break



"""
   with open(os.path.join(sar_folder, filename), 'r') as f: # open in readonly mode
      # do your stuff
      for i in range(5):
          image = rio.open(image_path).read().flatten()

"""



"""
image_path = images_dir + image_file
mask_path = images_dir + mask_file

image = rio.open(image_path).read().flatten()
mask = rio.open(mask_path).read().flatten()

filtered_image = image[np.logical_not(np.isnan(image))]
filtered_mask = image[np.logical_not(np.isnan(mask))]

max_img = np.max(filtered_image)
max_msk = np.max(filtered_mask)

min_img = np.min(filtered_image)
min_msk = np.min(filtered_mask)

print("IMG MAX & MIN", max_img, min_img)
print("MSK MAX & MIN", max_msk, min_msk)

print("image size", image.shape)
print("mask size", mask.shape)

print("filtered image size", filtered_image.shape)
print("filtered mask size", filtered_mask.shape)
"""

"""
image_tensor = torch.from_numpy(image.astype(np.float32))#.view(-1)
mask_tensor = torch.from_numpy(mask.astype(np.float32))#.view(-1)


shape_img = image_tensor.shape
image_tensor_reshaped = image_tensor.reshape(shape_img[0],-1)
#Drop all rows containing any nan:
image_tensor_reshaped = image_tensor_reshaped[~torch.all(image_tensor_reshaped.isnan(),dim=1)]

shape_msk = mask_tensor.shape
mask_tensor_reshaped = mask_tensor.reshape(shape_img[0],-1)
#Drop all rows containing any nan:
mask_tensor_reshaped = mask_tensor_reshaped[~torch.all(mask_tensor_reshaped.isnan(),dim=1)

#filtered_image = image_tensor[~torch.all(image_tensor.isnan(),dim=1)]#.view(-1)
#filrered_mask = mask_tensor[~torch.all(mask_tensor.isnan(),dim=1)]#.view(-1)

print("image size", image_tensor_reshaped.size())
print("mask size", mask_tensor_reshaped.size())

max_img = torch.max(image_tensor_reshaped)
max_msk = torch.max(mask_tensor_reshaped)

min_img = torch.min(image_tensor_reshaped)
min_msk = torch.min(mask_tensor_reshaped)

print("IMG MAX & MIN", max_img, min_img)
print("MSK MAX & MIN", max_msk, min_msk)"""
