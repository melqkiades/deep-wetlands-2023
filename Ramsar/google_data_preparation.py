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
import utils
from PIL import Image
#from jaccard_similarity import calculate_intersection_over_union

class CustomDataset(Dataset):
    def __init__(self, image_dir, target_dir, train=False):
        self.image_dir = image_dir
        self.mask_dir = target_dir
        self.images = os.listdir(image_dir)
        #self.transforms = transforms.ToTensor()

    def __getitem__(self, index):

        #image = Image.open(self.image_paths[index])
        
        
        path = os.path.join(self.image_dir, self.images[index])
        image_vh_vv = rio.open(path).read()
        image = np.expand_dims(image_vh_vv[0], axis=0)
        #filtered_image = image[np.logical_not(np.isnan(image))]


        with rio.open(path) as src:
            dataset_array_vh_vv = src.read()
            dataset_array = dataset_array_vh_vv [0]
            minValue = np.nanpercentile(dataset_array, 1)
            maxValue = np.nanpercentile(dataset_array, 99)

        image[image > maxValue] = maxValue
        image[image < minValue] = minValue
        image = (image - minValue) / (maxValue - minValue)
        #print("SIZE image", torch.is_tensor(image))
        #print("SHAPE image", image.shape)
    
        path_mask = os.path.join(self.mask_dir, self.images[index].replace("SAR", "annotated_vh"))
        #mask = Image.open(self.target_paths[index])
        #path = os.path.join(ndwi_folder, filename)
        mask = np.array(rio.open(path_mask).read())
        #print("SHAPE MASK", mask.shape)
        #print()


        #t_image = self.transforms(image)
        return image, mask
        #return t_image, mask

    
    def __len__(self):
        return len(self.images)

def main():

    load_dotenv()
    
    data_dir = os.getenv('DATA_DIR') + '/'
    
    sar_dir_tr = os.getenv('SAR_ANN_TR')
    mask_dir_tr = os.getenv('MASK_ANN_TR')

    sar_dir_val = os.getenv('SAR_ANN_VAL')
    mask_dir_val = os.getenv('MASK_ANN_VAL')
  
  
    sar_folder_tr = sar_dir_tr # +'/'
    mask_folder_tr = mask_dir_tr # + '/'

    
    #folder_data = os.getenv('FOLDER_DATA')
    #folder_mask = os.getenv('FOLDER_MASK')

    image_paths_tr = [os.path.join(sar_dir_tr, filename) for filename in os.listdir(sar_dir_tr)]
    mask_paths_tr = [os.path.join(mask_dir_tr, filename) for filename in os.listdir(mask_dir_tr)]

    print("LEN images TR", len(image_paths_tr))
    print("LEN masks TR", len(mask_paths_tr))
    
    image_paths_val = [os.path.join(sar_dir_val, filename) for filename in os.listdir(sar_dir_val) if "Svartadalen" in filename]
    mask_paths_val = [os.path.join(mask_dir_val, filename) for filename in os.listdir(mask_dir_val) if "Svartadalen" in filename]

    print()
    print("LEN images VAL", len(image_paths_val))
    print("LEN masks VAL", len(mask_paths_val))


    #train_dataset = CustomDataset(image_paths_tr, mask_paths_tr, train=True)
    ###train_dataset = CustomDataset(sar_dir_tr, mask_dir_tr, train=True)
    ###train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)

    ###print()
    ###print("Len TRAIN dataset", len(train_dataset))

    #test_dataset = CustomDataset(image_paths_val, mask_paths_val, train=False)
    test_dataset = CustomDataset(sar_dir_val, mask_dir_val, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=1)
    
    print("Len TEST dataset", len(test_dataset))
    print()

    ###prova = os.listdir(sar_dir_tr)

    ###print("PROVA_1", prova[0])
    ###print("PROVA_2", sar_dir_tr)
    ###print("PROVA_3", os.path.join(sar_dir_tr, prova[0]))
    
    ###prova_2 = os.listdir(mask_dir_tr)
    ###print("PROVA_4", prova_2[0])
    ###print("PROVA_5", mask_dir_tr)
    ###print("PROVA_6", os.path.join(mask_dir_tr, prova_2[0].replace("SAR", "annotated_vh")))


    """
    prova_img, prova_msk = test_dataset[0]
    print("IMG size", prova_img.shape)
    print("MSK size", prova_msk.shape)

    max_img = np.max(prova_img)
    min_img = np.min(prova_img)

    print("IMG  MAX & MIN", max_img, min_img)
    print()

    max_msk = np.max(prova_msk)
    min_msk = np.min(prova_msk)

    print("MASK  MAX & MIN", max_msk, min_msk)
    #print("MASK size", mask.size)
    # print("MASK type",type(mask))
    # print()
    """

    images, masks = next(iter(test_loader))
    print("images shape", images.shape)
    print("masks shape", masks.shape)
    
    for elem in masks:
        print("IMG MAX & MIN", torch.max(elem), torch.min(elem))
   

main()

