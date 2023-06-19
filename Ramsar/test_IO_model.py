import wandb
import os
import random
import json
import time
from dotenv import load_dotenv, dotenv_values
#from tqdm.notebook import tqdm
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image,ImageFile
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio as rio
import utils
import train_3 as tr
import wetlands_manually_annotated as wet

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchmetrics
import torch.optim as optim
import torchvision
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Dataset, DataLoader

def save_predictions_as_imgs(loader, model, device="cuda"):
    model.eval()
    for x, y in tqdm(loader, total=len(loader)):
        
        ###x, y = next(iter(loader))
        x = torch.unsqueeze(x[:,0,:,:],dim=1)
        x = x.float()
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = (preds > 0.5).float()

        #print("TEST", len(loader))

        #print('Type of preds', preds.size())
        #print('Type of y', y.unsqueeze(1).size())
        #print('Len preds', len(preds))

            f, axarr = plt.subplots(1,4)
            for id, img in enumerate(preds):
                axarr[id].imshow(img.permute(1, 2, 0).numpy(force=True))
            f.tight_layout(pad=0.5)
            print("sample prediction tensor", img)
            max_pred = torch.max(img)
            min_pred = torch.min(img)


            f2, axr = plt.subplots(1,4)
            for id, img in enumerate(y):
                axr[id].imshow(img.permute(1, 2, 0).numpy(force=True))
            f2.tight_layout(pad=0.5)
            print("sample mask tensor", img)
            max_mask = torch.max(img)
            min_mask = torch.min(img)

            f3, axo = plt.subplots(1,4)
            for id, img in enumerate(x):
                axo[id].imshow(img.permute(1, 2, 0).numpy(force=True))
            f3.tight_layout(pad=0.5)
            print("sample input image shape and  tensor", img.shape, img)
            max_in = torch.max(img)
            min_in = torch.min(img)

            print()
            print ("Max_&_Min_pred", max_pred, min_pred)
            print ("Max_&_Min_mask",max_mask, min_mask)
            print ("Max_&_Min_orig", max_in, min_in)
            print()
        
            image_in = wandb.Image(f2, caption="Ground_Truth")
            image_out = wandb.Image(f, caption="Predicted_mask")
            original = wandb.Image(f3, caption="Input_Image")

            wandb.log({"Predicted_Mask": image_out,"Ground Truth": image_in, "Input_Image": original})

    
    return f, f2, f3, max_pred, min_pred, max_mask, min_mask, max_in, min_in


#1- take 1 element from the NDWI dataset
#2- print it and save images
#3- evaluate it with the loaded model

wandb.login()

load_dotenv()

sar_dir_tr = os.getenv('SAR_ANN_TR')
mask_dir_tr = os.getenv('MASK_ANN_TR')
sar_dir_val = os.getenv('SAR_ANN_VAL')
mask_dir_val = os.getenv('MASK_ANN_VAL')
n_epochs = int(os.getenv('EPOCHS'))
learning_rate = float(os.getenv('LEARNING_RATE'))
seed = int(os.getenv('RANDOM_SEED'))
batch_size = int(os.getenv('BATCH_SIZE'))
num_workers = int(os.getenv('NUM_WORKERS'))
model_dir = os.getenv('MODELS_DIR')
region = os.getenv('REGION_ASCII_NAME')
date = os.getenv('START_DATE')
polarization = os.getenv('SAR_POLARIZATION')
orbit_pass = os.getenv('ORBIT_PASS')
patch_size = int(os.getenv('PATCH_SIZE'))
ndwi_input = os.getenv('NDWI_INPUT')
load_model = os.getenv('LOAD_MODEL')
model_file = os.getenv('MODEL_FILE')
tiles_data_file = os.getenv('TILES_FILE')
input_data = "tiles Ez"#"ndwi"#"tiles"
mask_test_Ez = os.getenv('MASK_TEST_Ez') + '/'
sar_test_Ez = os.getenv('SAR_TEST_Ez') + '/'


#print("TEST_TILES_FILE", tiles_data_file)
tiles_data = pd.read_csv(tiles_data_file)
#print()

config = {"input": input_data, "batch_size": batch_size}
wandb.init(project="debug", config=config)

utils.plant_random_seed(seed)
device = utils.get_device()

if load_model:
    model = wet.load_model_func(model_file, device)

###images_dir = os.getenv('SAR_DIR') + '/'
###masks_dir = os.getenv('NDWI_MASK_DIR') + '/'
tiles_data_file = os.getenv('TILES_FILE')

print()
print("START_TILES_FILE", tiles_data_file)

#NDWI data-set
tiles_data = pd.read_csv(tiles_data_file)
#dataloaders = tr.get_dataloaders(tiles_data, batch_size, num_workers, images_dir, masks_dir)
dataloaders = tr.get_dataloaders(tiles_data, batch_size, num_workers, sar_test_Ez, mask_test_Ez)

print()
print("LEN TRAIN", len(dataloaders['train']))
print("LEN TEST", len(dataloaders['test']))
print()

pred_img, in_img, orig, max_pred, min_pred, max_mask, min_mask, max_orig, min_orig = save_predictions_as_imgs(dataloaders["train"], model, device)

#Ramsar data-set
###val_dataset = wet.Ramsar(sar_dir_val, mask_dir_val)#, val_transform)
###val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=num_workers)
###pred_img, in_img, orig, max_pred, min_pred, max_mask, min_mask, max_orig, min_orig = save_predictions_as_imgs(val_loader, model, device)

#print()
#print ("Max_&_Min_pred", max_pred, min_pred)
#print ("Max_&_Min_mask",max_mask, min_mask)
#print ("Max_&_Min_orig", max_orig, min_orig)
#print()

###image_in = wandb.Image(in_img, caption="Ground_Truth")
###image_out = wandb.Image(pred_img, caption="Predicted_mask")
###original = wandb.Image(orig, caption="Input_Image")


#1- take 1 element from the ramsar dataset
#2- print it and save images
#3- evaluate it with the loaded model

###wandb.log({"Predicted_Mask": image_out,"Ground Truth": image_in, "Input_Image": original})#,
            # "Max_pred": max_pred, "Min_pred": min_pred, "Max_mask": max_mask, "Min_mask": min_mask, 
            # "Max_orig": max_orig, "Min_orig": min_orig})#,"Learning_Rate": lr[0],
              # "epoch": epoch})

wandb.finish()


