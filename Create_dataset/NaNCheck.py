import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import shutil


input_directory = 'SAR_norm_unsplit/SAR/data'
mask_directory = 'Unsplit_optical/Annotations/data'
optical_directory = 'Unsplit_optical/Optical/data'
output_directory = 'moved_sar'
output_mask_directory = 'moved_annotations'
output_optical_directory = 'moved_optical'


filelist = glob.glob(os.path.join(input_directory, '*'))
vh_count = 0
vv_count = 0

for file in sorted(filelist):
    nan = False
    dataset = rasterio.open(file)
    file_name = file.split('/')[3]
    file_parts = file_name.split('_')
    file_mask_name = file_parts[0] + '_annotated_vh_' + file_parts[2] + '_' + file_parts[3] + '_' + file_parts[4]
    file_optical_name = file_parts[0] + '_optical_' + file_parts[2] + '_' + file_parts[3] + '_' + file_parts[4]

    vh = dataset.read(1)
    if np.isnan(vh).any():
        #print(file_name + ' contains NaN in VH')
        vh_count += 1
        nan = True

    vv = dataset.read(2)
    if np.isnan(vv).any():
        #print(file_name + ' contains NaN in VV')
        vv_count += 1
        nan = True

    if nan:
        shutil.move(file, output_directory + '/' + file_name)
        #shutil.move(mask_directory + '/' + file_mask_name, output_mask_directory + '/' + file_mask_name)
        #shutil.move(optical_directory + '/' + file_optical_name, output_optical_directory + '/' + file_optical_name)

print(str(vh_count) + ' images contain NaN in VH')
print(str(vv_count) + ' images contain NaN in VH')


