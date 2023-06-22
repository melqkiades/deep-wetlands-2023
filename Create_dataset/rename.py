import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import shutil

mask_directory = 'SAR_dataset/Annotations/data'

filelist = glob.glob(os.path.join(mask_directory, '*'))

for file in sorted(filelist):
    file_name = file.split('/')[3]
    file_parts = file_name.split('_')
    if file_parts[0] == 'Hornborgasjon':
        new_name = 'Hornborgarsjon_' + file_parts[1] + '_' + file_parts[2] + '_' + file_parts[3] + '_' + file_parts[4] + '_' + file_parts[5]

        shutil.move(file, mask_directory + '/' + new_name)



