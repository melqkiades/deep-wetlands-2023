import json
import sys
import os
import time
from PIL import Image
import numpy as np
import rasterio as rio
import torch
import wandb
from dotenv import load_dotenv, dotenv_values
from matplotlib import pyplot as plt
from rasterio.windows import Window
import geopandas as gpd
from osgeo import gdal
from osgeo import ogr
import imageio as iio

import estimate_water as est
import train_models, utils, viz_utils

import unet as un
import archs as arc
import SegNet as seg

def visualize_predicted_image(image, model, device):

    patch_size = int(os.getenv('PATCH_SIZE'))
    width = image.shape[0] - image.shape[0] % patch_size
    height = image.shape[1] - image.shape[1] % patch_size
    pred_mask = predict_water_mask(image, model, device)
    fig, ax = plt.subplots(figsize=(10, 10))
    pred_mask = 1 - pred_mask

    plt.imshow(pred_mask)
    plt.show()
    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(image[:width, :height], cmap='gray')
    plt.show()
    plt.clf()

    return pred_mask


def predict_water_mask(sar_image, model, device):
    pred = est.otsu_threshold(sar_image)
    return pred


def generate_raster(image, src_tif, dest_file, step_size):
    with rio.open(src_tif) as src:
        # Create a Window and calculate the transform from the source dataset
        width = src.width - src.width % step_size
        height = src.height - src.height % step_size
        window = Window(0, 0, width, height)
        transform = src.window_transform(window)

        out_meta = src.meta
        out_meta.update({
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "transform": transform
        })
        with rio.open(dest_file, "w", **out_meta) as dest:
            dest.write(image.astype(rio.uint8), 1)


def generate_raster_image(pred_mask, pred_file, tif_file, step_size):
    generate_raster(pred_mask, tif_file, pred_file, step_size)
    mask = rio.open(pred_file)

    # Plot image and corresponding boundary
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(label=os.getenv("MODEL_NAME"))
    plt.imshow(mask.read(1))
    plt.savefig('plot')
    #plt.show()
    plt.clf()

    return 'plot', mask.read(1)


def polygonize_raster(src_raster_file, dest_file, tolerance=0.00005):
    src_raster = gdal.Open(src_raster_file)
    band = src_raster.GetRasterBand(1)
    band_array = band.ReadAsArray()
    driver = ogr.GetDriverByName("ESRI Shapefile")
    out_data_src = driver.CreateDataSource(dest_file)
    out_layer = out_data_src.CreateLayer("polygonized", srs=None)
    gdal.Polygonize(band, band, out_layer, -1, [], callback=None)
    out_data_src.Destroy()
    polygons = gpd.read_file(dest_file).simplify(tolerance)
    polygons.to_file(dest_file)


def polygonize_raster_full(cwd, pred_file, shape_name, start_date):
    out_shape_file = cwd + "{}_polygonized_{}.shp".format(shape_name, start_date)
    out_shape_file = out_shape_file.replace(' ', '_')
    polygonize_raster(pred_file, out_shape_file)
    print(f'Exported shape file to: {out_shape_file}')
    polygons = gpd.read_file(out_shape_file)
    ax = polygons.plot(figsize=(10, 10))
    ax.set_title(label=os.getenv("MODEL_NAME"))
    plt.savefig('polig')
    #plt.show()
    plt.clf()

    return 'polig'

def visualize_segmentation_performance(gt_file, prediction, output_path):

    # Load the images
    gt_img = Image.open(gt_file)
    pred_img = Image.fromarray(prediction)

    # Crop the prediction image to match the ground truth image
    pred_img = pred_img.crop((0, 0, gt_img.width, gt_img.height))

    # Convert images to numpy arrays
    gt_array = np.array(gt_img)
    pred_array = np.array(pred_img)

    # Initialize an empty array for the output
    output_array = np.zeros((gt_array.shape[0], gt_array.shape[1], 3), dtype=np.uint8)

    # Set the color codes
    GREEN = [0, 255, 0]  # True Positive
    RED = [255, 0, 0]   # False Positive
    BLACK = [0, 0, 0]    # True Negative
    CYAN = [0, 255, 255] # True Negative

    # Conditions
    TP = np.logical_and(gt_array == 255, pred_array == 1)
    FP = np.logical_and(gt_array == 0, pred_array == 1)
    FN = np.logical_and(gt_array == 255, pred_array == 0)
    TN = np.logical_and(gt_array == 0, pred_array == 0)

    # Assigning colors based on conditions
    output_array[TP] = GREEN
    output_array[FP] = CYAN
    output_array[FN] = RED
    output_array[TN] = BLACK

    # Save the output image
    output_img = Image.fromarray(output_array)
    output_img.save(output_path)

    return output_img


def full_cycle():

    cwd = os.getenv('TRAIN_CWD_DIR') + '/'
    home = os.getenv('HOME_DIR') + '/'
    start_date = '2021-04-19'
    shape_name = os.getenv('REGION_NAME')
    patch_size = int(os.getenv('PATCH_SIZE'))
    sar_polarization = os.getenv('SAR_POLARIZATION')
    tif_file = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Tiling/SAR/Svartadalen/2020-2022/Svartadalen_vh_2021-04-19.tif'

    image = viz_utils.load_image(tif_file, sar_polarization, ignore_nan=False)

    device = utils.get_device()

#OTSU
    #pred_file = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Predictions/pred_OTSU_04-2022.tif'
    #pred_file = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Predictions/pred_OTSU_09-2021.tif'
    pred_file = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Predictions/pred_OTSU_04-2021.tif'

    #gt_file = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Tiling/Mask/Svartadalen/2020-2022/Svartadalen_vh_2022-04-02.tif'
    #gt_file = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Tiling/Mask/Svartadalen/2020-2022/Svartadalen_vh_2021-09-10.tif'
    gt_file = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Tiling/Mask/Svartadalen/2020-2022/Svartadalen_vh_2021-04-19.tif'


    #output_path = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Graphs/OTSU_BEST_2022_04.png'
    output_path = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Graphs/OTSU_BEST_2021_04.png'
    #output_path = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Graphs/OTSU_BEST_2021_09.png'

    wandb.init(
    # set the wandb project where this run will be logged
    project="visualize_ramsar",
    # track hyperparameters and run metadata
    config = dict(
    dataset="Ramsar",
    architecture="OTSU",
    region = shape_name,
    date = start_date,
    )
  )

    model ='OTSU'

    pred_mask = visualize_predicted_image(image, model, device)

    fig , mask = generate_raster_image(1 - pred_mask, pred_file, tif_file, patch_size)
    ras = polygonize_raster_full(cwd, pred_file, shape_name, start_date)

    img = iio.imread("plot.png")
    ras = iio.imread("polig.png")

    image = wandb.Image(img, caption="Input_Image")
    p_m = wandb.Image(mask, caption="Predicted_mask")
    rs = wandb.Image(ras, caption="Poligonised_raster")

    out_color = visualize_segmentation_performance(gt_file, mask, output_path)
    seg_perf = wandb.Image(out_color, caption="Poligonised_raster")

    wandb.log({"Predicted_Mask": p_m, "Input_Image": image, "Poligonised_raster": rs, "Seg_Perf":seg_perf})

    wandb.finish()


def main():
    load_dotenv()
    config = dotenv_values()
    full_cycle()


# start = time.time()
main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
