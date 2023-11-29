#Code adapted from original source:: https://github.com/melqkiades/deep-wetlands/blob/master/wetlands/generate_sar.py

import json
import os
import time
import numpy
import pandas
import rasterio
import rasterio as rio
import rasterio.mask
from PIL import Image
from dotenv import load_dotenv, dotenv_values
from matplotlib import pyplot as plt
from tqdm import tqdm
import utils, viz_utils, geo_utils

def export_sar_data(tiles, tif_file):
    export_folder = os.getenv('SAR_DIR')
    patch_size = int(os.getenv('PATCH_SIZE'))

    with rio.open(tif_file) as src:
        dataset_array = src.read()
        minValue = numpy.nanpercentile(dataset_array, 1)
        maxValue = numpy.nanpercentile(dataset_array, 99)

    exported_files = []

    for index in tqdm(range(len(tiles)), total=len(tiles)):

        with rio.open(tif_file) as src:

            shape = [tiles.iloc[index]['geometry']]
            name = tiles.iloc[index]['id']
            out_image, out_transform = rio.mask.mask(src, shape, crop=True)
            if np.isnan(out_image).any():
                raise ValueError(f'An image contains NaN values: {name}')

            if out_image.shape[1] == patch_size + 1:
                out_image = out_image[:, :-1, :]
            if out_image.shape[2] == patch_size + 1:
                out_image = out_image[:, :, 1:]

            if out_image.shape[1] != patch_size or out_image.shape[2] != patch_size:
                continue

            # Min-max scale the data to range [0, 1]
            out_image[out_image > maxValue] = maxValue
            out_image[out_image < minValue] = minValue
            out_image = (out_image - minValue) / (maxValue - minValue)

            # Get the metadata of the source image and update it
            # with the width, height, and transform of the cropped image
            out_meta = src.meta
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Save the cropped image as a temporary TIFF file.
            temp_tif = export_folder + '/{}-sar.tif'.format(name)
            with rasterio.open(temp_tif, "w", **out_meta) as dest:
                dest.write(out_image)

            # Save the cropped image as a temporary PNG file.
            temp_png = export_folder + '/{}-sar.png'.format(name)

            # Get the color map by name:
            cm = plt.get_cmap('gray')

            # Apply the colormap like a function to any array:
            colored_image = cm(out_image[0])

            # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
            # But we want to convert to RGB in uint8 and save it:
            Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(temp_png)

            exported_files.append({'index': tiles.index.values[index], 'id': name})

    return exported_files


def full_cycle():
    file_name = os.getenv('GEOJSON_FILE')
    region_name = os.getenv('REGION_NAME')
    tif_files = os.getenv('SAR_TIFF_FILE_test')#os.getenv('SAR_TIFF_FILE_train')
    country_code = os.getenv('COUNTRY_CODE')
    region_admin_level = os.getenv("REGION_ADMIN_LEVEL")
    patch_size = int(os.getenv("PATCH_SIZE"))

    utils.download_country_boundaries(country_code, region_admin_level, file_name)
    geoboundary = utils.get_region_boundaries(region_name, file_name)

    tiles_lists = []

    for file in os.listdir(tif_files):

        path = tif_files + '/' + file
        tiles = geo_utils.get_tiles(region_name, path, file, geoboundary, patch_size)
        tiles_list = export_sar_data(tiles, path)
        tiles_lists.append(tiles_list)

    tiles_dataframe = pandas.DataFrame(tiles_lists[0])

    for list in tiles_lists[1:]:
        tiles_dataframe = pandas.concat([tiles_dataframe, pandas.DataFrame(list)],ignore_index=True)# , axis = 0)

    tiles_file = os.getenv("TILES_FILE")
    tiles_dataframe.to_csv(tiles_file, columns=['id'], index_label='index')


def main():
    load_dotenv()
    config = dotenv_values()
    full_cycle()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
