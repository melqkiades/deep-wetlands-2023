import os
from itertools import product
import rasterio as rio
from rasterio import windows
import glob

in_path = 'NormalizedData'
out_path = '../Downloads/SAR_norm_unsplit/SAR/data'

filelist = glob.glob(os.path.join(in_path, '*'))


def get_tiles(ds, width=64, height=64):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        if window.width == width and window.height == height:
            transform = windows.transform(window, ds.transform)
            yield window, transform


for file in sorted(filelist):
    input_filename = file.split('/')[1]
    input_filename = input_filename.split('.')[0]
    output_filename = input_filename + '_tile_{}-{}.tif'
    with rio.open(file) as inds:
        tile_width, tile_height = 64, 64

        meta = inds.meta.copy()

        for window, transform in get_tiles(inds):
            print(window)
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            outpath = os.path.join(out_path,output_filename.format('{:04d}'.format(int(window.col_off)), '{:04d}'.format(int(window.row_off))))
            with rio.open(outpath, 'w', **meta) as outds:
                outds.write(inds.read(window=window))

