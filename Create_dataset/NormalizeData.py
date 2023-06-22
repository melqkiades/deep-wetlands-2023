import rasterio
import glob
import numpy as np
import os

source = 'Source'
output_source = 'NormalizedData'

source_list = glob.glob(os.path.join(source, '*.tif'))
count = 0
for file in source_list:
    count += 1
    with rasterio.open(file) as tif:
        meta = tif.meta.copy()
        filename = file.split('/')[1]

        vh = tif.read(1)
        highest = np.nanmax(vh)
        lowest = np.nanmin(vh)
        vh_norm = (vh - lowest) / (highest - lowest)

        vv = tif.read(2)
        highest = np.nanmax(vv)
        lowest = np.nanmin(vv)
        vv_norm = (vv - lowest) / (highest - lowest)

        outpath = os.path.join(output_source, filename)
        with rasterio.open(outpath, 'w', **meta) as outds:
            outds.write_band(1, vh_norm)
            outds.write_band(2, vv_norm)

print(count)
