import torch
from torch.utils.data import Dataset
import glob
import os
import rasterio
import numpy as np

class SARDataset(Dataset):
    def __init__(self, input_path, target_path, three_channels, transform=None):
        self.input_path = input_path
        self.input = sorted(glob.glob(os.path.join(self.input_path, '*.tif')))
        self.target_path = target_path
        self.target = sorted(glob.glob(os.path.join(self.target_path, '*.tif')))
        self.transform = transform
        self.three_channels = three_channels

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        input = rasterio.open(self.input[item])
        vh = input.read(1)
        if self.three_channels:
            vh = np.array([vh, vh, vh])
            vh = np.transpose(vh, (1, 2, 0))

        target = rasterio.open(self.target[item])
        mask = target.read(1)
        mask = mask // 255

        if self.transform is not None:
            aux = self.transform(image=vh, mask=mask)
            vh = aux['image']
            mask = aux['mask']
        vh = vh.float()
        return vh, mask


#train_set = SARDataset('../../../Source', '../../../Target', None)
#
#vh, mask = train_set[0]
#
#fig, axs = plt.subplots(2, 1)
#axs[0].imshow(np.transpose(vh, (1, 2, 0)))
#axs[0].set_axis_off()
#axs[1].imshow(np.transpose([mask, mask, mask], (1, 2, 0)))
#axs[1].set_axis_off()
#
#fig.show()
