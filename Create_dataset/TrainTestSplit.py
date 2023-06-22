import splitfolders

splitfolders.ratio('Unsplit_optical/SAR', 'SAR+optical_dataset/SAR', seed=1234, ratio=(0.8, 0.2), move=False)

splitfolders.ratio('Unsplit_optical/Annotations', 'SAR+optical_dataset/Annotations', seed=1234, ratio=(0.8, 0.2), move=False)

splitfolders.ratio('Unsplit_optical/Optical', 'SAR+optical_dataset/Optical', seed=1234, ratio=(0.8, 0.2), move=False)
