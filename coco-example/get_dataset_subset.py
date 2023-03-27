from pycocotools.coco import COCO
import json


coco_train = COCO("/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/instances_train2017.json")
coco_val = COCO("/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/instances_val2017.json")

datasets = {'train':coco_train, 'val':coco_val}

# define a list of category names
category_names = ['cat']

for dataset in datasets:
    # get the category IDs of the specified categories
    category_ids = datasets[dataset].getCatIds(catNms=category_names)

    # get the IDs of the images that contain the specified categories
    img_ids = datasets[dataset].getImgIds(catIds=category_ids)

    # load the metadata for the specified images
    img_metadata = datasets[dataset].loadImgs(img_ids)

    # create a new dictionary for the subset
    subset_data = {
        'images': [],
        'annotations': []
    }

    # iterate over the metadata for the specified images
    for img in img_metadata:
        # add the image metadata to the subset
        subset_data['images'].append(img)

        # get the annotations for the image
        ann_ids = datasets[dataset].getAnnIds(imgIds=img['id'], catIds=category_ids)
        anns = datasets[dataset].loadAnns(ann_ids)

        # add the annotations to the subset
        subset_data['annotations'].extend(anns)

    print(len(img_metadata))

    # save the subset to a new JSON file
    with open('/path/to/dir/instances_'+ dataset + '2017_subset_cat.json', 'w') as f:
        json.dump(subset_data, f)