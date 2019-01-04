from __future__ import print_function, division

import torch
import torchvision
import numpy as np
from torchvision import datasets, models, transforms
import os
import cv2
from pdb import set_trace as bp
from PIL import Image, ExifTags
import imageio


if __name__ == '__main__':

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'digits_dataset'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=False, num_workers=2)
                for x in ['train', 'val']}

    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

    class_names = image_datasets['train'].classes #List of the class names.
    class_to_idx = image_datasets['train'].class_to_idx #Dict with items (class_name, class_index).
    imgs = image_datasets['train'].imgs #List of (image path, class_index) tuples


    print("\nClass names: " + str(class_names))
    print("\nclass_to_idx: " + str(class_to_idx))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_count = 0

    # for data, target in dataloaders['train']:
    for batch_idx, (data, target) in enumerate(dataloaders['train']):
        data, target = data.to(device), target.to(device)
        print("\n" + str(batch_idx))
        # print("data shape: " + str(data.shape))
        print("target: " + str(target))

        # bp()
        print("class_to_idx: " + str(class_to_idx))

        ## SAVE Batch images to grid
        # im_name = "batch_" + str(batch_idx)
        # batch_path = 'out_images/' + str(im_name) + '.jpg'
        # torchvision.utils.save_image(data, batch_path)


        for ind, (image) in enumerate(data):
            img = image.permute(1, 2, 0).detach().numpy()
            ind_label = target[ind].detach().numpy()
         
            image_name = "label_" + str(class_names[ind_label]) + "_batch_" + str(batch_idx) + "_image_" + str(ind)
            image_path = 'out_images/' + image_name + '.jpg'
            print("imageName: " + str(image_name))

            # # OpenCV style saving
            # cv2_image = img[:, :, [2, 1, 0]] #  OpenCV is in BGR mode
            # cv2.imwrite(image_path, cv2_image*255.0)

            # # imageio style saving
            imageio.imwrite(image_path, (img * 255.).astype(np.uint8))

            image_count += 1

    print("\nImages COUNT: " + str(image_count))
    print("\n")


