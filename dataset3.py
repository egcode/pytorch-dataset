from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import sys

import dataset_helpers
from pdb import set_trace as bp
from skimage import io, transform
import imageio

class FacesDataset(data.Dataset):

    def __init__(self,image_list, label_list):

        self.image_list = image_list
        self.label_list = label_list
        # print("image_list---------")
        # print("--" + str(len(self.image_list)))
        # print(self.image_list)
        # print("label_list---------")
        # print("--" + str(len(self.label_list)))
        # print(self.label_list)
        # print("-------------------")

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        self.transforms = T.Compose([
            T.RandomResizedCrop(160, scale=(0.7, 1.0)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = Image.open(img_path)
        data = img.convert('RGB')
        data = self.transforms(data)

        label = self.label_list[index]

        return data.float(), label

    def __len__(self):
        return len(self.image_list)

def get_data(data_dir, device, num_workers, batch_size, batch_size_test):
    dataset = dataset_helpers.get_dataset(data_dir)
    train_set, val_set = dataset_helpers.split_dataset(dataset, 0.05, 0, 'SPLIT_IMAGES')
    
    train_image_list, train_label_list, _ = dataset_helpers.get_image_paths_and_labels(train_set)
    val_image_list, val_label_list, _ = dataset_helpers.get_image_paths_and_labels(val_set)

    train_faces_dataset = FacesDataset(train_image_list, train_label_list)
    test_faces_dataset = FacesDataset(val_image_list, val_label_list)


    trainloader = data.DataLoader(train_faces_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)
    testloader = data.DataLoader(test_faces_dataset, batch_size=batch_size_test,
                                                shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir='digits_dataset/train'
    num_workers = 2
    batch_size = 1
    batch_size_test = 1
    trainloader, testloader = get_data(data_dir, device, num_workers, batch_size, batch_size_test)

    ########### TRAIN IMAGES
    train_image_count = 0
    for i, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)
        for (ii, image) in enumerate(data):
            img = image.permute(1, 2, 0).detach().numpy() 
            ind_label = label[ii].detach().numpy()
            image_name = "label_" + "_" + "batch_" + str(i) + "_image_" + str(ii)
            image_path = 'out_images/train/' + image_name + '.jpg'
            print("imageName: " + str(image_name))

            # # imageio style saving
            imageio.imwrite(image_path, (img * 255.).astype(np.uint8))
            train_image_count += 1
    print("\nTRAIN Images COUNT: " + str(train_image_count))
    print("\n")

    ########### TEST IMAGES
    test_image_count = 0
    for i, (data, label) in enumerate(testloader):
        data, label = data.to(device), label.to(device)
        for (ii, image) in enumerate(data):
            img = image.permute(1, 2, 0).detach().numpy() 
            ind_label = label[ii].detach().numpy()
            image_name = "label_" + "_" + "batch_" + str(i) + "_image_" + str(ii)
            image_path = 'out_images/test/' + image_name + '.jpg'
            print("imageName: " + str(image_name))

            # # imageio style saving
            imageio.imwrite(image_path, (img * 255.).astype(np.uint8))
            test_image_count += 1
    print("\TEST Images COUNT: " + str(test_image_count))
    print("\n")
