import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys

import dataset_helpers
import cv2
from pdb import set_trace as bp
from skimage import io, transform

class FacesDataset(data.Dataset):

    def __init__(self, data_dir, phase='train', input_shape=(1, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape

        faces_groups = dataset_helpers.get_dataset(data_dir)
        self.image_list, self.label_list, self.names_list = dataset_helpers.get_image_paths_and_labels(faces_groups)

        print("image_list---------")
        print("--" + str(len(self.image_list)))
        print(self.image_list)
        print("label_list---------")
        print("--" + str(len(self.label_list)))
        print(self.label_list)
        print("--" + str(len(self.names_list)))
        print(self.names_list)
        print("-------------------")

        # with open(os.path.join(data_list_file), 'r') as fd:
        #     imgs = fd.readlines()

        # imgs = [os.path.join(root, img[:-1]) for img in imgs]
        # self.imgs = np.random.permutation(imgs)


        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = Image.open(img_path)
        data = img.convert('RGB')
        data = self.transforms(data)

        label = self.label_list[index]
        name = self.names_list[index]

        return data.float(), label, name


    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    faces_dataset = FacesDataset(data_dir='digits_dataset/train',
                      phase='train',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(faces_dataset, batch_size=1,
                                                shuffle=False, num_workers=1)

    image_count = 0


    for i, (data, label, names) in enumerate(trainloader):

        data, label = data.to(device), label.to(device)

        print("names: " + str(names))
        print("names: " + str(names))

        ## SAVE Batch images to grid
        # i_n = "batch_" + str(i)
        # i_n = 'out_images/' + str(i_n) + '.jpg'
        # torchvision.utils.save_image(data, i_n)

        for (ii, image) in enumerate(data):

            img = image.permute(1, 2, 0).detach().numpy() # CHTO TO TUT NE TAK
            img = img[:, :, [2, 1, 0]]

            ind_label = label[ii].detach().numpy()
            name = names[ii]
            image_name = "label_" + str(name) + "_" + "batch_" + str(i) + "_image_" + str(ii)
            print("imageName: " + str(image_name))
            cv2.imwrite('out_images/' + image_name + '.jpg', img*255.0)

            image_count += 1

    print("\nImages COUNT: " + str(image_count))
    print("\n")
