from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import cv2

import argparse
import dataset_helpers
import lfw
import h5py
import math
from scipy import misc

from pdb import set_trace as bp


tfe = tf.contrib.eager
tf.enable_eager_execution()

NUM_PARALLEL_CALLS = 2 # number of cpu cores
BATCH_SIZE = 7

def main(args):

    dataset = dataset_helpers.get_dataset(args.data_dir)
    image_list, label_list, names_list = dataset_helpers.get_image_paths_and_labels(dataset)

    print("image_list---------")
    print("--" + str(len(image_list)))
    print(image_list)
    print("label_list---------")
    print("--" + str(len(label_list)))
    print(label_list)
    print("--" + str(len(names_list)))
    print(names_list)
    print("-------------------")
    # bp()
    # create dataset
    # dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
    # dataset = dataset.map(_parse_function, num_parallel_calls=NUM_PARALLEL_CALLS)
    # dataset = dataset.repeat(5)
    # dataset = dataset.batch(BATCH_SIZE)

    # # print("-----------")
    # # print(dataset)
    # # print("-----------")

    # count = 0

    # # training_init_op = iterator.make_initializer(dataset)

    # for (batch, (images, labels)) in enumerate(dataset):
    #     # img_batch, label_batch = sess.run(next_batch)
    #     print("\nBatch : " + str(batch))
        
    #     for i, image in enumerate(images):
            # # print(count)
            # count += 1
    #         print(image.shape)
    #         print("label: " + str(labels[i].numpy()))


            # image_to_display = image.numpy()  
            # image_name = "label_" + str(labels[i].numpy()) + "__" + "image_" + str(count)
            # print("imageName: " + str(image_name))
            # write_to_image(image_name, image_to_display)

    
def write_to_image(name, array):
    array = array*255.0
    cv2.imwrite('out_images/' + name + '.jpg', array)


def _parse_function(filename, label):
    """Input parser for samples of the training set."""
    # convert label number into one-hot-encoding

    # one_hot = tf.one_hot(label, self.num_classes)

    # load and preprocess the image
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    image = tf.image.resize_images(img_decoded, [120, 120])


    """
    Data augmentation comes here.
    """
    # Flip Works
    # image = tf.image.flip_left_right(image) 


    # Random rotate Works
    image = tf.py_func(random_rotate_image, [image], tf.uint8)
    image = tf.cast(image, tf.float32)

    # Random Crop Works
    # image_size_crop = image_size[0]-5, image_size[1]-5
    # image = tf.random_crop(image, image_size_crop + (3,))
    # image = tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1])
    image = tf.py_func(random_crop_image, [image], tf.float32)

    # Random Flip Works
    # image = tf.image.random_flip_left_right(image) 
    image = tf.py_func(random_flip_image, [image], tf.float32)

    # Standartization Works (not sure what that is)
    image = (tf.cast(image, tf.float32) - 127.5)/128.0
    image = tf.image.per_image_standardization(image)


    # img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

    # RGB -> BGR
    # img_bgr = img_centered[:, :, ::-1]

    # Normalize
    # image = tf.divide(image, 255.0)

    return image, label

def random_flip_image(image):
    # if random.choice([True, False]):
    if random.random() < 0.3:
        # print("FLIP")
        return tf.image.flip_left_right(image) 
    else:
        # print("NOT FLIP")
        return image

def random_crop_image(image):
    if random.choice([True, False]):
        image_size = (int(image.shape[0]), int(image.shape[1]))
        image_size_crop = image_size[0]-5, image_size[1]-5
        image = tf.random_crop(image, image_size_crop + (3,))
        return tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1])
    else:
        return image


def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    # print("angle: " + str(angle))
    return misc.imrotate(image, angle, 'bicubic')
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
        help='Path to images',
        default='./digits_dataset/train')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
