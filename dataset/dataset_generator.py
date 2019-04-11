import tensorflow as tf
import numpy as np
import argparse
import os
from PIL import Image

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./cifar10/test/images')
    parser.add_argument('--label_dir', type=str, default='./cifar10/test/labels')
    parser.add_argument('--output_dir', type=str, default='./cifar10')
    parser.add_argument('--output_filename', type=str, default='test.tfrecord')
    parser.add_argument('--val_set_size', type=int, default=200)
    args = parser.parse_args()
    return args

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tfrecord(image_dir, label_dir, output_path):

    def image_and_label(image_dir, label_dir):
        image_list = os.listdir(image_dir)
        label_list = os.listdir(label_dir)
        for image_file, label_file in zip(image_list, label_list):
            image = Image.open(os.path.join(image_dir, image_file))
            label = open(os.path.join(label_dir, label_file), 'r')
            label = int(label.read())
            yield image, label

    count = 0
    writer = tf.python_io.TFRecordWriter(output_path)
    for image, label in image_and_label(image_dir, label_dir):
        height = np.array(image).shape[0]
        width = np.array(image).shape[1]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': int64_feature(height),
            'width': int64_feature(width),
            'image/raw': bytes_feature(image.tobytes()),
            'label': int64_feature(label)
        }))
        writer.write(example.SerializeToString())
        count += 1
        print('\r{0} done'.format(count), end='')
    writer.close()

def parse_tfrecord(record):

    keys_to_features = {
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64),
        'image/raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64),
    }

    features = tf.parse_single_example(record, features=keys_to_features)
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    image = tf.cast(features['image/raw'], tf.string)
    label = tf.cast(features['label'], tf.int64)

    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, shape=[height, width, -1])

    return image, label

if __name__ == "__main__":
    args = args()
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('[Input Arguments]')
    for arg in args.__dict__:
        print(arg, '->', args.__dict__[arg])
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    create_tfrecord(args.image_dir, args.label_dir, os.path.join(args.output_dir, args.output_filename))
