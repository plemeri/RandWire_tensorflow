import os
import tensorflow as tf
from dataset.dataset_generator import parse_tfrecord

def flip(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

def pad_and_crop(image, label, shape, pad_size=2):
    image = tf.image.pad_to_bounding_box(image, pad_size, pad_size, shape[0]+pad_size*2, shape[1]+pad_size*2)
    image = tf.image.random_crop(image, shape)
    return image, label

def standardization(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

# create batch iterator
def batch_iterator(dataset_dir, epochs, batch_size, augmentation=None, training=False, drop_remainder=True):
    if os.path.isfile(dataset_dir) is False:
        raise FileNotFoundError(dataset_dir, 'not exist')
    dataset = tf.data.TFRecordDataset(dataset_dir)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(standardization)
    if training is True:
        dataset = dataset.shuffle(100000)
        dataset = dataset.repeat(epochs)
        if augmentation is not None:
            for aug_func in augmentation:
                dataset = dataset.map(aug_func, num_parallel_calls=len(augmentation))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    iterator = dataset.make_initializable_iterator()

    return iterator
