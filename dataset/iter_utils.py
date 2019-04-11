import os
import tensorflow as tf
from dataset.dataset_generator import parse_tfrecord

# create batch iterator
def batch_iterator(dataset_dir, epochs, batch_size, training=False, drop_remainder=True):
    if os.path.isfile(dataset_dir) is False:
        raise FileNotFoundError(dataset_dir, 'not exist')
    dataset = tf.data.TFRecordDataset(dataset_dir)
    dataset = dataset.map(parse_tfrecord)
    if training is True:
        dataset = dataset.shuffle(100000)
        dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    iterator = dataset.make_initializable_iterator()

    return iterator