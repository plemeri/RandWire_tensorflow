

import tensorflow as tf
import numpy as np
import argparse
from dataset import iter_utils

# argument parser for options
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_num', type=int, default=10, help='number of class')  # number of class
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/best', help='directory for checkpoint')  # directory for checkpoint
    parser.add_argument('--test_record_dir', type=str, default='./dataset/cifar10/test.tfrecord', help='directory for test record')  # directory for test images
    parser.add_argument('--batch_size', type=int, default=256, help='number of images for each batch')  # number of images for each batch
    args = parser.parse_args()

    return args

# main function for test
def main(args):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('[Input Arguments]')
    for arg in args.__dict__:
        print(arg, '->', args.__dict__[arg])
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    with tf.Session() as sess:
        #restoring network and weight data
        try:
            saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(args.checkpoint_dir) + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
        except:
            print('failed to load network and checkpoint')
            return
        print('network graph and checkpoint restored')

        # create batch iterator

        test_iterator = iter_utils.batch_iterator(args.test_record_dir, None, args.batch_size, training=False, drop_remainder=False)
        test_images_batch, test_labels_batch = test_iterator.get_next()
        sess.run(test_iterator.initializer)

        graph = tf.get_default_graph()

        # get tensor for feed forward
        images = graph.get_tensor_by_name('images:0')
        labels = graph.get_tensor_by_name('labels:0')
        prediction = graph.get_tensor_by_name('accuracy/prediction:0')
        training = graph.get_tensor_by_name('training:0')

        predictions = 0
        dataset_size = 0

        # test
        while True:
            try:
                test_images, test_labels = sess.run([test_images_batch, test_labels_batch])
                test_labels = np.eye(args.class_num)[test_labels]
                prediction_ = sess.run(prediction, feed_dict={images: test_images, labels: test_labels, training: False})
                predictions += np.sum(prediction_.astype(int))
                dataset_size += len(prediction_)
                print('\r{0} done'.format(dataset_size), end='')
            except tf.errors.OutOfRangeError:
                print('\n')
                break

        print('test accuracy: ', (predictions / dataset_size) * 100, '%')


if __name__ == '__main__':
    args = args()
    main(args)
