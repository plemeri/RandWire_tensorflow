import pickle
import numpy as np
from PIL import Image


def read_cifar10(filename):
    with open(filename, 'rb') as raw:
        dict_ = pickle.load(raw, encoding='bytes')
        print(dict_[b'batch_label'])
        labels_ = dict_[b'labels']
        labels = np.zeros([len(labels_), len(np.unique(labels_))])
        for i in range(0, len(labels_)):
            labels[i, labels_[i]] = 1
        images = dict_[b'data']
        images = images.reshape([-1, 3, 32, 32])
        images = images.transpose(0, 2, 3, 1)
    return images, labels


def read_all_batch():
    images, labels = read_cifar10('./data_batch_1')
    for i in range(2, 6):
        temp_image, temp_label = read_cifar10('./data_batch_'+str(i))
        images = np.append(images, temp_image, 0)
        labels = np.append(labels, temp_label, 0)
    return images, labels


def read_meta(filename='./batches.meta'):
    with open(filename, 'rb') as file:
        dict_ = pickle.load(file, encoding='bytes')
        label = dict_[b'label_names']
        label = [s.decode('utf-8') for s in label]
    return label

def saveas_image(batch_dir, output_dir):
    image, label = read_cifar10(batch_dir)
    for i in range(image.shape[0]):
        Image.fromarray(image[i]).save(output_dir + '/images/{0}.png'.format(str(i+1)))
        f = open(output_dir + '/labels/{0}.txt'.format(str(i+1)), 'w')
        f.write(str(np.argmax(label[i])))
        f.close()

if __name__ == '__main__':
    image_dir = './train-images-idx3-ubyte'
    label_dir = './train-labels-idx1-ubyte'
    saveas_image('./test_batch', './test')