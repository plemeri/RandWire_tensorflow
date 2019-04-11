# MNIST image and label reader & dataset provider for tensorflow networks
# taehun kim
# taehoon1018@postech.ac.kr

import numpy as np
import struct
from PIL import Image

# read MNIST dataset - image
def read_image(filename):
    raw = open(filename, 'rb')
    magic_num = struct.unpack("i", raw.read(4)[::-1])[0]
    if magic_num != 2051:
        print(filename, ' could be wrong file, failed to read')
        return
    item_num = struct.unpack("i", raw.read(4)[::-1])[0]
    row_num = struct.unpack("i", raw.read(4)[::-1])[0]
    col_num = struct.unpack("i", raw.read(4)[::-1])[0]
    image = np.zeros([item_num, row_num * col_num])
    for i in range(0, item_num):
        image[i, :] = np.fromstring(raw.read(row_num * col_num), dtype=np.uint8)
    image = image.reshape(-1, row_num, col_num, 1)
    return image

# read MNIST dataset - label
def read_label(filename):
    raw = open(filename, 'rb')
    magic_num = struct.unpack("i", raw.read(4)[::-1])[0]
    if magic_num != 2049:
        print(filename, ' could be wrong file, failed to read')
        return
    item_num = struct.unpack("i", raw.read(4)[::-1])[0]
    label = np.zeros([item_num, 10])
    class_ = []
    for i in range(0, item_num):
        temp = np.fromstring(raw.read(1), dtype=np.uint8)
        label[i][temp] = 1
        if (temp in class_) == False:
            class_.append(temp)
    class_num = len(np.unique(class_))

    return label

def saveas_image(image_dir, label_dir, output_dir):
    image = read_image(image_dir)
    label = read_label(label_dir)
    for i in range(image.shape[0]):
        Image.fromarray(image[i]).save(output_dir + '/images/{0}.png'.format(str(i + 1)))
        f = open(output_dir + '/labels/{0}.txt'.format(str(i+1)), 'w')
        f.write(str(np.argmax(label[i])))
        f.close()

if __name__ == '__main__':
    image_dir = './train-images-idx3-ubyte'
    label_dir = './train-labels-idx1-ubyte'
    saveas_image(image_dir, label_dir, './train')