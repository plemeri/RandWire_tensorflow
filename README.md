# RandWireNN_tensorflow
tensorflow implementation of [Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569) using **Cifar10, MNIST**

## Requirements
* Tensorflow 1.x - GPU version recommended
* Python 3.x
* networkx 2.x
* pyyaml 5.x

## Dataset

Please download dataset from this [link](https://drive.google.com/drive/folders/1kr0bGAmf3xuOUkw1DTA8gSBsO9LTObyk?usp=sharing)
Both Cifar10 and MNIST dataset are converted into tfrecords format for conveinence. Put **train.tfrecords, test.tfrecords** files into **dataset/cifar10, dataset/mnist**

## Usage

**Cifar 10**
```sh
python train --class_num 10 --image_shape 32 32 3 --stages 5 --channel_count 64 --graph_model ws --graph_param 32 4 0.75 --dropout_rate 0.0 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --train_set_size 50000 --val_set_size 10000 --batch_size 64 --epochs 300 --checkpoint_dir ./checkpoint --checkpoint_name randwire_cifar10 --train_record_dir ./dataset/cifar10/train.tfrecord --val_record_dir ./dataset/cifar10/test.tfrecord
```

Options:
- `--class_num` (int) - output number of class. Cifar10 has 10 classes.
- `--image_shape` (int nargs) - shape of input image. Cifar10 has 32*32*3 shape.
- `--stages` (int) - stage (or block) number of randwire network. 
- `--channel_count` - channel count of randwire network. please refer to the paper
- `--graph_model` - currently randwire has 3 random graph models. you can choose from 'er', 'ba' and 'ws'.
- `--learning_rate` - initial learning rate
- `--momentum` - momentum from momentum optimizer
- `--weight_decay` - weight decay factor
- `--train_set_size` - number of training data. Cifar10 has 50000 data.
- `--val_set_size` - number of validating data. I used test data for validation, so there are 10000 data.
- `--batch_size` - size of mini batch
- `--epochs` - number of epoch
- `--checkpoint_dir` - directory to save checkpoint
- `--checkpoint_name` - file name of checkpoint
- `--train_record_dir` - file location of training set tfrecord
- `--test_record_dir` - file location of test set tfrecord (for validation)

**MNIST**
```sh
python train --class_num 10 --image_shape 28 28 1 --stages 4 --channel_count 32 --graph_model ws --graph_param 32 4 0.75 --dropout_rate 0.0 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --train_set_size 50000 --val_set_size 10000 --batch_size 64 --epochs 300 --checkpoint_dir ./checkpoint --checkpoint_name randwire_cifar10 --train_record_dir ./dataset/cifar10/train.tfrecord --val_record_dir ./dataset/cifar10/test.tfrecord
```

Options:
- options are same as Cifar10
