[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploring-randomly-wired-neural-networks-for/image-classification-imagenet-image-reco)](https://paperswithcode.com/sota/image-classification-imagenet-image-reco?p=exploring-randomly-wired-neural-networks-for)
# RandWire_tensorflow
tensorflow implementation of [Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569) using **Cifar10, MNIST**

![alt text](https://raw.githubusercontent.com/swdsld/RandWire_tensorflow/master/tensorboard.PNG)

## Requirements
* Tensorflow 1.x - GPU version recommended
* Python 3.x
* networkx 2.x
* pyyaml 5.x

## Dataset

Please download dataset from this [link](https://drive.google.com/drive/folders/1kr0bGAmf3xuOUkw1DTA8gSBsO9LTObyk?usp=sharing)
Both Cifar10 and MNIST dataset are converted into tfrecords format for conveinence. Put `train.tfrecords`, `test.tfrecords` files into `dataset/cifar10`, `dataset/mnist`

You can create tfrecord file with your own dataset with `dataset/dataset_generator.py`.
```sh
python dataset_generator.py --image_dir ./cifar10/test/images --label_dir ./cifar10/test/labels --output_dir ./cifar10 --output_filename test.tfrecord
```

Options:

- `--image_dir` (str) - directory of your image files. it is recommended to set the name of images to integers like `0.png`
- `--label_dir` (str) - directory of your label files. it is recommended to set the name of images to integers like `0.txt`. label text file must contain class label in integer like `8`. 
- `--output_dir` (str) - directory for output tfrecord file.
- `--outpuf_filename` (str) - filename of output tfrecord file.

## Experiments

Datasets | Model | Parameters | Accuracy | Epoch
----------|----------|----------|----------|----------
CIFAR-10 | ResNet110 (Paper) | 1.7M | 93.57% | 300
CIFAR-10 | RandWire (my_small_regime) | 1.2M | 93.64% | 60
CIFAR-100 | RandWire (my_regime) | 8M | 74.49% | 100

(19.04.18 changed) I trained on Cifar10 dataset and get `6.36 %` error on test set. You can download pretrained network from [here](https://drive.google.com/drive/folders/1Pi9Z306S3fvBLBOy6oPDGQDNzsKdrtzG?usp=sharing). Unzip the file and move all files under `checkpoint` file or your checkpoint directory and try running test script to check the accuracy.
The number of parameters used for cifar10 model is aboud 1.2M, which is similar result on ResNet-110 (6.43 %) which used 1.7M parameters.

(19.04.16 added) I trained on Cifar100 dataset and get `74.49%` accuracy on test set. You can download pretrained network from same link above.

## Training

**Cifar 10**
```sh
python train.py --class_num 10 --image_shape 32 32 3 --stages 4 --channel_count 78 --graph_model ws --graph_param 32 4 0.75 --dropout_rate 0.2 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --train_set_size 50000 --val_set_size 10000 --batch_size 100 --epochs 100 --checkpoint_dir ./checkpoint --checkpoint_name randwire_cifar10 --train_record_dir ./dataset/cifar10/train.tfrecord --val_record_dir ./dataset/cifar10/test.tfrecord
```

Options:
- `--class_num` (int) - output number of class. Cifar10 has 10 classes.
- `--image_shape` (int nargs) - shape of input image. Cifar10 has 32 32 3 shape.
- `--stages` (int) - stage (or block) number of randwire network. 
- `--channel_count` (int) - channel count of randwire network. please refer to the paper
- `--graph_model` (str) - currently randwire has 3 random graph models. you can choose from 'er', 'ba' and 'ws'.
- `--graph_param` (float nargs) - first value is node count. for 'er' and 'ba', there are one extra parameter so it would be like `32 0.4` or `32 7`. for 'ws' there are two extra parameters like above.
- `--learning_rate` (float) - initial learning rate
- `--momentum` (float) - momentum from momentum optimizer
- `--weight_decay` (float) - weight decay factor
- `--train_set_size` (int) - number of training data. Cifar10 has 50000 data.
- `--val_set_size` (int) - number of validating data. I used test data for validation, so there are 10000 data.
- `--batch_size` (int) - size of mini batch
- `--epochs` (int) - number of epoch
- `--checkpoint_dir` (str) - directory to save checkpoint
- `--checkpoint_name` (str) - file name of checkpoint
- `--train_record_dir` (str) - file location of training set tfrecord
- `--test_record_dir` (str) - file location of test set tfrecord (for validation)

**MNIST**
```sh
python train.py --class_num 10 --image_shape 28 28 1 --stages 4 --channel_count 78 --graph_model ws --graph_param 32 4 0.75 --dropout_rate 0.2 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --train_set_size 50000 --val_set_size 10000 --batch_size 100 --epochs 100 --checkpoint_dir ./checkpoint --checkpoint_name randwire_mnist --train_record_dir ./dataset/mnist/train.tfrecord --val_record_dir ./dataset/mnist/test.tfrecord
```

Options:
- options are same as Cifar10

**Cifar100**
(19.04.16 added)
```sh
python train.py --class_num 100 --image_shape 32 32 3 --stages 4 --channel_count 78 --graph_model ws --graph_param 32 4 0.75 --dropout_rate 0.2 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --train_set_size 50000 --val_set_size 10000 --batch_size 100 --epochs 100 --checkpoint_dir ./checkpoint --checkpoint_name randwire_cifar100 --train_record_dir ./dataset/cifar100/train.tfrecord --val_record_dir ./dataset/cifar100/test.tfrecord
```

Options:
- options are same as Cifar10

## Testing
```sh
python test.py --class_num --checkpoint_dir ./checkpoint/best --test_record_dir ./dataset/cifar10/test.tfrecord --batch_size 256
```
Options:
- `--class_num` (int) - the number of classes
- `--checkpoint_dir` (str) - directory for the checkpoint you want to load and test
- `--test_record_dir` (str) - directory for the test dataset
- `--batch_size` (int) - batch size for testing

test.py loads network graph and tensors from meta data and evalutes.

**Implementation Details**

- Learning rate decreases by multiplying 0.1 in 50% and 75% of entire training step.

- I made an option `init_subsample` in `my_regime`, `my_small_regime` and `small_regime` in `RandWire.py` which do not to use stride 2 for the initial convolutional layer since cifar10 and mnist has low resolution. if you set `init_subsample` False, then it will use stride 2.

- While training, it will save the checkpoint with best validation accuracy.

- While training, it will save tensorboard log for training and validation accuracy and loss in `[YOUR_CHECKPOINT_DIRECTORY]/log`. You can visualize yourself with tensorboard.

- I'm currently working on drop connection for regularization and downloading ImageNet dataset to train on my implementation.

- I added dropout layer after the Relu-Conv-BN triplet unit for regularization. You can set dropout_rate 0.0 to disable it.

- In train.py, you can use `small_regime` or `regular_regime` instead of `my_regime` and `my_small_regime`. Both do not use stride 2 in order to prevent subsampling to maintain the spatial information since cifar datasets are not large enough.

```python
  # output logit from NN
  output = RandWire.my_small_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
                              args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False, training)
  # output = RandWire.small_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
  #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False,
  #                             training)
  # output = RandWire.regular_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
  #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', training)
```
