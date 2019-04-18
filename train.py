import tensorflow as tf
import argparse
import numpy as np
from network import RandWire
from dataset import iter_utils
import os

# argument parser for options
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_num', type=int, default=10, help='number of class')  # number of class
    parser.add_argument('--image_shape', type=int, nargs='+', default=[32, 32, 3], help='shape of image - height, width, channel')  # shape of image - height, width, channel
    parser.add_argument('--stages', type=int, default=4, help='stage number of randwire')  # number of dense blocks
    parser.add_argument('--channel_count', type=int, default=78)
    parser.add_argument('--graph_model', type=str, default='ws')
    parser.add_argument('--graph_param', type=float, nargs='+', default=[32, 4, 0.75])
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate for dropout')  # dropout rate for dropout
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='initial learning rate')  # initial learning rate
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for momentum optimizer')  # momentum for momentum optimizer
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay factor')  # weight decay factor
    parser.add_argument('--train_set_size', type=int, default=50000, help='number of images for training set')  # number of images for training set
    parser.add_argument('--val_set_size', type=int, default=10000, help='number of images for validation set, 0 for skip validation')  # number of images for validation set, 0 for skip validation
    parser.add_argument('--batch_size', type=int, default=100, help='number of images for each batch')  # number of images for each batch
    parser.add_argument('--epochs', type=int, default=100, help='total epochs to train')  # total epochs to train
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='directory for checkpoint')  # directory for checkpoint
    parser.add_argument('--checkpoint_name', type=str, default='randwire_cifar10', help='filename for checkpoint')
    parser.add_argument('--train_record_dir', type=str, default='./dataset/cifar10/train.tfrecord', help='directory for training records')  # directory for training images
    parser.add_argument('--val_record_dir', type=str, default='./dataset/cifar10/test.tfrecord', help='directory for validation records')  # directory for training labels

    args = parser.parse_args()

    return args

# main function for training
def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('[Input Arguments]')
    for arg in args.__dict__:
        print(arg, '->', args.__dict__[arg])
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')


    images = tf.placeholder('float32', shape=[None, *args.image_shape], name='images')  # placeholder for images
    labels = tf.placeholder('float32', shape=[None, args.class_num], name='labels')  # placeholder for labels
    training = tf.placeholder('bool', name='training')  # placeholder for training boolean (is training)
    global_step = tf.get_variable(name='global_step', shape=[], dtype='int64', trainable=False)  # variable for global step
    best_accuracy = tf.get_variable(name='best_accuracy', dtype='float32', trainable=False, initializer=0.0)
    
    steps_per_epoch = round(args.train_set_size / args.batch_size)
    learning_rate = tf.train.piecewise_constant(global_step, [round(steps_per_epoch * 0.5 * args.epochs),
                                                              round(steps_per_epoch * 0.75 * args.epochs)],
                                                [args.learning_rate, 0.1 * args.learning_rate,
                                                 0.01 * args.learning_rate])
    # output logit from NN
    output = RandWire.my_small_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
                                args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False, training)
    # output = RandWire.my_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
    #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False, training)
    # output = RandWire.small_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
    #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False,
    #                             training)
    # output = RandWire.regular_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
    #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', training)

    #loss and optimizer
    with tf.variable_scope('losses'):
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output)
        loss = tf.losses.softmax_cross_entropy(labels, output, label_smoothing=0.1)
        loss = tf.reduce_mean(loss, name='loss')
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')

    with tf.variable_scope('optimizers'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=args.momentum, use_nesterov=True)
        #optimizer = tf.train.AdamOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(loss + l2_loss * args.weight_decay, global_step=global_step)
        train_op = tf.group([train_op, update_ops], name='train_op')

    #accuracy
    with tf.variable_scope('accuracy'):
        output = tf.nn.softmax(output, name='output')
        prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1), name='prediction')
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')

    # summary
    train_loss_summary = tf.summary.scalar("train_loss", loss)
    val_loss_summary = tf.summary.scalar("val_loss", loss)
    train_accuracy_summary = tf.summary.scalar("train_acc", accuracy)
    val_accuracy_summary = tf.summary.scalar("val_acc", accuracy)

    saver = tf.train.Saver()
    best_saver = tf.train.Saver()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.checkpoint_dir + '/log', sess.graph)

        sess.run(tf.global_variables_initializer())
        augmentations = [lambda image, label: iter_utils.pad_and_crop(image, label, args.image_shape, 4), iter_utils.flip]
        train_iterator = iter_utils.batch_iterator(args.train_record_dir, args.epochs, args.batch_size, augmentations, True)
        train_images_batch, train_labels_batch = train_iterator.get_next()
        val_iterator = iter_utils.batch_iterator(args.val_record_dir, args.epochs, args.batch_size)
        val_images_batch, val_labels_batch = val_iterator.get_next()
        sess.run(train_iterator.initializer)
        if args.val_set_size != 0:
            sess.run(val_iterator.initializer)

        # restoring checkpoint
        try:
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
            print('checkpoint restored. train from checkpoint')
        except:
            print('failed to load checkpoint. train from the beginning')

        #get initial step
        gstep = sess.run(global_step)
        init_epoch = round(gstep / steps_per_epoch)
        init_epoch = int(init_epoch)

        for epoch_ in range(init_epoch + 1, args.epochs + 1):

            # train
            while gstep * args.batch_size < epoch_ * args.train_set_size:
                try:
                    train_images, train_labels = sess.run([train_images_batch, train_labels_batch])
                    train_labels = np.eye(args.class_num)[train_labels]
                    gstep, _, loss_, accuracy_, train_loss_sum, train_acc_sum = sess.run(
                        [global_step, train_op, loss, accuracy, train_loss_summary, train_accuracy_summary],
                        feed_dict={images: train_images, labels: train_labels, training: True})
                    print('[global step: ' + str(gstep) + ' / epoch ' + str(epoch_) + '] -> train accuracy: ',
                          accuracy_, ' loss: ', loss_)
                    writer.add_summary(train_loss_sum, gstep)
                    writer.add_summary(train_acc_sum, gstep)
                except tf.errors.OutOfRangeError:
                    break

            predictions = []

            # validation
            if args.val_set_size != 0:
                while True:
                    try:
                        val_images, val_labels = sess.run([val_images_batch, val_labels_batch])
                        val_labels = np.eye(args.class_num)[val_labels]
                        loss_, accuracy_, prediction_, val_loss_sum, val_acc_sum = sess.run(
                            [loss, accuracy, prediction, val_loss_summary, val_accuracy_summary],
                            feed_dict={images: val_images, labels: val_labels, training: False})
                        predictions.append(prediction_)
                        print('[epoch ' + str(epoch_) + '] -> val accuracy: ', accuracy_, ' loss: ', loss_)
                        writer.add_summary(val_loss_sum, gstep)
                        writer.add_summary(val_acc_sum, gstep)
                    except tf.errors.OutOfRangeError:
                        sess.run(val_iterator.initializer)
                        break

            saver.save(sess, args.checkpoint_dir + '/' + args.checkpoint_name, global_step=global_step)

            predictions = np.concatenate(predictions)
            print('best: ', best_accuracy.eval(), '\ncurrent: ', np.mean(predictions))
            if best_accuracy.eval() < np.mean(predictions):
                print('save checkpoint')
                best_accuracy = tf.assign(best_accuracy, np.mean(predictions))
                best_saver.save(sess, args.checkpoint_dir + '/best/' + args.checkpoint_name)


if __name__ == '__main__':
    args = args()
    main(args)
