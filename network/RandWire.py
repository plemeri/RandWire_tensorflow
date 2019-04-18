import tensorflow as tf
import networkx as nx
from utils import graph_generator as gg
import numpy as np

# gg.watts_strogats(32, 4, 0.75)

def conv_block(input, kernels, filters, strides, dropout_rate, training, scope):
    with tf.variable_scope(scope):
        input = tf.nn.relu(input)
        input = tf.layers.separable_conv2d(input, filters=filters, kernel_size=[kernels, kernels], strides=[strides, strides], padding='SAME')
        input = tf.layers.batch_normalization(input, training=training)
        input = tf.layers.dropout(input, rate=dropout_rate, training=training)
    return input

def build_stage(input, filters, dropout_rate, training, graph_data, scope):
    graph, graph_order, start_node, end_node = graph_data

    interms = {}
    with tf.variable_scope(scope):
        for node in graph_order:
            if node in start_node:
                interm = conv_block(input, 3, filters, 2, dropout_rate, training, scope='node' + str(node))
                interms[node] = interm
            else:
                in_node = list(nx.ancestors(graph, node))
                if len(in_node) > 1:
                    with tf.variable_scope('node' + str(node)):
                        weight = tf.get_variable('sum_weight', shape=len(in_node), dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
                        weight = tf.nn.sigmoid(weight)
                        interm = weight[0] * interms[in_node[0]]
                        for idx in range(1, len(in_node)):
                            interm += weight[idx] * interms[in_node[idx]]
                        interm = conv_block(interm, 3, filters, 1, dropout_rate, training, scope='conv_block' + str(node))
                        interms[node] = interm
                elif len(in_node) == 1:
                    interm = conv_block(interms[in_node[0]], 3, filters, 1, dropout_rate, training, scope='node' + str(node))
                    interms[node] = interm

        output = interms[end_node[0]]
        for idx in range(1, len(end_node)):
            output += interms[end_node[idx]]

        return output

def small_regime(input, stages, filters, classes, dropout_rate, graph_model, graph_param, graph_file_path, init_subsample, training):
    with tf.variable_scope('conv1'):
        if init_subsample is True:
            input = tf.layers.separable_conv2d(input, filters=int(filters/2), kernel_size=[3, 3], strides=[2, 2], padding='SAME')
            input = tf.layers.batch_normalization(input, training=training)
        else:
            input = tf.layers.separable_conv2d(input, filters=int(filters / 2), kernel_size=[3, 3], strides=[1, 1],
                                               padding='SAME')

    input = conv_block(input, 3, filters, 2, dropout_rate, training, 'conv2')

    for stage in range(3, stages+1):
        graph_data = gg.graph_generator(graph_model, graph_param, graph_file_path, 'conv' + str(stage) + '_' + graph_model)
        input = build_stage(input, filters, dropout_rate, training, graph_data, 'conv' + str(stage))
        filters *= 2

    with tf.variable_scope('classifier'):
        input = conv_block(input, 1, 1280, 1, dropout_rate, training, 'conv_block_classifier')
        input = tf.layers.average_pooling2d(input, pool_size=input.shape[1:3], strides=[1, 1])
        input = tf.layers.flatten(input)
        input = tf.layers.dense(input, units=classes)

    return input

def regular_regime(input, stages, filters, classes, dropout_rate, graph_model, graph_param, graph_file_path, training):
    with tf.variable_scope('conv1'):
        input = tf.layers.separable_conv2d(input, filters=int(filters/2), kernel_size=[3, 3], strides=[2, 2], padding='SAME')
        input = tf.layers.batch_normalization(input, training=training)

    for stage in range(2, stages+1):
        graph_data = gg.graph_generator(graph_model, graph_param, graph_file_path, 'conv' + str(stage) + '_' + graph_model)
        input = build_stage(input, filters, dropout_rate, training, graph_data, 'conv' + str(stage))
        filters *= 2
    
    with tf.variable_scope('classifier'):
        input = conv_block(input, 1, 1280, 1, dropout_rate, training, 'conv_block_classifier')
        input = tf.layers.average_pooling2d(input, pool_size=input.shape[1:3], strides=[1, 1])
        input = tf.layers.flatten(input)
        input = tf.layers.dense(input, units=classes)
        input = tf.layers.dropout(input, rate=dropout_rate)

    return input

def my_regime(input, stages, filters, classes, dropout_rate, graph_model, graph_param, graph_file_path, init_subsample, training): #regular regime 기반
    with tf.variable_scope('conv1'):
        if init_subsample is True:
            input = tf.layers.separable_conv2d(input, filters=int(filters/2), kernel_size=[3, 3], strides=[2, 2], padding='SAME')
            input = tf.layers.batch_normalization(input, training=training)
        else:
            input = tf.layers.separable_conv2d(input, filters=int(filters / 2), kernel_size=[3, 3], strides=[1, 1],
                                               padding='SAME')
            input = tf.layers.batch_normalization(input, training=training)

    for stage in range(2, stages+1):
        graph_data = gg.graph_generator(graph_model, graph_param, graph_file_path, 'conv' + str(stage) + '_' + graph_model)
        input = build_stage(input, filters, dropout_rate, training, graph_data, 'conv' + str(stage))
        filters *= 2

    with tf.variable_scope('classifier'):
        input = conv_block(input, 1, 1280, 1, dropout_rate, training, 'conv_block_classifier')
        input = tf.layers.average_pooling2d(input, pool_size=input.shape[1:3], strides=[1, 1])
        input = tf.layers.flatten(input)
        input = tf.layers.dropout(input, rate=0.3, training=training)
        input = tf.layers.dense(input, units=classes)

    return input

def my_small_regime(input, stages, filters, classes, dropout_rate, graph_model, graph_param, graph_file_path, init_subsample, training): #regular regime 기반
    with tf.variable_scope('conv1'):
        if init_subsample is True:
            input = tf.layers.separable_conv2d(input, filters=int(filters/2), kernel_size=[3, 3], strides=[2, 2], padding='SAME')
            input = tf.layers.batch_normalization(input, training=training)
        else:
            input = tf.layers.separable_conv2d(input, filters=int(filters / 2), kernel_size=[3, 3], strides=[1, 1],
                                               padding='SAME')
            input = tf.layers.batch_normalization(input, training=training)

    input = conv_block(input, 3, filters, 1, dropout_rate, training, 'conv2')

    for stage in range(3, stages+1):
        graph_data = gg.graph_generator(graph_model, graph_param, graph_file_path, 'conv' + str(stage) + '_' + graph_model)
        input = build_stage(input, filters, dropout_rate, training, graph_data, 'conv' + str(stage))
        filters *= 2

    with tf.variable_scope('classifier'):
        input = conv_block(input, 1, 1280, 1, dropout_rate, training, 'conv_block_classifier')
        input = tf.layers.average_pooling2d(input, pool_size=input.shape[1:3], strides=[1, 1])
        input = tf.layers.flatten(input)
        input = tf.layers.dropout(input, rate=0.3, training=training)
        input = tf.layers.dense(input, units=classes)

    return input
