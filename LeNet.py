import tensorflow as tf
import config as cfg

slim = tf.contrib.slim

class LeNet:
    def network_fn(self, input, is_training):
        reuse = False if is_training else True
        net = tf.image.resize_images(input, [cfg.LENET_INPUT_W, cfg.LENET_INPUT_H], method=tf.image.ResizeMethod.BILINEAR)
        with slim.arg_scope([slim.conv2d], stride=1, normalizer_fn=slim.batch_norm, padding='VALID'):
            with tf.variable_scope('lenet', reuse=reuse):
                with tf.variable_scope('conv1'):
                    net = slim.conv2d(net, 6, [5, 5])
                    net = slim.avg_pool2d(net, 2, 2)
                with tf.variable_scope('conv2'):
                    net = slim.conv2d(net, 16, [5, 5])
                    net = slim.avg_pool2d(net, 2, 2)
                with tf.variable_scope('flatten'):
                    net = slim.flatten(net, scope='flat6')
                with tf.variable_scope('full_conn'):
                    net = slim.fully_connected(net, 120)
                    net = slim.fully_connected(net, 84)
                    #net = slim.dropout(net, DROPOUT_RATIO, is_training=is_training, scope='dropout8')
                    net = slim.fully_connected(net, cfg.LABEL_SIZE)
                return net