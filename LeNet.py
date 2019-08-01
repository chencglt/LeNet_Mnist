import tensorflow as tf
import config as cfg
import common

## tf.contrib.slim and tf.contrib.layers are just the same
slim = tf.contrib.slim
c_layers = tf.contrib.layers

def lenet_fn_slim(self, input, is_training):
    with slim.arg_scope([slim.conv2d], stride=1, normalizer_fn=slim.batch_norm, padding='VALID'):
        with tf.variable_scope('lenet', reuse=tf.AUTO_REUSE):
            net = tf.identity(input)
            with tf.variable_scope('conv1'):
                net = slim.conv2d(net, 6, [5, 5])
                net = slim.avg_pool2d(net, 2, 2)
            with tf.variable_scope('conv2'):
                net = slim.conv2d(net, 16, [5, 5])
                net = slim.avg_pool2d(net, 2, 2)
            with tf.variable_scope('flatten'):
                net = slim.flatten(net)
            with tf.variable_scope('fc1'):
                net = slim.fully_connected(net, 120, activation_fn = tf.nn.relu)
            with tf.variable_scope('fc2'):
                net = slim.fully_connected(net, 84, activation_fn = tf.nn.relu)
                #net = slim.dropout(net, DROPOUT_RATIO, is_training=is_training, scope='dropout8')
            with tf.variable_scope('fc3'):
                net = slim.fully_connected(net, cfg.LABEL_SIZE)
            return net

def lenet_fn_contrib_layers(input):
    with slim.arg_scope([c_layers.conv2d], stride=1, normalizer_fn=None, padding='VALID'):
        with tf.variable_scope("lenet", reuse=tf.AUTO_REUSE):
            net = tf.identity(input)
            with tf.variable_scope('conv1'):
                net = c_layers.conv2d(net, 6, [5, 5])
                net = c_layers.max_pool2d(net, 2, 2)
            with tf.variable_scope('conv2'):
                net = c_layers.conv2d(net, 16, [5, 5])
                net = c_layers.max_pool2d(net, 2, 2)
            with tf.variable_scope('flatten'):
                pool_shape = net.get_shape().as_list()
                nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
                net = tf.reshape(net, [-1, nodes])
            with tf.variable_scope('fc1'):
                net = c_layers.fully_connected(net, 120)
            with tf.variable_scope('fc2'):
                net = c_layers.fully_connected(net, 84)
            with tf.variable_scope('fc3'):
                net = c_layers.fully_connected(net, cfg.LABEL_SIZE)
        return net

'''
With self-defined convolution & full_connected api based on tf.nn
'''
def lenet_fn_custom(input):
    with tf.variable_scope('lenet', reuse=tf.AUTO_REUSE):
        net = tf.identity(input)
        net, _ =  common.conv_2d_layer (input = net, neurons = 20, filter_size = (5, 5), name = 'conv1')
        net = common.max_pool_2d_layer ( input = net, name = 'pool1')
        net, _ =  common.conv_2d_layer (input = net, neurons = 20, filter_size = (5, 5), name = 'conv2')
        net = common.max_pool_2d_layer ( input = net, name = 'pool2')
        
        net = common.flatten_layer(net, "flattern")
        net, _ = common.dot_product_layer(net, neurons=120, name="fc1")
        net, _ = common.dot_product_layer(net, neurons=84, name="fc2")
        net, _ = common.dot_product_layer(net, neurons=cfg.LABEL_SIZE, name="fc3")
    return net

def lenet_fn_layers(input, is_training=False):
    with tf.variable_scope("lenet", reuse=tf.AUTO_REUSE):
        net = tf.identity(input)
        with tf.variable_scope('conv1'):
            net = tf.layers.conv2d(inputs=net, filters=6, kernel_size=[5, 5], padding="same", activation=None)
            net = tf.layers.batch_normalization(inputs=net, training = is_training)
            net = tf.nn.relu(net)
        with tf.variable_scope('pool1'):
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        with tf.variable_scope('conv2'):
            net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=[5, 5], padding="same", activation=None)
            net = tf.layers.batch_normalization(inputs=net, training = is_training)
            net = tf.nn.relu(net)
        with tf.variable_scope('pool2'):
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        with tf.variable_scope('flattern'):
            pool_shape = net.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            net = tf.reshape(net, [-1, nodes])
        with tf.variable_scope('fc1'):
            net = tf.layers.dense(inputs=net, units=120, activation=tf.nn.relu)
        with tf.variable_scope('fc2'):
            net = tf.layers.dense(inputs=net, units=80, activation=tf.nn.relu)
        with tf.variable_scope('fc3'):
            net = tf.layers.dense(inputs=net, units=cfg.LABEL_SIZE, activation=tf.nn.relu)
        return net

'''
Contruct network using tf.nn
'''
def lenet_fn_tfnn(input, train=False):#, regularizer):
    with tf.variable_scope("lenet", reuse=tf.AUTO_REUSE):
        net = tf.identity(input)
        with tf.variable_scope('conv1'):
            conv1_weights = tf.get_variable("weight", [5, 5, 1, 6],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("bias", [6], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(net, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        with tf.name_scope("pool1"):
            pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")
        with tf.variable_scope("conv2"):
            conv2_weights = tf.get_variable("weight", [5, 5, 6, 16],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        with tf.name_scope("pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('flatten'):
            pool_shape = pool2.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(pool2, [-1, nodes])
        with tf.variable_scope('fc1'):
            fc1_weights = tf.get_variable("weight", [nodes, 120],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc1_biases = tf.get_variable("bias", [120], initializer=tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        with tf.variable_scope('fc2'):
            fc2_weights = tf.get_variable("weight", [120, 80],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc2_biases = tf.get_variable("bias", [80], initializer=tf.constant_initializer(0.1))
            fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)       
        with tf.variable_scope('fc3'):
            fc3_weights = tf.get_variable("weight", [80, cfg.LABEL_SIZE],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc3_biases = tf.get_variable("bias", [cfg.LABEL_SIZE], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc2, fc3_weights) + fc3_biases
        return logit

