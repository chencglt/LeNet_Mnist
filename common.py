import tensorflow as tf
import numpy as np

def initializer(shape, name = 'xavier'):
    with tf.variable_scope(name) as scope:
        stddev = 1.0 / tf.sqrt(float(shape[0]), name = 'stddev')
        inits = tf.truncated_normal(shape=shape, stddev=stddev, name = 'xavier_init')
    return inits

def softmax_layer (input, name = 'softmax'):
    with tf.variable_scope(name) as scope:        
        inference = tf.nn.softmax(input, name = 'inference')
        predictions = tf.argmax(inference, 1, name = 'predictions')
    return (inference, predictions)

def dot_product_layer(input, params = None, neurons = 1200, name = 'fc', activation = 'relu'):
    with tf.variable_scope(name) as scope:
        if params is None:
            weights = tf.Variable(initializer([input.shape[1].value,neurons], name = 'xavier_weights'),\
                                            name = 'weights')
            bias = tf.Variable(initializer([neurons], name = 'xavier_bias'), name = 'bias')
        else:
            weights = params[0]
            bias = params[1]

        dot = tf.nn.bias_add(tf.matmul(input, weights, name = 'dot'), bias, name = 'pre-activation')
        if activation == 'relu':
            activity = tf.nn.relu(dot, name = 'activity' )
        elif activation == 'sigmoid':
            activity = tf.nn.sigmoid(dot, name = 'activity' )            
        elif activation == 'identity':
            activity = dot                     
        params = [weights, bias]
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('bias', bias)
        tf.summary.histogram('activity', activity)
    return (activity, params)

def conv_2d_layer (input, 
                neurons = 20,
                filter_size = (5,5), 
                stride = (1,1,1,1), 
                padding = 'VALID',
                name = 'conv', 
                activation = 'relu',
                visualize = False):     
    f_shp = [filter_size[0], filter_size[1], input.shape[3].value, neurons]
    with tf.variable_scope(name) as scope:
        weights = tf.Variable(initializer(  f_shp, 
                                            name = 'xavier_weights'),\
                                            name = 'weights')
        bias = tf.Variable(initializer([neurons], name = 'xavier_bias'), name = 'bias')
        c_out = tf.nn.conv2d(   input = input,
                                filter = weights,
                                strides = stride,
                                padding = padding,
                                name = scope.name  )
        c_out_bias = tf.nn.bias_add(c_out, bias, name = 'pre-activation')
        if activation == 'relu':
            activity = tf.nn.relu(c_out_bias, name = 'activity' )
        elif activation == 'sigmoid':
            activity = tf.nn.sigmoid(c_out_bias, name = 'activity' )            
        elif activation == 'identity':
            activity = c_out_bias
        params = [weights, bias]
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('bias', bias)  
        tf.summary.histogram('activity', activity) 
        #if visualize is True:  
        #    visualize_filters(weights, name = 'filters_' + name)
    return (activity, params)        

def flatten_layer (input, name = 'flatten'):
    with tf.variable_scope(name) as scope:
        in_shp = input.get_shape().as_list()
        output = tf.reshape(input, [-1, in_shp[1]*in_shp[2]*in_shp[3]])
    return output 

def max_pool_2d_layer  (   input, 
                        pool_size = (1,2,2,1),
                        stride = (1,2,2,1),
                        padding = 'VALID',
                        name = 'pool' ):    
    with tf.variable_scope(name) as scope:
        output = tf.nn.max_pool (   value = input,
                                    ksize = pool_size,
                                    strides = stride,
                                    padding = padding,
                                    name = name ) 
    return output