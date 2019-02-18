import tensorflow as tf
from LeNet import LeNet
import config as cfg
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets(cfg.DATASET_PATH, one_hot=True)
slim = tf.contrib.slim

def inference(input_x, input_y):
    x = input_x
    y_ = input_y
    global_step = tf.Variable(0, trainable=False)

    #regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    le_network = LeNet()
    result_train = le_network.network_fn(x, is_training=True)
    result_test = le_network.network_fn(x, is_training=False)

    #correct_train = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_test = tf.equal(tf.argmax(result_test, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_test, "float"))

    # loss, learning rate, moving_average
    #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=result_train, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean # + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        cfg.LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / cfg.BATCH_SIZE, cfg.LEARNING_RATE_DECAY,
        staircase=True)

    # update_ops & tf.control_dependencies are needed when introducing batch_norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
    with tf.control_dependencies(update_ops): 
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        train_op = train_step

    return train_op, loss, global_step, accuracy

def main_loop():
    test_accuracy = []
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [None, cfg.IMAGE_W, cfg.IMAGE_H, 1], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, cfg.LABEL_SIZE], name='y_label')
        train_op, loss, global_step, infer_accuracy = inference(x, y_)
        tf.global_variables_initializer().run()
        for i in range(cfg.TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(cfg.BATCH_SIZE)
            xs = np.reshape(xs, [cfg.BATCH_SIZE, cfg.IMAGE_W, cfg.IMAGE_W, -1])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            if i % 1000 ==0:
                y_test_ = mnist.test.labels
                x_test = np.reshape(mnist.test.images, [len(mnist.test.images), cfg.IMAGE_W, cfg.IMAGE_W, -1])
                accuracy = sess.run(infer_accuracy, feed_dict={x: x_test, y_: y_test_})
                test_accuracy.append(accuracy)
                print("Test accuracy: %f" % (accuracy))
    plt.grid(True)  
    plt.plot(test_accuracy)

if __name__=='__main__':
    main_loop()