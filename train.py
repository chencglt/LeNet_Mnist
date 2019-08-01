import os
import tensorflow as tf
from LeNet import *
import config as cfg
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import argparse

mnist = input_data.read_data_sets(cfg.DATASET_PATH, one_hot=True)

parser = argparse.ArgumentParser(description='Usage: \
                                    python quick_test [-s save_dir -w checkpoint_dir]')
parser.add_argument('-w', dest='ckpt_dir', required=False)
parser.add_argument('-s', dest='save_dir', required=False, default="checkpoints")
args = parser.parse_args()

def update_lr(step):
    return tf.train.exponential_decay(
        cfg.LEARNING_RATE_BASE,
        step, mnist.train.num_examples / cfg.BATCH_SIZE, cfg.LEARNING_RATE_DECAY,
        staircase=True)

def compute_loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean

def forward(input, label, training):
    preds = lenet_fn_layers(input, training)
    loss = compute_loss(preds, label)

    correct_items = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_items, 'float'))
    return loss, accuracy

def placeholders():
    x = tf.placeholder(tf.float32, [None, cfg.IMAGE_W, cfg.IMAGE_H, 1], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, cfg.LABEL_SIZE], name='y_label')
    is_training = tf.placeholder(tf.bool)
    return x, y_, is_training

def main_loop():
    test_accuracy = []
    global_step = tf.Variable(0, trainable=False)
    x, y_, is_training = placeholders()

    loss, accuracy = forward(x, y_, is_training)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops): 
        train_op = tf.train.GradientDescentOptimizer(update_lr(global_step)).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)
        tf.global_variables_initializer().run()
        if not args.ckpt_dir is None:
            saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_dir))
        for i in range(cfg.TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(cfg.BATCH_SIZE)
            xs = np.reshape(xs, [cfg.BATCH_SIZE, cfg.IMAGE_W, cfg.IMAGE_W, -1])
            _, loss_ = sess.run([train_op, loss], feed_dict={x: xs, y_: ys, is_training: True})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (i, loss_))
            if i % 1000 ==0:
                saver.save(sess, save_path=os.path.join(args.save_dir, "lenet.ckpt"), global_step=i)
                y_test_ = mnist.test.labels
                x_test = np.reshape(mnist.test.images, [len(mnist.test.images), cfg.IMAGE_H, cfg.IMAGE_W, -1])
                loss_, accuracy_ = sess.run([loss, accuracy], feed_dict={x: x_test, y_: y_test_, is_training: False})
                test_accuracy.append(accuracy_)
                print("Test Loss/Accuracy: %f/%f" % (loss_, accuracy_))
    plt.grid(True)
    plt.plot(test_accuracy)

if __name__=='__main__':
    main_loop()
