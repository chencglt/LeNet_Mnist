import os
import cv2
import argparse
import numpy as np
import config as cfg
import tensorflow as tf

from LeNet import lenet_fn_layers
test_dir = "images"
test_images = [os.path.join(test_dir, image) for image in os.listdir(test_dir)]

parser = argparse.ArgumentParser(description='Usage: \
                                    python quick_test [-s save_dir -w checkpoint_dir]')
parser.add_argument('-w', dest='ckpt_dir', required=False)
parser.add_argument('-s', dest='save_dir', required=False, default="checkpoints")
args = parser.parse_args()

def test(input_x):
    result = lenet_fn_layers(input_x, is_training=False)
    return result, tf.argmax(result, 1)
    
def _main():
    tf.reset_default_graph()
    input_x = tf.placeholder(tf.float32, [1, cfg.IMAGE_W, cfg.IMAGE_H, 1])
    result, arg_res = test(input_x)
    with tf.Session() as sess:    
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint("checkpoints"))  
        for image in test_images:
            img = cv2.resize(cv2.imread(image, cv2.IMREAD_GRAYSCALE), (28, 28), interpolation=cv2.INTER_NEAREST)/255.0
            img = np.reshape(img, (28, 28, 1))
            res, pred = sess.run([result, arg_res], feed_dict={input_x: [img]})
            print(res, pred)

if __name__=="__main__":
    _main()