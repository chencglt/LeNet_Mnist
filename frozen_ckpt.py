import tensorflow as tf
from LeNet import *
import argparse

parser = argparse.ArgumentParser(description='Usage: \
                                    python quick_test -w [checkpoint_dir]')
parser.add_argument('-w', dest='ckpt_dir', required=True)
args = parser.parse_args()

def freeze_graph(sess, output_file, output_node_names):
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        output_node_names,
    )
    with tf.gfile.GFile(output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("=> {} ops written to {}.".format(len(output_graph_def.node), output_file))

def main():

    sess = tf.Session()
    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        input_x = tf.placeholder(tf.float32, [1, 28, 28, 1])
        result = lenet_fn_layers(input_x, is_training=False)
        print("=>", result.name[:-2])
        out_node = [result.name[:-2]]
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_dir))
        freeze_graph(sess, './lenet.pb', out_node)

if __name__=="__main__":
    main()