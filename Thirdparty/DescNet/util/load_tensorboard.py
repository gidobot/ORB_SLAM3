import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile
import sys

if len(sys.argv) != 2:
    sys.exit('Usage: load_tensorboard.py path_to_pb_file')

with tf.Session() as sess:
    # model_filename ='reg.pb'
    # model_filename ='saved_model.pb'
    with gfile.FastGFile(sys.argv[1], 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
    LOGDIR='/tmp'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
input()
