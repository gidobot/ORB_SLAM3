import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import sys

export_dir = sys.argv[1]
graph_pb = sys.argv[2]

INPUT_TENSOR = "input:0"
OUTPUT_TENSOR = "res5c:0"
SIGNATURE = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)

with tf.compat.v1.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    g = tf.compat.v1.get_default_graph()
    inp = g.get_tensor_by_name(INPUT_TENSOR)
    out = g.get_tensor_by_name(OUTPUT_TENSOR)

    sigs[SIGNATURE] = \
        tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
            {"input": inp}, {"output": out})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()