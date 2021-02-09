import sys
import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
dir = os.path.dirname(os.path.realpath(__file__))
pb_out_path = os.path.join(current_path, 'pb_models')


def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = pb_out_path + "/frozen_tGraph.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        gh = tf.get_default_graph().as_graph_def()
        tensors = []
        op = sess.graph.get_operations()
        tensors = [m.values() for m in op]
        for tensor in tensors:
            print(tensor)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


if __name__ == '__main__':
    # checkpoint directory.
    model_dir = dir + "/saved_model/word_to_vec"

    # comma seperated output node names.
    output_node_names = "look_up/embed_look_up,look_up/mean_embed,norm_look_up/norm_embed_look_up,norm_look_up/norm_mean_embed"

freeze_graph(model_dir, output_node_names)