import os
import sys
import numpy as np
import tensorflow as tf

class bottle_neck_graph():
    def __init__(self, dir_path, sess):
        self.graph = self.load_graph(dir_path)
        self.sess = tf.Session(graph=self.graph)
        self.print_graph()
        self.input = self.graph.get_tensor_by_name('inputs/word_2_vec_input:0')
        self.bottle_neck = self.graph.get_tensor_by_name('norm_look_up/norm_embed_look_up:0')

    def print_graph(self):
        print("bGraph layers")
        tensors = []
        with tf.Session(graph=self.graph) as _sess:
            op = _sess.graph.get_operations()
            tensors = [m.values() for m in op]
            for tensor in tensors:
                print(tensor)

    def get_bottle_neck_out(self, data):
        data = np.asarray(data)
        data = np.reshape(data, [-1])
        print(data)
        bneck = self.sess.run(self.bottle_neck, feed_dict={self.input: data})
        return bneck

    def load_graph(self, path):
        # Load protobuf file from the disk and retrive unserialized graph
        with tf.gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # import graph_def into a new graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        return graph


def main():
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    pg_model = os.path.join(current_path, 'pb_models/frozen_tGraph.pb')
    print(pg_model)
    bg = bottle_neck_graph(pg_model, None)
    print(bg.get_bottle_neck_out(0))
    print("sec")
    print(bg.get_bottle_neck_out([2, 0]))

if __name__ == "__main__":
    main()