import os
import json
import sys
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
raw_data = os.path.join(current_path, 'data/chatter.json')
log_path = os.path.join(current_path, 'log/doc_to_vec')
pg_model = os.path.join(current_path, 'pb_models/frozen_tGraph.pb')
w_id = os.path.join(current_path+"/data/", 'word_to_id_mapping.json')
model_path = os.path.join(current_path, 'saved_model/doc_to_vec')
metadata_path = os.path.join(current_path, 'saved_model/doc_to_vec/metadata.tsv')
ckptdata_path = os.path.join(current_path, 'saved_model/doc_to_vec/doc_vec.ckpt')

embedding_size = 200

class bottle_neck_graph():
    def __init__(self, dir_path, sess):
        self.graph = self.load_graph(dir_path)
        self.sess = tf.Session(graph=self.graph)
        self.print_graph()
        self.input = self.graph.get_tensor_by_name('inputs/word_2_vec_input:0')
        self.bottle_neck = self.graph.get_tensor_by_name('norm_look_up/norm_embed_look_up:0')
        self.average_vectors = self.graph.get_tensor_by_name('norm_look_up/norm_mean_embed:0')

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

    def get_average_vector(self, word_ids):
        data = np.asarray(word_ids)
        data = np.reshape(data, [-1])
        doc_vec = self.sess.run(self.average_vectors, feed_dict={self.input: data})
        return doc_vec

    def load_graph(self, path):
        # Load protobuf file from the disk and retrive unserialized graph
        with tf.gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # import graph_def into a new graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        return graph

def get_document_vector(bg, raw_data, data):
    # load the json file ( doc_id : blob of text )
    # example raw_data = ["I", "am", "getting", "error"]
    # for each word in the blob retrieve the word id
    word_ids = list()
    for word in raw_data:
        id = data[word] if word in data else 0
        word_ids.append(id)
    #print(word_ids) -> take a look here later, to make sure id's are translated correctly
    doc_vec = bg.get_average_vector(word_ids)
    return doc_vec


def get_query_vector(bg, query, data):
    q_words = query.split()
    word_ids = list()
    # convert word to word ids
    for word in q_words:
        id = data[word] if word in data else 0
        word_ids.append(id)
    q_vec = bg.get_average_vector(word_ids)
    return q_vec


def load_data(file_path):
    print(file_path)
    with open(file_path, 'r+') as f:
        data = json.load(f)
    print(data.keys())
    raw_dat = []
    for key in data.keys():
        #print(key, data[key])
        raw_dat.append(data[key])
    return data.keys(), raw_dat


def main():
    bg = bottle_neck_graph(pg_model, None)
    # for all documents compute doc id
    keys, sample = load_data(raw_data)
    total_documents = len(sample)

    # load the word id mapping
    with open(w_id) as json_file:
        data = json.load(json_file)

    # collect all document vectors
    embeds = list()
    for id, sam in enumerate(sample):
        doc_vec = get_document_vector(bg, sam, data)
        embeds.append(doc_vec)
    embeds = np.asarray(embeds)
    embeds = np.reshape(embeds, [total_documents, embedding_size])

    # convert query into vector
    query = "install-blt doesnt work on ubuntu"

    q_vector = get_query_vector(bg, query, data)
    q_vector = np.reshape(q_vector, [1, embedding_size])

    print(embeds.shape)
    print(q_vector)

    # euclidean distance
    diff = np.sum(np.sqrt(np.power(q_vector - embeds, 2)), 1)
    srt = np.argsort(diff)[:5] # sort and pick closest 5 document ids
    print(srt)


if __name__ == "__main__":
    main()
