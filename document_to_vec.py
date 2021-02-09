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

    def get_document_vector(self, word_ids):
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


def get_document_vector(bg, wrd_lst):
    # load the json file ( doc_id : blob of text )
    # example raw_data = ["I", "am", "getting", "error"]
    raw_data = wrd_lst#["I", "am", "getting", "error"]
    # for each word in the blob retrieve the word id
    word_ids = list()
    with open(w_id) as json_file:
        data = json.load(json_file)
        for word in raw_data:
            if word in data:
                id = data[word]
            else:
                id = 0
            word_ids.append(id)
    doc_vec = bg.get_document_vector(word_ids)
    return doc_vec


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

    print(pg_model)
    bg = bottle_neck_graph(pg_model, None)

    # for all documents compute doc id
    keys, sample = load_data(raw_data)

    query_sample = "Having issues with SDB start blt --project --sdb-go"
    query_sample = query_sample.split()
    sample.append(query_sample)

    total_documents = len(sample)

    # write words and ids to metadata for tensorboard
    with open(metadata_path, 'w') as f:
        for key in keys:
            f.write("doc_id_"+str(key) + '\n')
        f.write("doc_id_query" + '\n')

    # collect all document vectors
    embeds = list()
    for id, sam in enumerate(sample):
        doc_vec = get_document_vector(bg, sam)
        embeds.append(doc_vec)
    embeds = np.asarray(embeds)
    embeds = np.reshape(embeds, [total_documents, embedding_size])
    print(embeds.shape)
    # save it to a json file
    # {doc_id: doc_vec, .. }

    # setup tensorbord for visualizing
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        input = tf.placeholder(tf.int32, shape=[None], name="input")
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(initial_value=embeds, name="doc_embeddings")
            #embed = tf.nn.embedding_lookup(embeddings, input, name="embed_look_up")

        writer = tf.summary.FileWriter(log_path, sess.graph)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.save(sess, ckptdata_path)

        print("setting up tensorboard projector...")
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embeddings.name
        embedding_config.metadata_path = metadata_path
        projector.visualize_embeddings(tf.summary.FileWriter(model_path), config)

if __name__ == "__main__":
    main()