import os
import sys
import math
import collections
import tensorflow as tf
import numpy as np
import zipfile
import random
import json
from gensim import corpora
from gensim.parsing import preprocessing
from tensorboard.plugins import projector

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
log_path = os.path.join(current_path, 'log/word_to_vec')
model_path = os.path.join(current_path, 'saved_model/word_to_vec')
metadata_path = os.path.join(current_path, 'saved_model/word_to_vec/metadata.tsv')


data_source = os.path.join(current_path+"/data/", 'compressed')
w_id = os.path.join(current_path+"/data/", 'word_to_id_mapping.json')
id_w = os.path.join(current_path+"/data/", 'id_to_word_mapping.json')

data_index = 0

vocabulary_size = 17719
batch_size = 128
skip_window = 4
num_skips = 2
num_sampled = 4
embedding_size = 200
embeddings_shape = [vocabulary_size, embedding_size]
rand_sampled = 4

# Read the data into a list of strings.
def read_data(data_source_path):
    corpus = []
    for filename in os.listdir(data_source_path):
        if filename.endswith(".zip"):
            filename = os.path.join(data_source_path, filename)
            """Extract the first file enclosed in a zip file as a list of words."""
            with zipfile.ZipFile(filename) as f:
                data = preprocessing.remove_stopwords(f.read(f.namelist()[0]).lower())
                data = preprocessing.strip_multiple_whitespaces(data)
                data = preprocessing.strip_numeric(data)
                #data = preprocessing.split_alphanum(data)
                #data = f.read(f.namelist()[0])
                data = tf.compat.as_str(data).split()
                #data = preprocessing(data)
                corpus.append(data)
    return corpus


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    all_words = []
    for seg in words:
        all_words.extend(seg)
    count.extend(collections.Counter(all_words).most_common(n_words - 1))
    dictionary = {}
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    total_word_count = len(all_words) + 1
    word_probablities = []
    for w in count:
        word_probablities.append(w[1]/total_word_count)
    for word in all_words:
      index = dictionary.get(word, 0)
      if index == 0:  # dictionary['UNK']
        unk_count += 1
      data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print("vocab size : {}".format(len(dictionary)))
    print("storing word id mappings to disk...")
    f = open(w_id, 'w')
    f.write(json.dumps(dictionary))
    f.close()
    f = open(id_w, 'w')
    f.write(json.dumps(reversed_dictionary))
    f.close()
    return data, dictionary, reversed_dictionary, word_probablities

'''
def build_dataset(raw_data, n_words):
    """Process raw inputs into a dataset."""
    corp_dict = corpora.Dictionary(raw_data)
    dictionary = corp_dict.token2id
    print(dictionary)
    frequency = corp_dict.cfs
    for item in dictionary:
        dictionary[item] = frequency[dictionary[item]]
    raw_data = sorted(dictionary.keys(), key = lambda x: dictionary[x], reverse = True)
    corp_dict = corpora.Dictionary(raw_data)
    dictionary = corp_dict.token2id
    print(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    all_words = []
    for doc in raw_data:
        all_words.extend(doc)
    data = [dictionary[word] for word in all_words]
    word_probablities = [wr[1] for wr in corp_dict.doc2bow(all_words)]
    print("vocab size : {}".format(len(dictionary)))
    print("storing word id mappings to disk...")
    f = open(w_id, 'w')
    f.write(json.dumps(dictionary))
    f.close()
    f = open(id_w, 'w')
    f.write(json.dumps(reversed_dictionary))
    f.close()
    return data, dictionary, reversed_dictionary, word_probablities
'''

def generate_batch(batch_size, num_skips, skip_window, data):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # example : the cat sat on the
    # target_word -> sat
    # context words -> the, cat, on, the
    # skip_window -> 2, ==> i.e the cat sat, sat on the
    # num_skips -> 2, ==> number of words randomly sampled from context words
    #                 ==> (sat, cat), (sat, the) i.e two samples
    # span -> 2 * skip_window + 1 ==> if you start at 0, move index to 2 * skip_window + 1
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
      data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(int(batch_size / num_skips)):
      context_words = [w for w in range(span) if w != skip_window]
      words_to_use = random.sample(context_words, num_skips)
      for j, context_word in enumerate(words_to_use):
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[context_word]
      if data_index == len(data):
        buffer.extend(data[0:span])
        data_index = span
      else:
        buffer.append(data[data_index])
        data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

class word_to_vec():
    def __init__(self, graph, sess, word_probablities, name):
        self.name = name
        self.sess = sess
        self.graph = graph
        self.word_prob = word_probablities

        with self.graph.as_default():
            # Input data
            with tf.name_scope('inputs'):
                self.input = tf.placeholder(tf.int32, shape=[None], name=self.name+"input")
                #self.words_input = tf.placeholder(tf.int32, shape=[None], name=self.name + "words_input")
                self.target = tf.placeholder(tf.int32, shape=[batch_size, 1], name=self.name+"target")
                self.valid_dataset = tf.placeholder(tf.int32, shape=[None], name=self.name+"validate")

            with tf.name_scope('embeddings'):
                init_width = 0.5 / embedding_size
                self.embeddings = tf.Variable(tf.random_uniform(embeddings_shape, -init_width, init_width), name="word_embeddings")

            with tf.name_scope('look_up'):
                self.embed = tf.nn.embedding_lookup(self.embeddings, self.input, name="embed_look_up")
                self.mean_embed = tf.reduce_mean(self.embed, axis=0, name="mean_embed")

            with tf.name_scope('norm_look_up'):
                norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
                normalized_embeddings = self.embeddings / norm
                self.norm_embed = tf.nn.embedding_lookup(normalized_embeddings, self.input, name="norm_embed_look_up")
                self.mean_embed = tf.reduce_mean(self.norm_embed, axis=0, name="norm_mean_embed")


            with tf.name_scope('weights'):
                dev = 1.0 / math.sqrt(embedding_size)
                self.nce_weights = tf.Variable(tf.truncated_normal(embeddings_shape, stddev=dev), name="word_weights")

            with tf.name_scope('biases'):
                self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="word_biases")

            self.labels_matrix = tf.reshape(tf.cast(self.target, dtype=tf.int64), [batch_size, 1])

            with tf.name_scope('loss'):
                loss = tf.nn.nce_loss(weights=self.nce_weights,
                                      biases=self.nce_biases,
                                      labels=self.target,
                                      inputs=self.embed,
                                      num_sampled=num_sampled,
                                      num_classes=vocabulary_size)
                self.loss = tf.reduce_mean(loss)

            # Negative sampling.
            self.sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=self.labels_matrix,
                num_true=1,
                num_sampled=rand_sampled,
                unique=True,
                unigrams=self.word_prob,
                range_max=vocabulary_size))

            self.true_w = tf.nn.embedding_lookup(self.nce_weights, self.target)
            true_b = tf.nn.embedding_lookup(self.nce_biases, self.target)
            sampled_w = tf.nn.embedding_lookup(self.nce_weights, self.sampled_ids)
            sampled_b = tf.nn.embedding_lookup(self.nce_biases, self.sampled_ids)


            self.mul = tf.multiply(self.embed, self.true_w)

            self.true_logits = tf.reduce_sum(self.mul, 1) + true_b

            sampled_b_vec = tf.reshape(sampled_b, [rand_sampled])


            # distance between true words and random words
            sampled_logits = tf.matmul(self.embed, sampled_w, transpose_b=True) + sampled_b_vec

            self.nceloss = self.nce_loss(self.true_logits, sampled_logits)

            with tf.name_scope('optimizer'):
                opt = tf.train.GradientDescentOptimizer(0.1)
                #opt = tf.train.AdamOptimizer(0.1)
                self.train_step = opt.minimize(self.nceloss)

            # building summaries
            tf.summary.scalar('loss', self.nceloss)
            self.merged = tf.summary.merge_all()

            self.cos_sim, self.normalized_embeddings = self.cosine_similarity(self.valid_dataset)

    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / batch_size
        return nce_loss_tensor

    def cosine_similarity(self, valid_dataset):
        with tf.name_scope('cosine_sim'):
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            normalized_embeddings = self.embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        return similarity, normalized_embeddings

    # by default we work with normalized embedding
    def get_embedding(self, word_id):
        # example : word_id = dictionary["the"]
        feed_dict = {self.input: np.asarray([word_id])}
        return self.sess.run(self.normalized_embeddings, feed_dict=feed_dict)

    def train_batch(self, batch_inputs, batch_labels, word_probablities):


        #print("--")
        #print(batch_inputs)
        #print("--")
        #print(batch_labels)

        feed_dict = {self.input: batch_inputs, self.target: batch_labels}
        run_metadata = tf.RunMetadata()

        #print(self.sess.run(self.true_logits, feed_dict=feed_dict))


        ops = [self.train_step, self.nceloss, self.merged]
        _, loss_val, summary = self.sess.run(ops, feed_dict=feed_dict, run_metadata=run_metadata)
        return loss_val, summary

def main():

    vocabulary = read_data(data_source)
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        data, dictionary, reverse_dictionary, word_probablities = build_dataset(vocabulary, vocabulary_size)
        del vocabulary # delete the vocab to save memory

        # write words and ids to metadata for tensorboard
        with open(metadata_path, 'w') as f:
            for i in range(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        # init the model
        wvec = word_to_vec(graph, sess, word_probablities, "word_2_vec_")
        writer = tf.summary.FileWriter(log_path, sess.graph)
        saver = tf.train.Saver()
        # init the graph
        tf.global_variables_initializer().run()

        # load pre trained graph
        checkpoint = tf.train.get_checkpoint_state(model_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        # Sample data what is fed to the graph
        print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

        num_steps = 30000
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data)
            if step == 0:
                for i in range(batch_size):
                    print(batch_inputs[i], reverse_dictionary[batch_inputs[i]], '->', batch_labels[i, 0], reverse_dictionary[batch_labels[i, 0]])

            loss, summary = wvec.train_batch(batch_inputs, batch_labels, word_probablities)
            average_loss += loss

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            writer.add_summary(summary, step)

        saver.save(sess, os.path.join(model_path, 'model.ckpt'))
        print("model saved to {}".format(model_path))

        print("setting up tensorboard projector...")
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = wvec.embeddings.name
        embedding_config.metadata_path = metadata_path
        projector.visualize_embeddings(tf.summary.FileWriter(model_path), config)

    writer.close()

if __name__ == "__main__":
    main()
