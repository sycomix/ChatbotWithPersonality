import numpy as np
import tensorflow as tf
import math
import gc

NUM_TOKENS = 3

_SOS = "<_SOS>"
_EOS = "<_EOS>"
_UNK = "<_UNK>"

SOS_TOKEN = 0   # Start of sentence token
EOS_TOKEN = 1   # End of sentence token and padding
UNK_TOKEN = 2   # Unknown word token

class EmbeddingsManager(object):
    def __init__(self, embedding_file, embedding_size, vocabulary_size_limit=None,
                vocabulary_restrain=None, sorted_vocabulary_restrain=None):
        self.embedding_file = embedding_file
        self.embedding_size = embedding_size
        self.vocabulary_size_limit = vocabulary_size_limit
        self.vocabulary_restrain = vocabulary_restrain
        self.sorted_vocabulary_restrain = sorted_vocabulary_restrain

    def load_embeddings(self):
        word2index = {_SOS: SOS_TOKEN, _EOS: EOS_TOKEN, _UNK: UNK_TOKEN}
        index2word = {SOS_TOKEN: _SOS, EOS_TOKEN: _EOS, UNK_TOKEN: _UNK}
        weights = []

        # SOS, EOS, UNK
        weights.append(np.random.randn(self.embedding_size))
        weights.append(np.random.randn(self.embedding_size))
        weights.append(np.random.randn(self.embedding_size))

        index = NUM_TOKENS
        with open(self.embedding_file, 'r', encoding='utf8') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                # Vocabulary size limit
                if self.vocabulary_size_limit and index == self.vocabulary_size_limit:
                    break

                elements = line.split()
                word = elements[0]

                # Get the word2vec representation for just a set of given words
                if self.vocabulary_restrain and word not in self.vocabulary_restrain:
                    continue

                word_weights = np.asarray(elements[1:], dtype=np.float32)
                word2index[word] = index
                index2word[index] = word
                weights.append(word_weights)
                index += 1


        # Add the words that were not found in the pre-trained embeddings
        if self.vocabulary_restrain:
            for w in self.sorted_vocabulary_restrain:
                if self.vocabulary_size_limit and index == self.vocabulary_size_limit:
                    break

                if w not in word2index:
                    word2index[w] = index
                    index2word[index] = w
                    weights.append(np.random.randn(self.embedding_size))
                    index += 1

        weights = np.asarray(weights, dtype=np.float32)

        self.vocabulary_size = weights.shape[0]
        self.word2index = word2index
        self.index2word = index2word
        self.weights = weights
        self.glove_weights_initializer = tf.constant_initializer(self.weights)

        gc.collect()

    def get_word_index(self, word):
        if word in self.word2index:
            return self.word2index[word]

        return self.word2index[_UNK]

    def get_index_word(self, index):
        return self.index2word[index]

    def sentence_to_word_indexes(self, sentence):
        lst = []
        for word in sentence:
            cnt = self.get_word_index(word)
            lst.append(cnt)

        return lst

    def index_sentence_to_words(self, sentence):
        return [self.get_index_word(index) for index in sentence]

    def embeddings_lookup(self, data_type):
        return tf.get_variable(name='embedding_weights',
                               shape=[self.vocabulary_size, self.embedding_size],
                               initializer=self.glove_weights_initializer,
                               dtype=data_type, trainable=False)