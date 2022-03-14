import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops.rnn_cell import ResidualWrapper

from tensorflow.python.layers.core import Dense

import math
import CorpusProcessor as sp

class TripletEncoder(object):

    def __init__(self, params):

        # Encoder parameters configuration
        self.dtype = tf.float32

        self.hidden_units_num = params["hidden_units_num"]
        self.deep_layers_num = params["deep_layers_num"]

        self.enable_residual_wrapper = params["enable_residual_wrapper"]

        self.enable_dropout_wrapper = params["enable_dropout_wrapper"]
        self.input_keep_prob = params["input_keep_prob"]
        self.output_keep_prob = params["output_keep_prob"]

        self.vocabulary_size = params["vocabulary_size"]
        self.embedding_size = params["embedding_size"]
        self.glove_weights_initializer = params["glove_weights_initializer"]
        # -----------------------------------

        self.initialize_data_placeholders()


    # Set the encoder's input placeholders ----------------------------------------------
    def initialize_data_placeholders(self):
        self.questions_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='questions_inputs')
        self.questions_inputs_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='questions_inputs_len')

        self.answers_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='answers_inputs')
        self.answers_inputs_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='answers_inputs_len')

        self.negatives_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='negatives_inputs')
        self.negatives_inputs_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='negatives_inputs_len')

    def get_LSTM_cell(self):
        # Creates a full LSTM cell of a fixed hidden units number with default 'tanh' activation function
        LSTM_cell = LSTMCell(self.hidden_units_num)

        # Skip connections for passing layers and avoid gradient explosion or vanish
        if self.enable_residual_wrapper:
            LSTM_cell = ResidualWrapper(LSTM_cell)

        # Use dropout for improving model performance and reducing overfitting chance
        if self.enable_dropout_wrapper:
            LSTM_cell = DropoutWrapper(LSTM_cell, dtype=self.dtype,
                                       input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)

        return LSTM_cell

    def get_deep_LSTM_cell(self):
        return MultiRNNCell([self.get_LSTM_cell() for _ in range(self.deep_layers_num)])

    def create_custom_LSTM_encoder(self):

        with tf.variable_scope('encoder'):
            # Create a dense layer that normalizes the number of words in a sentence to the hidden layers number
            input_proj_layer = Dense(units=self.hidden_units_num, dtype=self.dtype, name='input_projection_layer')

            # Retrieves a deep LSTM network
            self.deep_cell = self.get_deep_LSTM_cell()

            # Create an embedding, embedd the input and project it
            self.embeddings = tf.get_variable(name='embedding_weights',
                                              shape=[self.vocabulary_size, self.embedding_size],
                                              initializer=self.glove_weights_initializer,
                                              dtype=self.dtype, trainable=True)

            embedded_questions = tf.nn.embedding_lookup(params=self.embeddings, ids=self.questions_inputs)
            self.proj_em_questions = input_proj_layer(embedded_questions)

            embedded_answers = tf.nn.embedding_lookup(params=self.embeddings, ids=self.answers_inputs)
            self.proj_em_answers = input_proj_layer(embedded_answers)

            embedded_negatives = tf.nn.embedding_lookup(params=self.embeddings, ids=self.negatives_inputs)
            self.proj_em_negatives = input_proj_layer(embedded_negatives)



            # Create a custom dynamic LSTM with the embedded and normalized inputs
            self.questions_outputs, self.questions_state = tf.nn.dynamic_rnn(cell=self.deep_cell,
                                                                             inputs=self.proj_em_questions,
                                                                             sequence_length=self.questions_inputs_len,
                                                                             dtype=self.dtype,
                                                                             time_major=False)

            self.answers_outputs, self.answers_state = tf.nn.dynamic_rnn(cell=self.deep_cell,
                                                                            inputs=self.proj_em_answers,
                                                                            sequence_length=self.answers_inputs_len,
                                                                            dtype=self.dtype,
                                                                            time_major=False)

            self.negatives_outputs, self.negatives_state = tf.nn.dynamic_rnn(cell=self.deep_cell,
                                                                            inputs=self.proj_em_negatives,
                                                                            sequence_length=self.negatives_inputs_len,
                                                                            dtype=self.dtype,
                                                                            time_major=False)



