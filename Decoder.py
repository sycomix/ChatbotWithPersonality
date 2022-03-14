import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops.rnn_cell import ResidualWrapper

from tensorflow.python.layers.core import Dense

from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

import math
import CorpusProcessor as sp
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops

SOS_TOKEN = 0   # Start of sentence token
EOS_TOKEN = 1   # End of sentence token and padding
UNK_TOKEN = 2   # Unknown word token

class Decoder(object):

    def __init__(self, params, encoder, model_type):

        self.encoder = encoder
        self.model_type = model_type

        # Decoder parameters configuration
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

        self.attention_method = params["attention_method"]
        self.beam_width = params["beam_width"]

        self.learning_rate = params["learning_rate"]
        self.optimizer_type = params["optimizer_type"]
        self.gradient_clipping_norm = params["gradient_clipping_norm"]

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.max_decode_step = params["max_decode_step"]

        self.initialize_data_placeholders()


    # Set the decoder's placeholders ----------------------------------------------
    def initialize_data_placeholders(self):
        self.inputs_batch_size = tf.shape(self.encoder.inputs)[0]

        if self.model_type == "model_predict":
            return

        self.inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='dec_inputs')
        self.inputs_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='dec_inputs_len')

        # Column list of the sentence padded starting word
        start_word = tf.ones([self.inputs_batch_size, 1], tf.int32)
        start_word *= SOS_TOKEN

        # Column list of the sentence padded ending word
        end_word = tf.ones([self.inputs_batch_size, 1], tf.int32)
        end_word *= EOS_TOKEN

        # decoder_inputs_train: [batch_size , max_time_steps + 1]
        # insert _GO symbol in front of each decoder input
        self.train_inputs = tf.concat([start_word, self.inputs], axis=1)

        # decoder_inputs_length_train: [batch_size]
        self.train_inputs_len = self.inputs_len + 1

        # decoder_targets_train: [batch_size, max_time_steps + 1]
        # insert EOS symbol at the end of each decoder input
        self.train_targets = tf.concat([self.inputs, end_word], axis=1)

    def get_LSTM_cell(self):
        # Creates a full LSTM cell of a fixed hidden units number
        LSTM_cell = LSTMCell(self.hidden_units_num)

        # Use dropout for improving model performance and reducing overfitting chance
        if self.enable_dropout_wrapper:
            LSTM_cell = DropoutWrapper(LSTM_cell, dtype=self.dtype,
                                       input_keep_prob=self.input_keep_prob, output_keep_prob=self.output_keep_prob)

        # Skip connections for passing layers and avoid gradient explosion or vanish
        if self.enable_residual_wrapper:
            LSTM_cell = ResidualWrapper(LSTM_cell)

        return LSTM_cell

    def get_deep_LSTM_cell_list(self):
        return [self.get_LSTM_cell() for _ in range(self.deep_layers_num)]

    def tile_data_for_beamsearch(self):
        # DE MODIFICAT AICI DACA NU MERGE CEVA
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(self.encoder.outputs, multiplier=self.beam_width)
        tiled_encoder_final_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_width), self.encoder.state)
        tiled_encoder_inputs_len = tf.contrib.seq2seq.tile_batch(self.encoder.inputs_len,
                                                                 multiplier=self.beam_width)

        return tiled_encoder_outputs, tiled_encoder_final_state, tiled_encoder_inputs_len


    def get_attention_mechanism(self, tiled_encoder_outputs, tiled_encoder_inputs_len):
        attention = None

        if self.attention_method == 'luong':
            attention = attention_wrapper.LuongAttention(num_units=self.hidden_units_num, memory=tiled_encoder_outputs,
                                                         memory_sequence_length=tiled_encoder_inputs_len, )

        if self.attention_method == "bahdanau":
            attention = attention_wrapper.BahdanauAttention(num_units=self.hidden_units_num, memory=tiled_encoder_outputs,
                                                        memory_sequence_length=tiled_encoder_inputs_len, )
        return attention


    def get_deep_attention_LSTM_cell(self):

        # Obtain a deep LSTM RNN for the decoder
        self.deep_cell = self.get_deep_LSTM_cell_list()

        tiled_encoder_outputs = self.encoder.outputs
        tiled_encoder_final_state = self.encoder.state
        tiled_encoder_inputs_len = self.encoder.inputs_len

        # Tile the data for beamsearch
        if self.model_type == "predict_model":
            tiled_encoder_outputs, tiled_encoder_final_state, tiled_encoder_inputs_len = self.tile_data_for_beamsearch()

        # Create an attention mechanism and assign it to the last deep layer of the decoder
        self.attention_mechanism = self.get_attention_mechanism(tiled_encoder_outputs, tiled_encoder_inputs_len)

        def attn_decoder_input_fn(inputs, attention):
            _input_layer = Dense(self.hidden_units_num, dtype=self.dtype,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        self.deep_cell[-1] = attention_wrapper.AttentionWrapper(cell=self.deep_cell[-1],
                                                                attention_mechanism=self.attention_mechanism,
                                                                attention_layer_size=self.hidden_units_num,
                                                                cell_input_fn=attn_decoder_input_fn,
                                                                initial_cell_state=tiled_encoder_final_state[-1],
                                                                alignment_history=False,
                                                                name='attention_wrapper')

        # Convert the last state of the encoder to a beamsearch + attention state of the decoder
        decoder_initial_state = [state for state in tiled_encoder_final_state]

        attention_batch_size = self.inputs_batch_size
        if self.model_type != "train_model":
            attention_batch_size *= self.beam_width

        decoder_initial_state[-1] = self.deep_cell[-1].zero_state(dtype=self.dtype,
                                                                  batch_size=attention_batch_size)

        # Done using the last deep layer
        self.state = tuple(decoder_initial_state)
        self.deep_cell = MultiRNNCell(self.deep_cell)

    def get_RNN_optimizer(self):
        if self.optimizer_type == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(
                                                    learning_rate=self.learning_rate,
                                                    beta1=0.9,
                                                    beta2=0.999,
                                                    epsilon=1e-08
                                                    )
        elif self.optimizer_type == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        trainable_variables = tf.trainable_variables()
        variables_gradients = tf.gradients(self.loss_function, trainable_variables)

        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(variables_gradients, self.gradient_clipping_norm)

        # Update the model
        self.gradient_updates = self.optimizer.apply_gradients(zip(clip_gradients, trainable_variables),
                                                               global_step=self.global_step)

    def get_train_model(self, input_proj_layer, output_proj_layer):


        embedded_inputs = tf.nn.embedding_lookup(params=self.embeddings, ids=self.train_inputs)
        self.projected_embedded_inputs = input_proj_layer(embedded_inputs)

        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.projected_embedded_inputs,
                                                            sequence_length=self.train_inputs_len,
                                                            time_major=False,
                                                            name='training_helper')

        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.deep_cell,
                                                           helper=training_helper,
                                                           initial_state=self.state,
                                                           output_layer=output_proj_layer)

        # Maximum decoder time_steps in current batch
        max_decoder_length = tf.reduce_max(self.train_inputs_len)

        # decoder_outputs_train: BasicDecoderOutput
        #                        namedtuple(rnn_outputs, sample_id)
        # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
        #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
        # decoder_outputs_train.sample_id: [batch_size], tf.int32
        (self.train_outputs, self.train_state, self.train_outputs_len) = (tf.contrib.seq2seq.dynamic_decode(
                                                                            decoder=training_decoder,
                                                                            output_time_major=False,
                                                                            impute_finished=True,
                                                                            maximum_iterations=max_decoder_length))

        # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
        self.train_logits = tf.identity(self.train_outputs.rnn_output)

        # Use argmax to extract decoder symbols to emit
        self.train_prediction = tf.argmax(self.train_logits, axis=-1, name='dec_pred_train')

        # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
        masks = tf.sequence_mask(lengths=self.train_inputs_len,
                                 maxlen=max_decoder_length, dtype=self.dtype, name='masks')

        # Computes per word average cross-entropy over a batch
        # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
        self.loss_function = tf.contrib.seq2seq.sequence_loss(logits=self.train_logits,
                                                              targets=self.train_targets,
                                                              weights=masks,
                                                              average_across_timesteps=True,
                                                              average_across_batch=True)
        # Training summary for the current batch_loss
        tf.summary.scalar('loss', self.loss_function)

        # Contruct graphs for minimizing loss
        self.get_RNN_optimizer()

    def get_predict_model(self, input_proj_layer, output_proj_layer):

        start_word = tf.ones([self.inputs_batch_size, ], tf.int32)
        start_word *= SOS_TOKEN

        end_word = EOS_TOKEN

        def embedd_project_input(inputs):
            return input_proj_layer(tf.nn.embedding_lookup(self.embeddings, inputs))

        # Beamsearch is used to approximately find the most likely translation
        print("building beamsearch decoder..")
        inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.deep_cell,
                                                                  embedding=embedd_project_input,
                                                                  start_tokens=start_word,
                                                                  end_token=end_word,
                                                                  initial_state=self.state,
                                                                  beam_width=self.beam_width,
                                                                  output_layer=output_proj_layer, )
        # For BeamSearchDecoder, return
        # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
        #                         namedtuple(predicted_ids, beam_search_decoder_output)
        # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width] if output_time_major=False
        #                                       [max_time_step, batch_size, beam_width] if output_time_major=True
        # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance
        #                                                    namedtuple(scores, predicted_ids, parent_ids)

        (self.decoder_outputs_decode, self.decoder_last_state_decode,
         self.decoder_outputs_length_decode) = (tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                                  output_time_major=False,
                                                                                  maximum_iterations=self.max_decode_step))

        # Use beam search to approximately find the most likely translation
        # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
        self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

    def create_custom_LSTM_decoder(self):
        with tf.variable_scope('decoder'):
            self.get_deep_attention_LSTM_cell()
            self.embeddings = tf.get_variable(name='embedding_weights',
                                              shape=[self.vocabulary_size, self.embedding_size],
                                              initializer=self.glove_weights_initializer,
                                              dtype=self.dtype, trainable=False)

            # Projection layer for projecting encoder's output to encoder num of hidden nodes
            input_proj_layer = Dense(self.hidden_units_num, dtype=self.dtype, name='input_projection_layer')

            # Projection layer for projecting decoder's outputs to probability selection of a vocabulary word
            output_proj_layer = Dense(self.vocabulary_size, name='output_projection_layer')

            if self.model_type == 'train_model':
                self.get_train_model(input_proj_layer, output_proj_layer)
            elif self.model_type == 'predict_model':
                self.get_predict_model(input_proj_layer, output_proj_layer)