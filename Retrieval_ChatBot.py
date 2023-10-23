import tensorflow as tf
import numpy as np

from tensorflow.python.layers.core import Dense
from TripletEncoder import TripletEncoder

class Retrieval_ChatBot(object):

    def __init__(self, params):
        self.dtype = tf.float32
        self.hidden_units_num = params["hidden_units_num"]
        self.gradient_clipping_norm = params["gradient_clipping_norm"]
        self.learning_rate = params["learning_rate"]

        # Create an encoder instance
        self.encoder = TripletEncoder(params)
        self.encoder.create_custom_LSTM_encoder()

        # Apply updates on the network
        self.initialize_hidden_state_results()
        self.update_variables()

    def initialize_hidden_state_results(self):
        self.questions_encoding = self.encoder.questions_state[-1].h
        self.answers_encoding = self.encoder.answers_state[-1].h
        self.negatives_encoding = self.encoder.negatives_state[-1].h

    def get_cosine_similarity(self, questions_encoding, answers_encoding):
        num = tf.reduce_sum(tf.matmul(questions_encoding, tf.transpose(answers_encoding)), axis=1)

        questions_norm = tf.sqrt(tf.reduce_sum(tf.matmul(questions_encoding, tf.transpose(questions_encoding)), axis=1))
        answers_norm = tf.sqrt(tf.reduce_sum(tf.matmul(answers_encoding, tf.transpose(answers_encoding)), axis=1))
        denom = tf.multiply(questions_norm, answers_norm)

        margin = tf.ones(tf.shape(num), tf.float32)
        num = tf.add(num, margin)
        denom = tf.add(num, margin)

        return tf.div(num, denom)

    def get_triplet_loss_function(self):
        positive_sim = self.get_cosine_similarity(self.questions_encoding, self.answers_encoding)
        negative_sim = self.get_cosine_similarity(self.questions_encoding, self.negatives_encoding)

        self.positive_sim = positive_sim
        self.negative_sim = negative_sim
        margin = tf.ones(tf.shape(positive_sim), tf.float32) * 0.4

        loss = tf.maximum(0., tf.add(tf.subtract(negative_sim, positive_sim), margin))
        loss = tf.reduce_mean(loss)

        return loss

    def update_variables(self):
        self.triplet_loss = self.get_triplet_loss_function()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        trainable_variables = tf.trainable_variables()
        variables_gradients = tf.gradients(self.triplet_loss, trainable_variables)

        clip_gradients, _ = tf.clip_by_global_norm(variables_gradients, self.gradient_clipping_norm)

        # Update the model
        self.gradient_updates = self.optimizer.apply_gradients(zip(clip_gradients, trainable_variables))

    # Trainning section ------------------------------------------------------------------------------
    def train(self, sess, questions, questions_len, answers, answers_len, negatives, negatives_len):
        # Feed the last hidden state of each type of sentences
        # And get minimize the loss function
        input_feed = {
            self.encoder.questions_inputs.name: questions,
            self.encoder.questions_inputs_len.name: questions_len,
            self.encoder.answers_inputs.name: answers,
            self.encoder.answers_inputs_len.name: answers_len,
            self.encoder.negatives_inputs.name: negatives,
            self.encoder.negatives_inputs_len.name: negatives_len,
        }
        output_feed = [self.gradient_updates, self.positive_sim, self.negative_sim, self.triplet_loss,
                       self.questions_encoding, self.answers_encoding]

        output = sess.run(output_feed, input_feed)
        #print(output[1])
        #print(output[2])

        return output[3]

    # Prediction section --------------------------------------------------------------------
    def get_retrieval_answers_encoding(self, sess, answers, answers_len):
        input_feed = {
            self.encoder.answers_inputs.name: answers,
            self.encoder.answers_inputs_len.name: answers_len,
        }
        output_feed = [self.encoder.answers_outputs, self.encoder.answers_state]
        self.retrieval_answers_encoding = sess.run(output_feed, input_feed)[1][-1].h
        print(self.retrieval_answers_encoding)

    def get_active_cosine_similarity(self, questions_encoding, answers_encoding):
        num = np.sum(np.matmul(questions_encoding, np.transpose(answers_encoding)), axis=1)

        questions_norm = np.sqrt(np.sum(np.matmul(questions_encoding, np.transpose(questions_encoding)), axis=1))
        answers_norm = np.sqrt(np.sum(np.matmul(answers_encoding, np.transpose(answers_encoding)), axis=1))
        denom = np.multiply(questions_norm, answers_norm)

        return np.division(num, denom)

    def predict(self, sess, questions, questions_len):
        input_feed = {
            self.encoder.questions_inputs.name: questions,
            self.encoder.questions_inputs_len.name: questions_len,
        }
        output_feed = [self.encoder.questions_outputs, self.encoder.questions_state]
        questions_encoding = sess.run(output_feed, input_feed)[1][-1].h

        for _ in range(len(self.retrieval_answers_encoding) - 1):
            questions_encoding.append(questions_encoding[0])

        results = self.get_active_cosine_similarity(questions_encoding, self.retrieval_answers_encoding)

        Max = 0
        index = 0
        for i in range(results):
            if results[i] > Max:
                Max = results[i]
                index = i

        return index

    def save(self, sess, path, var_list=None, global_step=None):
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print(f'model saved at {save_path}')

    def restore(self, sess, path, var_list=None):
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print(f'model restored from {path}')