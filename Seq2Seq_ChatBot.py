import tensorflow as tf
import numpy as np
from Encoder import Encoder
from Decoder import Decoder
from copy import deepcopy
import operator

class Seq2Seq_ChatBot(object):

    def __init__(self, params, model_type):
        # Create an encoder instance
        self.encoder = Encoder(params)
        self.encoder.create_custom_LSTM_encoder()

        # Create a decoder instance
        self.decoder = Decoder(params, self.encoder, model_type)
        self.decoder.create_custom_LSTM_decoder()

    def train(self, sess, encoder_inputs, encoder_inputs_len, decoder_inputs, decoder_inputs_len):
        input_feed = {
            self.encoder.inputs.name: encoder_inputs,
            self.encoder.inputs_len.name: encoder_inputs_len,
            self.decoder.inputs.name: decoder_inputs,
            self.decoder.inputs_len.name: decoder_inputs_len,
        }
        output_feed = [self.decoder.gradient_updates, self.decoder.loss_function]

        outputs = sess.run(output_feed, input_feed)
        return outputs[1]

    def predict(self, sess, encoder_inputs, encoder_inputs_len):
        input_feed = {
            self.encoder.inputs.name: encoder_inputs,
            self.encoder.inputs_len.name: encoder_inputs_len,
        }
        output_feed = [self.decoder.decoder_pred_decode]
        outputs = sess.run(output_feed, input_feed)

        return outputs[0]  # BeamSearchDecoder: [batch_size, max_time_step, beam_width]

    def get_retrieval_answers_encoding(self, sess, answers, answers_len):
        input_feed = {
            self.encoder.inputs.name: answers,
            self.encoder.inputs_len.name: answers_len,
        }
        output_feed = [self.encoder.outputs, self.encoder.state]
        self.retrieval_answers_encoding = sess.run(output_feed, input_feed)[1][-1].h

    def get_active_cosine_similarity(self, questions_encoding, answers_encoding):
        return np.sum(np.square( np.subtract(questions_encoding, answers_encoding)), 1)

    def retrieval_predict(self, sess, questions, questions_len):
        input_feed = {
            self.encoder.inputs.name: questions,
            self.encoder.inputs_len.name: questions_len,
        }

        output_feed = [self.encoder.outputs, self.encoder.state]
        questions_enc = sess.run(output_feed, input_feed)[1][-1].h

        indexes = []
        for i in range(len(questions_enc)):
            questions_encoding = [deepcopy(questions_enc[i]) for _ in range(len(self.retrieval_answers_encoding))]

            results = self.get_active_cosine_similarity(questions_encoding, self.retrieval_answers_encoding)
            sorted_results = [(results[i], i) for i in range(len(results))]
            sorted_results = sorted(sorted_results, reverse=False)

            indexes.append(deepcopy(sorted_results[:5]))
        return indexes

    def save(self, sess, path, var_list=None, global_step=None):
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print(f'model saved at {save_path}')

    def restore(self, sess, path, var_list=None):
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print(f'model restored from {path}')


def get_active_cosine_similarity(questions_encoding, answers_encoding):
    num = np.sum(np.matmul(questions_encoding, np.transpose(answers_encoding)), axis=1)

    questions_norm = np.sqrt(np.sum(np.matmul(questions_encoding, np.transpose(questions_encoding)), axis=1))
    answers_norm = np.sqrt(np.sum(np.matmul(answers_encoding, np.transpose(answers_encoding)), axis=1))
    denom = np.multiply(questions_norm, answers_norm)

    margin = np.ones(np.shape(num)).astype('float64')

    return np.subtract(margin, np.divide(num, denom))

if __name__ == '__main__':
    from nltk.translate.bleu_score import corpus_bleu

    references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']],[["aha"]]]
    candidates = [['this', 'is', 'a', 'test'], ["p"]]
    score = corpus_bleu(references, candidates, weights=(1.0,0,0,0))
    print(score)

