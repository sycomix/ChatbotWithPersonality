import tensorflow as tf

import os
import gc
import numpy as np

import CorpusProcessor as cp
import EmbeddingsManager as em
import Retrieval_ChatBot as bot

import Params as p
import random

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.1


SW = 0

def create_model(session, params, model_type):
    model = bot.Retrieval_ChatBot(params)

    ckpt = tf.train.get_checkpoint_state("./retrieval_saved_models")
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Loading saved model...")
        model.restore(session, ckpt.model_checkpoint_path)
    else:
        session.run(tf.global_variables_initializer())

    return model

def train_on_dataset():

    with tf.Session(config=config) as session:
        # Get the model parameters ------------------------------------------------------------------------
        params = p.get_model_params()

        # Load the training corpus
        print("Loading corpus...")
        corpus_processor = cp.CorpusProcessor("movie_lines.txt", "movie_conversations.txt")
        corpus_processor.load_clean_corpus("./train_data/", ["cornell_train_data.txt", "twitter_train_data.txt"])
        corpus_processor.get_word_sentences()
        print("Done")

        # Load Glove embeddings
        print("Loading embeddings...")
        embedding_manager = em.EmbeddingsManager(params["embedding_file"], params["embedding_size"],
                                                 params["vocabulary_size"], corpus_processor.vocabulary,
                                                 corpus_processor.sorted_vocabulary)
        embedding_manager.load_embeddings()
        params["glove_weights_initializer"] = embedding_manager.glove_weights_initializer
        print("Done")

        # Compute word indexes using the loaded corpus and Glove embeddings
        print("Computing word embeddings...")
        corpus_processor.get_index_sentences(embedding_manager, params["sentence_max_len"])
        corpus_processor.get_personal_answers("./train_data/", "clean_einstein.txt", embedding_manager)
        params["vocabulary_size"] = embedding_manager.vocabulary_size
        print("Done")

        # Build the Seq2Seq trainning model
        print("Creating model...")
        model = create_model(session, params, "train_model")
        print("Done")
        # ----------------------------------------------------------------------------------------------

        print("Trainning model ...")
        MAX_EPOCHS = 30
        MAX_STEPS = None
        MAX_SWI = None
        SAVE_STEP = 10

        #personal_batch_size, personal, personal_len = corpus_processor.get_personal_answers_batch()
        #model.get_retrieval_answers_encoding(session, personal, personal_len)

        minimum_loss = 0x3f3f3f3f
        corpus_processor.shuffle_train_data()
        for epoch in range(MAX_EPOCHS):

            step = 0
            steps_without_improvement = 0
            train_position = 0
            while(True):
                # End of training
                if train_position >= len(corpus_processor.word_questions):
                    print("Training ended")
                    break

                # Exceeded maximum steps available
                if MAX_STEPS and step >= MAX_STEPS:
                    print("Max training steps")
                    break

                # Maximum steps ithout improvement
                if MAX_SWI and steps_without_improvement >= MAX_SWI:
                    print("Model converged")
                    break

                questions, questions_len, answers, answers_len = corpus_processor.get_train_batch(train_position,
                                                                                                  params["batch_size"],
                                                                                                  params["sentence_max_len"])

                negatives, negatives_len = corpus_processor.get_random_batch(train_position,
                                                                             params["batch_size"],
                                                                             params["sentence_max_len"])

                step_loss = model.train(session, questions, questions_len, answers, answers_len, negatives, negatives_len)
                #model.predict(session, questions, questions_len)

                print(epoch,", ", step, ": ", step_loss)

                # Check for model training convergence
                if step_loss >= minimum_loss:
                    steps_without_improvement += 1
                else:
                    minimum_loss = step_loss
                    steps_without_improvement = 0

                if step % SAVE_STEP == 0:
                    path = os.path.join("./retrieval_saved_models/", "chat_bot.ckpt")
                    model.save(session, path)
                    print("Model saved...")

                # Prepare data for the next epoch
                corpus_processor.shuffle_train_data()
                train_position += params["batch_size"]
                step += 1

                gc.collect()

        # Save the trained model into memory
        path = os.path.join("./retrieval_saved_models/", "chat_bot.ckpt")
        model.save(session, path)
        print("Done... model saved")

def main():


    sw = SW
    if sw == 0:
        train_on_dataset()

if __name__ == '__main__':
    main()