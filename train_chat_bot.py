import tensorflow as tf
from sys import stdin
from copy import deepcopy
from nltk.translate.bleu_score import corpus_bleu
import random
import os
import gc

import CorpusProcessor as cp
import EmbeddingsManager as em
import Seq2Seq_ChatBot as bot
import Params as p

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.1

SW = 1
OP = "predict_seq2sqq"
SIM_EPS = 17

def create_model(session, params, model_type):
    model = bot.Seq2Seq_ChatBot(params, model_type)

    ckpt = tf.train.get_checkpoint_state("./saved_models")
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Loading saved model...")
        model.restore(session, ckpt.model_checkpoint_path)
    else:
        session.run(tf.global_variables_initializer())

    return model

def train_on_corpus():

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
        params["vocabulary_size"] = embedding_manager.vocabulary_size
        print(embedding_manager.vocabulary_size)
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

                step_loss = model.train(session,
                                                 encoder_inputs=questions, encoder_inputs_len=questions_len,
                                                 decoder_inputs=answers, decoder_inputs_len=answers_len)

                print(epoch,", ", step, ": ", step_loss)

                # Check for model training convergence
                if step_loss >= minimum_loss:
                    steps_without_improvement += 1
                else:
                    minimum_loss = step_loss
                    steps_without_improvement = 0

                if step % SAVE_STEP == 0:
                    path = os.path.join("./saved_models/", "chat_bot.ckpt")
                    model.save(session, path)
                    print("Model saved...")

                # Prepare data for the next epoch
                corpus_processor.shuffle_train_data()
                train_position += params["batch_size"]
                step += 1

                gc.collect()

        # Save the trained model into memory
        path = os.path.join("./saved_models/", "chat_bot.ckpt")
        model.save(session, path)
        print("Done... model saved")

# EVALUATION/PREDICTION ---------------------------------------------------------------------------------
def evaluate_BLEU_on_corpus(session, model, corpus_processor, embedding_manager, params):

    print("Evaluating BLEU score...")

    MAX_STEPS = 10
    step = 0
    eval_position = 0
    BLEU_references = []
    BLEU_answers = []

    corpus_processor.shuffle_train_data()
    while (True):
        # End of evaluation
        if eval_position >= len(corpus_processor.word_questions):
            print("Evaluation ended")
            break

        # Exceeded maximum steps available
        if MAX_STEPS and step >= MAX_STEPS:
            print("Max evaluation steps")
            break

        questions, questions_len, answers, answers_len = corpus_processor.get_train_batch(
            eval_position,
            params["batch_size"],
            None)

        outputs = model.predict(session, questions, questions_len)
        for i in range(0, params["batch_size"]):
            current_question = embedding_manager.index_sentence_to_words(corpus_processor.word_questions[eval_position + i])
            current_answer = embedding_manager.index_sentence_to_words(corpus_processor.word_answers[eval_position + i])

            beam = params["beam_width"]
            large_answer_index = random.randrange(max(0, beam - 5), beam)
            small_answer_index = random.randrange(1, min(5, beam))

            choice = random.random()
            answer_index = small_answer_index
            if choice > 0:
                answer_index = large_answer_index

            generated_answer = []
            for sentence in outputs[i]:
                current_sentence = embedding_manager.index_sentence_to_words(sentence)

                if current_sentence[answer_index] == '<_EOS>':
                    break
                generated_answer.append(current_sentence[answer_index])

            BLEU_references.append([deepcopy(current_answer)])
            BLEU_answers.append(deepcopy(generated_answer))

        BLEU_score = corpus_bleu(BLEU_references, BLEU_answers, weights=(1.0, 0, 0, 0))
        print(BLEU_score)

        eval_position += params["batch_size"]
        step += 1
        gc.collect()

    BLEU_score = corpus_bleu(BLEU_references, BLEU_answers, weights=(1.0, 0, 0, 0))
    print(BLEU_score)


def seq2seq_predict(session, model, corpus_processor, embedding_manager, params, BLEU_on, single_question):
    #corpus_processor.shuffle_train_data()

    start = 0
    if not single_question:
        questions, questions_len, answers, answers_len = corpus_processor.get_train_batch(
                                                         start, params["batch_size"],
                                                         params["sentence_max_len"])
    else:
        questions, questions_len = corpus_processor.process_single_question(single_question, embedding_manager,
                                                                            params["sentence_max_len"])

    outputs = model.predict(session, questions, questions_len)

    if BLEU_on:
        BLEU_references = []
        BLEU_answers = []

    for i in range(0, params["batch_size"]):
        current_question = embedding_manager.index_sentence_to_words(corpus_processor.word_questions[start + i])
        current_answer = embedding_manager.index_sentence_to_words(corpus_processor.word_answers[start + i])

        beam = params["beam_width"]
        large_answer_index = random.randrange(max(0, beam - 5), beam)
        small_answer_index = random.randrange(1, min(5, beam))

        choice = random.random()
        answer_index = small_answer_index
        if choice > 0.2:
            answer_index = large_answer_index

        generated_answer = []
        for sentence in outputs[i]:
            current_sentence = embedding_manager.index_sentence_to_words(sentence)

            if current_sentence[answer_index] == '<_EOS>':
                break
            generated_answer.append(current_sentence[answer_index])

        if BLEU_on:
            print(current_question)
            print(current_answer)
            BLEU_references.append([deepcopy(current_answer)])
            BLEU_answers.append(deepcopy(generated_answer))

        generated_string = ""
        for item in generated_answer:
            generated_string = generated_string + item + " "

        #generated_string = deepcopy(generated_string[0].upper()) + deepcopy(generated_string[1:]) + "."
        print(generated_string)
        print()

    if BLEU_on:
        BLEU_score = corpus_bleu(BLEU_references, BLEU_answers, weights=(1.0, 0, 0, 0))
        print(BLEU_score)

        BLEU_score = corpus_bleu(BLEU_references, BLEU_answers, weights=(0.50, 0.50, 0, 0))
        print(BLEU_score)

        BLEU_score = corpus_bleu(BLEU_references, BLEU_answers, weights=(0.33, 0.33, 0.33, 0))
        print(BLEU_score)

        BLEU_score = corpus_bleu(BLEU_references, BLEU_answers, weights=(0.25, 0.25, 0.25, 0.25))
        print(BLEU_score)
        print()

def retrieval_predict(session, model, corpus_processor, embedding_manager, params, single_question):

    start = 0
    num_answers = 1

    if not single_question:
        questions, questions_len, answers, answers_len = corpus_processor.get_train_batch(start, params["batch_size"],
                                                                                      params["sentence_max_len"])
    else:
        questions, questions_len = corpus_processor.process_single_question(single_question, embedding_manager,
                                                                            params["sentence_max_len"])

    outputs = model.retrieval_predict(session, questions, questions_len)

    for i, out in enumerate(outputs):
        current_question = embedding_manager.index_sentence_to_words(corpus_processor.word_questions[start + i])
        #print(current_question)

        for j in range(num_answers):
            personal_answer = corpus_processor.personal_answers[out[j][1]]

            if out[j][0] <= SIM_EPS:
                print(personal_answer)
                #print(out[j][0])

        if single_question:
            return out[0][0]

        print()

def evaluation_prediction(operation_type):
    with tf.Session(config=config) as session:

        # Get the model parameters
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
        model = create_model(session, params, "predict_model")
        print("Done")

        # ------------------------------------------------------------------------------------------

        # Batch retrieval and prediction ------------------------------------------------------------
        if operation_type == "evaluate_BLEU":
            evaluate_BLEU_on_corpus(session, model, corpus_processor, embedding_manager, params)
        elif operation_type == "predict_seq2seq":
            seq2seq_predict(session, model, corpus_processor, embedding_manager, params, True, None)
        elif operation_type == "predict_retrieval":

            # Get personal answers batch -----------------
            personal_batch_size, personal, personal_len = corpus_processor.get_personal_answers_batch()
            model.get_retrieval_answers_encoding(session, personal, personal_len)

            retrieval_predict(session, model, corpus_processor, embedding_manager, params, None)
        else:
            # Get personal answers batch -----------------
            personal_batch_size, personal, personal_len = corpus_processor.get_personal_answers_batch()
            model.get_retrieval_answers_encoding(session, personal, personal_len)

            params["batch_size"] = 1
            for question in stdin:
                if question == "":
                    continue

                params["input_keep_prob"] = 1.0 - 0.0
                params["output_keep_prob"] = 1.0 - 0.0
                sim = retrieval_predict(session, model, corpus_processor, embedding_manager, params, question)

                params["input_keep_prob"] = 1.0 - 0.2
                params["output_keep_prob"] = 1.0 - 0.2
                if sim > SIM_EPS:
                    seq2seq_predict(session, model, corpus_processor, embedding_manager, params, False, question)

def main():

    sw = SW
    if sw == 0:
        train_on_corpus()
    else:
        evaluation_prediction(OP)

if __name__ == '__main__':
    main()