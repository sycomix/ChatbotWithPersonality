import os
import string
from ast import literal_eval
import numpy as np
import operator
from random import shuffle
import gc

import EmbeddingsManager as em
from nltk import sent_tokenize
import re
import random
from collections import OrderedDict

SOS_TOKEN = 0   # Start of sentence token
EOS_TOKEN = 1   # End of sentence token and padding
UNK_TOKEN = 2   # Unknown word token

class CorpusProcessor(object):

    def __init__(self, movie_lines_file, movie_conversations_file):
        self.movie_lines_file = movie_lines_file
        self.movie_conversations_file = movie_conversations_file

    # RAW CORPUS operations ------------------------------------------------
    def get_hashed_movie_lines(self):
        line2content = {}

        max_elements_per_line = 5
        delimitator = ' +++$+++ '
        line_code_index = 0
        line_content_index = -1

        with open(self.movie_lines_file, 'r', encoding='iso-8859-1') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                elements = line.split(delimitator)

                if len(elements) < max_elements_per_line:
                    continue

                line_code = elements[line_code_index]
                line_content = elements[line_content_index]
                line2content[line_code] = line_content

        return line2content

    def get_coded_conversations(self):
        coded_conversations = []

        max_elements_per_line = 4
        delimitator = ' +++$+++ '
        coded_conversation_index = -1

        with open(self.movie_conversations_file, 'r', encoding='iso-8859-1') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                elements = line.split(delimitator)

                if len(elements) < max_elements_per_line:
                    continue

                coded_conversation_string = elements[coded_conversation_index]
                coded_conversation_list = literal_eval(coded_conversation_string)
                coded_conversations.append(coded_conversation_list)

        return coded_conversations

    def get_question_answer_set(self, line2content, coded_conversations):
        questions = []
        answers = []

        for conversation in coded_conversations:
           if len(conversation) < 2:
               continue

           for index in range(0, len(conversation), 2):
               if index + 1 < len(conversation):
                   question_code = conversation[index]
                   answer_code = conversation[index + 1]

                   questions.append(line2content[question_code])
                   answers.append(line2content[answer_code])

        return questions, answers


    def save_clean_corpus(self, questions, answers, path):
        train_data_filepath = os.path.join(path, 'cornell_train_data.txt')
        with open(train_data_filepath, 'w', encoding='utf8') as train_data:
            for i in range(len(questions)):
                train_data.write(questions[i])
                train_data.write(answers[i])


    def compute_clean_corpus(self, path):
        line2content = self.get_hashed_movie_lines()
        coded_conversations = self.get_coded_conversations()
        questions, answers = self.get_question_answer_set(line2content, coded_conversations)

        self.questions = questions
        self.answers = answers

        self.save_clean_corpus(questions, answers, path)
    # -------------------------------------------------------------------------------------------------

    # CLEAN CORPUS operations -------------------------------------------------------------------------
    def load_clean_corpus(self, path, file_names):
        questions = []
        answers = []

        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            with open(file_path, 'r', encoding='utf8') as file:
                for index, line in enumerate(file):
                    if index % 2 == 0:
                        questions.append(line)
                    else:
                        answers.append(line)

        self.questions = questions
        self.answers = answers


    def get_word_sentences(self):
        word_questions = []
        word_answers = []
        vocabulary = {}

        remove_punctuation = str.maketrans('', '', string.punctuation)
        remove_digits = str.maketrans('', '', string.digits)
        for index in range(len(self.questions)):
            self.questions[index] = self.questions[index].lower().translate(remove_digits).translate(remove_punctuation)
            q_words = self.questions[index].split()

            self.answers[index] = self.answers[index].lower().translate(remove_digits).translate(remove_punctuation)
            a_words = self.answers[index].split()

            if len(q_words) > 0 and len(a_words) > 0:
                word_questions.append(list(q_words))
                word_answers.append(list(a_words))

                for w in q_words:
                    if w not in vocabulary:
                        vocabulary[w] = 1
                    else:
                        vocabulary[w] += 1

                for w in a_words:
                    if w not in vocabulary:
                        vocabulary[w] = 1
                    else:
                        vocabulary[w] += 1

        self.word_questions = word_questions
        self.word_answers = word_answers
        self.vocabulary = vocabulary
        self.sorted_vocabulary = sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)

    def filter_unk_sentences(self, sentence, sentence_max_len):
        num_unk_tokens = sum(1 for token in sentence if token == UNK_TOKEN)
        sentence_len = min(len(sentence), sentence_max_len)

        return 100.0*num_unk_tokens/sentence_len <= 20


    def get_index_sentences(self, embedding_manager, sentence_max_len):
        word_questions = []
        word_answers = []
        for index in range(len(self.word_questions)):
            q_converted_sentence = embedding_manager.sentence_to_word_indexes(self.word_questions[index])
            a_converted_sentence = embedding_manager.sentence_to_word_indexes(self.word_answers[index])

            if len(q_converted_sentence) > 0 and len(a_converted_sentence) > 0 \
                    and self.filter_unk_sentences(q_converted_sentence, sentence_max_len) \
                    and self.filter_unk_sentences(a_converted_sentence, sentence_max_len):
                word_questions.append(q_converted_sentence)
                word_answers.append(a_converted_sentence)

        self.word_questions = word_questions
        self.word_answers = word_answers

        # Free unused memory
        gc.collect()
    # -----------------------------------------------------------------------------------------------------------

    # Training methods ------------------------------------------------------------------------------------------
    def get_train_batch(self, position, batch_size, max_len):
        left = position
        if position + batch_size - 1 >= len(self.word_questions):
            left = len(self.word_questions) - batch_size

        def truncate(sentence):
            return sentence[:max_len] if max_len and len(sentence) > max_len else sentence

        questions = [truncate(self.word_questions[i]) for i in range(left, left + batch_size)]
        questions_len = [len(question) for question in questions]

        answers = [truncate(self.word_answers[i]) for i in range(left, left + batch_size)]
        answers_len =[len(answer) for answer in answers]

        questions_len = np.array(questions_len)
        answers_len = np.array(answers_len)

        maxlen_questions = np.max(questions_len)
        maxlen_answers = np.max(answers_len)

        padded_questions = np.ones((batch_size, maxlen_questions)).astype('int32') * em.EOS_TOKEN
        padded_answers = np.ones((batch_size, maxlen_answers)).astype('int32') * em.EOS_TOKEN

        for index, [question, answer] in enumerate(zip(questions, answers)):
            padded_questions[index, :questions_len[index]] = question
            padded_answers[index, :answers_len[index]] = answer

        return padded_questions, questions_len, padded_answers, answers_len

    def shuffle_train_data(self):
        cumulated_data = list(zip(self.word_questions, self.word_answers))
        shuffle(cumulated_data)

        self.word_questions = [sentence[0] for sentence in cumulated_data]
        self.word_answers = [sentence[1] for sentence in cumulated_data]

    # Einstein wikipedia corpus processing ----------------------------------------------
    def split_into_sentences(self, path, file_name, data_file_name):
        impersonal2personal = [("Albert Einstein is", 'I am'),
                               ("Albert Einstein's", 'my'),
                               ("Albert Einstein", 'I'),
                               ('Albert is', 'I am'),
                               ('Einstein is', 'I am'),
                               ("Albert's", 'my'),
                               ("Einstein's", 'my'),
                               ('Albert', 'I'),
                               ('Einstein', 'I'),
                               ('he is', 'I am'),
                               ('He is', 'I am'),
                               ('he', 'I'),
                               ('He', 'I'),
                               ('his', 'my'),
                               ('His', 'My'),
                               ('him', 'me'),
                               ('Him', 'Me'),
                               ('theirs', 'ours'),
                               ('Theirs', 'Ours'),
                               ('their', 'our'),
                               ('Their', 'Our'),
                               ]

        file_path = os.path.join(path, file_name)
        with open(file_path, 'r', encoding='utf8') as file:
            data_file_path = os.path.join(path, data_file_name)
            datafile = open(data_file_path, 'w', encoding='utf8')

            for index, line in enumerate(file):
                sentences = sent_tokenize(line)

                for index in range(len(sentences)):
                    sentences[index] = re.sub("\[\d+\]", "", sentences[index])

                    for key in impersonal2personal:
                        sentences[index] = re.sub(r"\b%s\b" % key[0], key[1], sentences[index])

                    if len(sentences[index]) > 5:
                        datafile.write(sentences[index])
                        datafile.write('\n')

        datafile.close()

    def get_personal_answers(self, path, file_name, embedding_manager):
        file_path = os.path.join(path, file_name)
        file = open(file_path, 'r', encoding='utf8')

        personal_answers = list(file)
        self.personal_answers = personal_answers

        word_personal_answers = []
        remove_punctuation = str.maketrans('', '', string.punctuation)
        remove_digits = str.maketrans('', '', string.digits)
        for personal_answer in self.personal_answers:
            formatted = (
                personal_answer.lower()
                .translate(remove_digits)
                .translate(remove_punctuation)
            )
            personal_words = formatted.split()

            word_personal_answers.append(list(personal_words))

        self.word_personal_answers = []
        for word_personal_answer in word_personal_answers:
            converted_sentence = embedding_manager.sentence_to_word_indexes(
                word_personal_answer
            )

            if len(converted_sentence) > 0:
                self.word_personal_answers.append(converted_sentence)


    def get_random_batch(self, position, batch_size, max_len):
        left = position
        if position + batch_size - 1 >= len(self.word_questions):
            left = len(self.word_questions) - batch_size

        def truncate(sentence):
            return sentence[:max_len] if len(sentence) > max_len else sentence

        negative_answers = []
        for i in range(batch_size):
            while True:
                negative_index = random.randrange(len(self.word_answers))
                if negative_index < left or negative_index >= left + batch_size:
                    break

            negative_answers.append(truncate(self.word_answers[negative_index]))

        negative_answers_len = [len(answer) for answer in negative_answers]
        negative_answers_len = np.array(negative_answers_len)
        maxlen_answers = np.max(negative_answers_len)

        padded_answers = np.ones((batch_size, maxlen_answers)).astype('int32') * em.EOS_TOKEN

        for index, answer in enumerate(negative_answers):
            padded_answers[index, :negative_answers_len[index]] = answer

        return padded_answers, negative_answers_len

    def get_personal_answers_batch(self):
        batch_size = len(self.word_personal_answers)
        answers = [self.word_personal_answers[i] for i in range(batch_size)]

        answers_len = [len(answer) for answer in answers]
        answers_len = np.array(answers_len)

        maxlen_answers = np.max(answers_len)
        padded_answers = np.ones((batch_size, maxlen_answers)).astype('int32') * em.EOS_TOKEN

        for index, answer in enumerate(answers):
            padded_answers[index, :answers_len[index]] = answer

        return batch_size, padded_answers, answers_len

    def process_single_question(self, question, embedding_manager, max_len):
        remove_punctuation = str.maketrans('', '', string.punctuation)
        remove_digits = str.maketrans('', '', string.digits)

        formatted = question.lower().translate(remove_digits).translate(remove_punctuation)
        question_words = formatted.split()
        question_indexes = embedding_manager.sentence_to_word_indexes(question_words)

        def truncate(sentence):
            return sentence[:max_len] if max_len and len(sentence) > max_len else sentence

        questions = [truncate(question_indexes)]
        questions_len = [len(question) for question in questions]

        questions_len = np.array(questions_len)
        maxlen_questions = np.max(questions_len)
        padded_questions = np.ones((1, maxlen_questions)).astype('int32') * em.EOS_TOKEN

        for index, question in enumerate(questions):
            padded_questions[index, :questions_len[index]] = question

        return padded_questions, questions_len

if __name__ == '__main__':
    processor = CorpusProcessor("movie_lines.txt", "movie_conversations.txt")
    #processor.compute_clean_corpus("./")

    processor.split_into_sentences("./train_data/", "raw_einstein.txt", "clean_einstein.txt")
    processor.load_clean_corpus("./train_data/", ["cornell_train_data.txt", "twitter_train_data.txt"])
    processor.get_word_sentences()

    embedding_manager = em.EmbeddingsManager("./embeddings/glove.6B.50d.txt", 50, 10000, processor.vocabulary, processor.sorted_vocabulary)
    embedding_manager.load_embeddings()

    processor.get_index_sentences(embedding_manager, 10)
    processor.get_personal_answers("./train_data/", "clean_einstein.txt", embedding_manager)
