from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, issparse
import numpy as np

class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, questions, answers):
        self.vectorizer.fit(np.append(questions, answers))

    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)

        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T)
        result = np.asarray(result).flatten()

        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]