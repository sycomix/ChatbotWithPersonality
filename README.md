# ChatbotWithPersonality
Chatbot that embodies the personality of Albert Einstein by using a hybrid architecture, combining generative and retrieval based chatbot models.

# Architecture
Based on a Hybrid model of Seq2Seq generative and retrieval chatbot models.
- The generative model is trained on the Cornell Movie Dialogs and Twitter datasets, using GloVe word embeddings (training corpus).
- The retrieval based model is trained on a hand-picked corpus containing information about the life of Albert Einstein (personal corpus).

The model is composed of 3 stages:
- Stage1: TF-IDF is employed in order to predict the best response from the personal corpus
- Stage2: in case of a low TF-IDF similarity score, the Encoder of the Seq2Seq model is used in order to embed the input question, and cosine similarity is computed with the possible responses from the personal corpus. The best response is selected.
- Stage3: in case of a low cosine similarity score, the Decoder part of the generative model is used, and a response in generated word by word using BeamSearch.

# Architecture Diagram   
![image](https://user-images.githubusercontent.com/37878793/158273168-b9791a22-1004-49f5-9b09-1f44a9e1dfc3.png)

# Implementation
- The project is implemented using Tensorflow and Python.

