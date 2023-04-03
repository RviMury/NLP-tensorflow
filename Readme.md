# NLP Tensorflow
The following repo has various projects based on NPL using Deeplearning Framework TensorFlow. To install use
```python
!pip install tensorflow
```
### Some basic techniques for NLP
The basic step in any NLP task is converting the strings into machine interpretable format.
1. Tokenization is breaking the raw text into small chunks. Tokenization breaks the raw text into words, sentences called tokens. These tokens help in understanding the context or developing the model for the NLP. The tokenization helps in interpreting the meaning of the text by analyzing the sequence of the words.

2. Padding is the process of making all numerically converted sentences of equal length, by padding the smaller sentences with zeros in fornt or back. The maximum length of sentences could be chosen by user or could be largest sentence in the corpus.

The above two task could be completed by using follwoing code. Sentences variable stores our sentences
For Training Data
```python
sentences=['I hate that part of the movie','Weather is great','who would hate this']
tokenizer=Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(sentences)
padded_sequences=pad_sequences(sequences)
padded_sequences
```

For Test Data
```python
test_sentences=['Can Someone bring me water','The sky is blue']
tokenizer.texts_to_sequences(test_sentences)
```

3. Embedding is the process of conceptually capturing the meaning of words. Every word is represented in n-dimensional vector. The distance between similar words, say man and women are closer than man and weather.


## Sentiment_Classification
The code classifies news headlines to dectet weather they are sarcastic or not
-- Dataset: Dataset can be found on kaggle :https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection.
-- There are two classes, with 0: Non Sarcastic and 1: Sacrastic.
-- The model uses a corpus with vocabulary size of 28165 with 706,574 trainable parameters.
-- The model performs with an accuracy of 0.76 with recall value of 0.8 on sarcastic class.
