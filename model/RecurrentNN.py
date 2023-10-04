import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense
from keras.layers import *
from keras.callbacks import EarlyStopping


class RecurentNeuralNetwork:
    def __init__(self):
        embedding_dim = 100  # Dimension of word embeddings
        rnn_units = 100  # Number of units in the RNN layer
        max_words = 8000  # Maximum number of words in the vocabulary
        max_sequence_length = 127  # Maximum length of sequences
        self.model = Sequential()
        self.model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
        self.model.add(LSTM(rnn_units))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    def fit(self, X, Y, epochs, batch_size):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle= True, verbose= 1,
                              callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    def evaluate(self, X, Y):
        loss, accuracy= self.model.evaluate(X, Y)
        print(loss, accuracy)
        return loss, accuracy




