import utils
import config
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
import tensorflow as tf


class PoetryGeneratorModel:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(Embedding(config.TOTAL_WORDS,
                                 config.EMBEDDING_DIM, input_length=config.MAX_LEN-1))
        self.model.add(Bidirectional(LSTM(250,return_sequences=True)))
        self.model.add(Bidirectional(LSTM(250)))
        self.model.add(Dense(config.TOTAL_WORDS, activation="softmax"))
        self.model.compile(
            loss=config.LOSS, optimizer=config.OPTIMIZER, metrics=['accuracy'])

    def train(self, data, labels):
        history = self.model.fit(data, labels, epochs=config.EPOCHS)
        utils.save(self.model)
        return history
