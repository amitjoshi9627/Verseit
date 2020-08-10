import config
import dataset
import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def preprocess_data(data):

    tokenizer = config.TOKENIZER
    tokenizer.fit_on_texts(data)
    utils.save_tokenizer(tokenizer)

    input_sequences = []
    for line in data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_len = max(len(i) for i in input_sequences)
    input_sequences = pad_sequences(
        input_sequences, maxlen=max_len, padding=config.PADDING)
    train = input_sequences[:, :-1]
    labels = input_sequences[:, -1]
    total_words = len(tokenizer.word_index) + 1
    labels = to_categorical(labels, num_classes=total_words)
    config.set_value(max_len, total_words)
    return train, labels


def predict(seed_data):
    model = utils.load()
    tokenizer = utils.load_tokenizer()
    file = utils.load_file()
    max_len = file["max_len"]
    for _ in range(config.NEXT_WORDS):
        token_list = tokenizer.texts_to_sequences([seed_data])[0]
        pad_sequence = pad_sequences(
            [token_list], maxlen=max_len-1, padding=config.PADDING)
        prediction = model.predict_classes(pad_sequence, verbose=0)

        output_word = ''
        for word, ind in tokenizer.word_index.items():
            if ind == prediction:
                output_word = word
                break
        seed_data += " " + output_word

    return seed_data
