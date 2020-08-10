from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
import utils

LOSS = categorical_crossentropy
OPTIMIZER = Adam(lr=0.01)

EPOCHS = 50

MODEL_PATH = "../saved_model/Model.h5"
TOKENIZER_PATH = "../saved_model/tokenizer.pickle"
DATA_PATH = "../data/poem_dataset.csv"

EMBEDDING_DIM = 300

TOKENIZER = Tokenizer()

PADDING = "pre"
NEXT_WORDS = 20

MAX_LEN, TOTAL_WORDS = None, None


def set_value(max_len, total_words):

    global MAX_LEN
    global TOTAL_WORDS

    MAX_LEN = max_len
    TOTAL_WORDS = total_words

    file = {"max_len": MAX_LEN, "total_words": TOTAL_WORDS}
    utils.save_file(file)


FILE_NAME = "../saved_model/file.pickel"
