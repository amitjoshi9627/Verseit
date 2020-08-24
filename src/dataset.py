import config
import pandas as pd


def get_data():
    poem_data = []
    data = pd.read_csv(config.DATA_PATH, usecols=[
                       'Content']).iloc[:25].values.reshape(25,)

    for i in data:
        poem_data += i.lower().replace('\xa0', '').split('\n')

    return poem_data
