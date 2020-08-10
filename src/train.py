import engine
import utils
from dataset import get_data
from model import PoetryGeneratorModel


def train():
    data = get_data()
    data, labels = engine.preprocess_data(data)
    model = PoetryGeneratorModel()
    history = model.train(data, labels)
    utils.plot_graphs(history)


if __name__ == "__main__":
    train()
