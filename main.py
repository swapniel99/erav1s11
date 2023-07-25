from utils import *
from utils.experiment import Experiment
from models.resnetv2 import ResNet18
from datasets import CIFAR10

set_seed(42)
batch_size = 32
cifar10 = CIFAR10(batch_size)

model = ResNet18()


def print_summary():
    print(model_summary(model, input_size=(batch_size, 3, 32, 32)))


def create_experiment():
    return Experiment(model, cifar10, criterion='crossentropy', epochs=20, scheduler='one_cycle')


if __name__ == '__main__':
    create_experiment()
    experiment = create_experiment()
    experiment.execute()
    experiment.train.plot_stats()
    experiment.test.plot_stats()
    experiment.show_incorrect()
