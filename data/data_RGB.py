import os
from .dataset_RGB import DataLoaderTrain, DataLoaderVal


def get_training_data(rgb_file, img_options):
    assert os.path.exists(rgb_file)
    return DataLoaderTrain(rgb_file, img_options)


def get_validation_data(rgb_file, img_options):
    assert os.path.exists(rgb_file)
    return DataLoaderVal(rgb_file, img_options)
