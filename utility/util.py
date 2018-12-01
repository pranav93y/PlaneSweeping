import os


def join_path(string, value):
    return os.path.join(string, value)


def sort_and_reshape_to_1D(array):
    array.sort(axis=0)
    array = array.reshape(array.shape[0])
    return array
