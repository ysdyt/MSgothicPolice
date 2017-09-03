import os

def make_dir(path):
    if not os.path.exists(path):
        return os.makedirs(path)
