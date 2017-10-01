import glob
import os


def make_dir(path):
    if not os.path.exists(path):
        return os.makedirs(path)


def get_n_files_indir(data_dir):
    """Count how may files in specified dir

    purpose for decide batch size and steps per epoch
    """
    glob_pattern = '*/*.jpg'
    files_indir = glob.glob(os.path.join(data_dir, glob_pattern))
    return len(files_indir)


def get_n_classes(data_dir):
    return len(os.listdir(data_dir))
