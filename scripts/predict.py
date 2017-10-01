import os
import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg') # Absolutely set under the line of "import matplotlib"
import matplotlib.pyplot as plt

sys.path.append('./../modules')
from utils import get_n_files_indir, get_n_classes
from layers import SpatialPyramidPooling2D

from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator


def get_best_weight_file(result_dir_path):
    """Acquires the weight file having the largest file number from the specified directory.

    :param result_dir_path: Directory path where hdf 5 is stored (str)
    :return File name of the hdf5-file with the lowest loss value (str)
    """
    result_dir_list = glob.glob(os.path.join(result_dir_path,'*'))
    weight_file_list = [file for file in result_dir_list if os.path.splitext(file)[-1] == '.hdf5']
    best_weight_file = weight_file_list[-1]
    return best_weight_file


def get_y_true(class_labels, filenames_path_list):
    """Acquire the correct label name from the file name.

    :param class_labels: Class label to retrieve from train generator (dict)
    :param filenames_path_list: List of file names to be obtained from the test data directory (list)
    :return y_true: true label index (list)
    :return y_true_label: true label (list)
    """
    class_labels_sorted = sorted(class_labels.items(), key=lambda x: x[1]) # Sort for just in case

    y_true = []
    y_true_label = []

    for file_name_path in filenames_path_list:
        file_name = os.path.basename(file_name_path)
        file_name_split = file_name.split('.')[0].split('_')
        file_name_split.pop(0)
        true_file_name = '_'.join(file_name_split)

        # get actual info from filename
        try:
            actual_label, actual_num = \
            [(label, class_num) for label, class_num in class_labels_sorted if label == true_file_name][0]
            y_true.append(actual_num)
            y_true_label.append(actual_label)
        except IndexError as e:
            print(e)

    return y_true, y_true_label


def get_y_predicted_from_prob(probs, class_labels):
    """Acquire the predicted label index and label name.

    :param probs: A ndarray containing probability values of each class acquired by predict (ndarray)
    :param class_labels: Class label to retrieve from train generator (dict)
    :return y_pred: pred label index (list)
    :return y_pred_label: pred label (list)
    """
    if isinstance(probs, list) or probs.shape[1] == 1:
        raise ValueError('probs shape need to be at least 2d.')
    y_pred = [p.tolist().index(max(p)) for p in probs]
    class_labels_sorted = sorted(class_labels.items(), key=lambda x: x[1])
    y_pred_label = [class_labels_sorted[idx][0] for idx in y_pred]
    return y_pred, y_pred_label


def confusion_matrix_as_df(y_pred, y_true, class_labels):
    """Create a confusion matrix from y_pred and y_true.

    :param y_pred: (list)
    :param y_true: (list)
    :param class_labels: (dict)
    :return (pandas dataframe)
    """
    class_labels_sorted = sorted(class_labels.items(), key=lambda x: x[1])
    df = pd.DataFrame(confusion_matrix(y_pred, y_true),
                      index=['Pred:{}'.format((v, k)) for k, v in class_labels_sorted if v in set(y_pred + y_true)],
                      columns=['True:{}'.format((v, k)) for k, v in class_labels_sorted if v in set(y_pred + y_true)])
    return df


def make_result_vis(probs, result_dir_path, class_labels, filenames, save_csv_name):
    """Create a heat map from the confusion matrix and save it

    :param probs:
    :param result_dir_path:
    :param class_labels:
    :param filenames:
    :param save_csv_name:
    :return:
    """
    # get true label
    y_true, y_true_label = get_y_true(class_labels, filenames)
    # get predict label
    y_pred, y_pred_label = get_y_predicted_from_prob(probs, class_labels)
    # save accuracy score
    print("Accuracy Score: {:.2f}".format(accuracy_score(y_pred, y_true)))
    np.savetxt(save_csv_name, np.array([accuracy_score(y_pred, y_true)]), delimiter=',')

    df = confusion_matrix_as_df(y_pred, y_true, class_labels)

    # plot confusion matrix and save it
    f, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(df, annot=True, fmt="d", linewidths=.2, cmap="YlGnBu", ax=ax)
    filename = "confusion_matrix.png"
    matrix_img_path = os.path.join(result_dir_path, filename)
    plt.savefig(matrix_img_path)


if __name__ == '__main__':

    # ------------ Set Paths and Parameters ----------- #
    result_dir_path = '/root/share/local_data/MSgothicPolice/result/Rotate_epoch200'
    train_dir = '/root/share/local_data/FontKaruta/yoshida_work/data/true_48fonts/48fonts_dataset_tr700_va200_te100_half_rotate/train/'
    test_dir = '/root/share/local_data/FontKaruta/yoshida_work/data/true_48fonts/48fonts_dataset_tr700_va200_te100_half_rotate/test/'

    best_weight_file = get_best_weight_file(result_dir_path)
    print('Selected weight file: ', best_weight_file)

    test_n_sample = get_n_files_indir(test_dir)
    print('Number of test images: ', test_n_sample)
    class_num = get_n_classes(train_dir)
    print('Number of classes: ', class_num)

    model = load_model(os.path.join(result_dir_path, best_weight_file),
                       custom_objects={'SpatialPyramidPooling2D': SpatialPyramidPooling2D})

    # ------------ Data generator ----------- #
    datagen = ImageDataGenerator(rescale=1./255)

    # Make a generator just to get class_indices
    train_generator = datagen.flow_from_directory(
        train_dir
        )

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(320, 240),
        batch_size=test_n_sample // 32, # 32 is original batch size
        class_mode=None,
        shuffle=False
        )

    # ------------ Prediction ----------- #
    print('Start predict per batch...')
    predicted_list = []
    for test_data_batch in test_generator:
        print('batch_index:', test_generator.batch_index)
        predicted_batch = model.predict_on_batch(test_data_batch)
        predicted_list.append(predicted_batch)
        if test_generator.batch_index == 0:  # finish the batch loop
            break
    print('Finish predicted all images.')

    probs = np.empty((0, class_num), float)  # Create an empty array of 0rows and 48columns (48 is the number of classes)
    for predicted_batch in predicted_list:
        for predicted in predicted_batch:
            probs = np.append(probs, np.array([predicted]), axis=0)

    # ------------ Make result visualization ----------- #
    make_result_vis(probs,
                    result_dir_path,
                    class_labels=train_generator.class_indices,
                    filenames=test_generator.filenames,
                    save_csv_name=os.path.join(result_dir_path, 'accuracy_score.csv'))
    print('Saved confusion_matrix.png')
