import os
import logging
import sys
import time

sys.path.append('./../modules')

from keras.preprocessing.image import ImageDataGenerator
from visualize import plot_history
from model import KarutaNet
from utils import make_dir

logging.basicConfig(level=logging.DEBUG)


def train(argv=None):

    # ------------ Set Paths and Parameters ----------- #
    train_path = '/root/share/local_data/FontKaruta/yoshida_work/data/true_48fonts/48fonts_dataset_tr700_va200_te100_half_rotate/train'
    val_path = '/root/share/local_data/FontKaruta/yoshida_work/data/true_48fonts/48fonts_dataset_tr700_va200_te100_half_rotate/validation'
    result_path = '/root/share/local_data/MSgothicPolice/result/'

    img_resize = (320, 240)
    input_shape = (None, None, 3)
    batch_size = 32
    steps_per_epoch = 150
    validation_steps = 100
    epochs = 50
    n_categories = 48

    # ------------ Data generator ----------- #
    datagen = ImageDataGenerator(rescale=1./255)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_resize,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = datagen.flow_from_directory(
        val_path,
        target_size=img_resize,
        batch_size=batch_size,
        class_mode='categorical')

    # -------------- Build Model -------------- #
    time_string = time.strftime('%Y%m%d_%H%M%S')
    checkpoints_path = os.path.join(result_path, time_string)
    make_dir(checkpoints_path)

    print('Creating KarutaNet...')
    knet = KarutaNet(input_shape=input_shape, n_categories=n_categories, checkpoints_path=checkpoints_path)
    knet.build()

    # -------------- Training -------------- #
    print('Training...')
    start_time = time.time()

    _ = knet.fit_generator(train_generator=train_generator,
                           steps_per_epoch=steps_per_epoch,
                           epochs=epochs,
                           validation_generator=validation_generator,
                           validation_steps=validation_steps)

    end_time = time.time()
    train_time = (end_time - start_time)
    print('Training time (minutes): %.3f' % (train_time / 60))

    history_data = _.history
    plot_history(checkpoints_path, history_data)

    return 0

if __name__ == '__main__':
    sys.exit(train())
