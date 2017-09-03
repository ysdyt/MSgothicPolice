import os

from keras.models import load_model, Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard

from layers import SpatialPyramidPooling2D


class KarutaNet(object):
    """KarutaNet class.

    Args:
        input_shape (tuple of ints): Input image of shape (height, width, channels).
        n_categories (int): Number of document categories (size of softmax output).
        n_filters (list): Number of filters of convolutional layers.
        kernel_size (tuple of ints): Kernel size of convolutional layers.
        conv_padding (str): Type of convolution padding ('valid' or 'same')
        spp_levels (tuple of ints): Number of bins into which each of the axes of the input is partitioned
            for pooling (per channel)
        n_hidden (int): Number of hidden layers.
        dropout_rate (float): Dropout rate.
        checkpoints_path (str): Path where model files are saved.
        checkpoint_file_format (str): Format of checkpoint file names.
            See https://keras.io/callbacks/#modelcheckpoint for more details.
        optimizer (str): Name of a Keras optimizer. A Keras optimizer object can also be passed directly.
        save_best_only (bool): If True, save a checkpoint only when the validation loss improves.

    Methods:
        build: Build model and print model summary.
        load_from_file: Loads Keras model from specified path.
        fit_generator: Fits the model on data yielded from keras data generator.
        predict: Predict using model given an input.
    """

    def __init__(self, input_shape, n_categories,
                 n_filters=64, kernel_size=(3, 3), conv_padding='same',
                 spp_levels=(1, 2, 4, 8),  n_hidden=64, dropout_rate=0.5,
                 checkpoints_path='/tmp/', checkpoint_file_format='{epoch:02d}-{val_loss:.5f}.hdf5',
                 optimizer='adadelta', save_best_only=True):

        self.input_shape = input_shape
        self.n_categories = n_categories
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.conv_padding = conv_padding
        self.spp_levels = spp_levels
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate

        self.checkpoints_path = checkpoints_path
        self.checkpoint_file_format = checkpoint_file_format

        self.optimizer = optimizer
        self.save_best_only = save_best_only

        self._model = None

    def _build_model(self):
        """Build and compile the Keras model according to the specified parameters.
        """

        self._model = Sequential()

        self._model.add(Conv2D(self.n_filters // 2, self.kernel_size, padding=self.conv_padding, input_shape=self.input_shape))
        self._model.add(Activation('relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(self.n_filters // 2, self.kernel_size, padding=self.conv_padding))
        self._model.add(Activation('relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(self.n_filters, self.kernel_size, padding=self.conv_padding))
        self._model.add(Activation('relu'))

        if self.spp_levels:
            self._model.add(SpatialPyramidPooling2D(self.spp_levels))
        else:
            self._model.add(Flatten())

        self._model.add(Dense(self.n_hidden))
        self._model.add(Activation('relu'))

        self._model.add(Dropout(self.dropout_rate))
        self._model.add(Dense(self.n_categories))
        self._model.add(Activation('softmax'))

        self._model.compile(loss='categorical_crossentropy',
                            optimizer=self.optimizer,
                            metrics=['accuracy'])

    def _build_callbacks(self):
        """Build callback objects.

        Returns:
            A list containing the following callback objects:
                - TensorBoard
                - ModelCheckpoint
        """

        tensorboard_path = os.path.join(self.checkpoints_path, 'tensorboard')
        tensorboard = TensorBoard(log_dir=tensorboard_path)

        checkpoint_path = os.path.join(self.checkpoints_path, self.checkpoint_file_format)
        checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=self.save_best_only)

        return [tensorboard, checkpointer]

    def build(self):
        """Build model and print model summary.
        """
        self._build_model()
        self._model.summary()

    def load_from_file(self, path):
        """Loads Keras model from specified path.

        Args:
            path (str): Path to model file.
        """
        self._model = load_model(path)

    def fit_generator(self, train_generator, steps_per_epoch, epochs,
                      validation_generator, validation_steps):
        """Fits the model on data yielded from keras data generator.

        Args:
            train_generator: A data generator that yields (x, y) tuples of training data/labels.
            steps_per_epoch: Steps (number of batches) per epoch.
            epochs: Number of epochs.
            validation_generator: A data generator that yields (x, y) tuples of validation data/labels.
            validation_steps: Validation steps (number of batches).

        Returns:
            Keras History object with history of training losses.
        """

        callbacks = self._build_callbacks()

        history = self._model.fit_generator(generator=train_generator,
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=epochs,
                                            callbacks=callbacks,
                                            validation_data=validation_generator,
                                            validation_steps=validation_steps)

        return history

    def predict(self, img):
        """Predict using model given an input.

        Args:
            img (ndarray): Model input.
                This is a (batch of) RGB image of shape (samples, height, width, 3).

        Returns:
            Model prediction (ndarray). Predicted class probabilities.
                Array of shape (samples, n_categories)
        """

        # Input check
        if len(img.shape) != 4:
            raise ValueError('img must be a array of shape (samples, height, width, 1).')

        if img.shape[-1] != 1:
            raise ValueError('Last dimension of img must be of size 1.')

        if img.dtype != 'float32':
            raise ValueError('img dtype must be float32.')

        return self._model.predict(img)
