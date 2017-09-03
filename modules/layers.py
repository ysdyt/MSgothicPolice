from itertools import product
from keras.engine.topology import Layer
from keras import backend as K


class SpatialPyramidPooling2D(Layer):
    """2D Spatial Pyramid Pooling.

    Currently, only supports Tensorflow format.

    Args:
        nb_bins_per_level (tuple): Number of bins into which each of the axes of the input is partitioned
            for pooling (per channel). Each element corresponds to a level of the pyramid.

            e.g. if [1, 2, 4] then each channel is pooled over
            1 (=1**2) + 4 (=2**2) + 16 (=4**2) = 21 regions in 3 levels,
            producing 21*nb_channels features.

    # Input shape
        4D tensor with shape:
        `(samples, dim1, dim2, channels)`
    # Output shape
        2D tensor with shape:
        `(samples, nb_features)` where nb_features is equal to
        nb_channels * sum([nb_bins**2 for nb_bins in nb_bins_per_layer])
    """

    def __init__(self, nb_bins_per_level, **kwargs):
        self.nb_bins_per_level = nb_bins_per_level
        self.nb_channels = None
        super(SpatialPyramidPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[3]
        super(SpatialPyramidPooling2D, self).build(input_shape)

    def get_config(self):
        config = {'nb_bins_per_level': self.nb_bins_per_level}
        base_config = super(SpatialPyramidPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):

        nb_samples = input_shape[0]
        nb_outputs_per_channel = sum([nb_bins ** 2 for nb_bins in self.nb_bins_per_level])
        nb_features = self.nb_channels * nb_outputs_per_channel

        return nb_samples, nb_features

    def call(self, x):

        input_shape = K.shape(x)

        len_i, len_j = input_shape[1], input_shape[2]

        outputs = []

        for nb_bins in self.nb_bins_per_level:
            bin_size_i = K.cast(len_i, 'int32') // nb_bins
            bin_size_j = K.cast(len_j, 'int32') // nb_bins

            for i, j in product(range(nb_bins), range(nb_bins)):
                # each combination of i,j is a unique rectangle
                i1, i2 = bin_size_i * i, bin_size_i * (i + 1)
                j1, j2 = bin_size_j * j, bin_size_j * (j + 1)

                pooled_features = K.max(x[:, i1:i2, j1:j2, :], axis=(1, 2))

                outputs.append(pooled_features)

        return K.concatenate(outputs, axis=1)
