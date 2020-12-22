# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/50701913/how-to-split-the-input-into-different-channels-in-keras

from keras.layers import Layer


class SplitLayer(Layer):
    def __init__(self, sizes, target_dim=None, **kwargs):
        self.sizes = sizes
        self.target_dim = target_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_slice(self, number_of_dimentions, first, last):
        out = []
        for i in range(number_of_dimentions):
            if i == self.target_dim:
                out.append(slice(first, last))
            else:
                out.append(slice(None))
        return tuple(out)

    def call(self, input_layer):
        last = 0
        branches = []
        number_of_dimentions = len(input_layer.shape)
        for size in self.sizes:
            if isinstance(size, int):
                new = last+size
            else:
                new = None
            slice = self.get_slice(number_of_dimentions, last, new)
            branches.append(input_layer[slice])
            last = new
        return branches

    def compute_output_shape(self, input_shape):
        return [branch.shape for branch in self.call(numpy.zeros(input_shape))]

    def get_config(self):
        config = {"sizes": self.sizes,
                  "target_dim": self.target_dim}
        base_config = super().get_config()
        return config.update(base_config)
        #return dict(list(base_config.items()) + list(config.items()))


class DuplicateLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, input_layer):
        return [input_layer, input_layer]

    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]
