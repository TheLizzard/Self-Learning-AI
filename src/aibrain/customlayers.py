# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/50701913/how-to-split-the-input-into-different-channels-in-keras

from keras.layers import Layer
from copy import deepcopy
import numpy as np


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
        input_shape = (10, )+input_shape[1:]
        output = self.call(np.zeros(input_shape))
        output_shape = [(None, )+branch.shape[1:] for branch in output]
        return output_shape

    def get_config(self):
        config = {"sizes": self.sizes,
                  "target_dim": self.target_dim}
        base_config = super().get_config()
        config.update(base_config)
        return config


class DuplicateLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, input_layer):
        return [input_layer, input_layer]

    def compute_output_shape(self, input_shape):
        return [deepcopy(input_shape), deepcopy(input_shape)]
