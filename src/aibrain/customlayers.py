# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/50701913/how-to-split-the-input-into-different-channels-in-keras

from keras.layers import Layer


class SplitLayer(Layer):
    def __init__(self, loc, target_dim=None, **kwargs):
        self.target_dim = target_dim
        self.loc = loc
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, input_layer):
        #shape = list(input_layer.get_shape())
        #dim = len(shape)
        last = 0
        branch = []
        for j in self.loc:
            branch.append(self.calc_output(input_layer, self.target_dim,
                                           first=last, second=last+j))
            last += j
        return branch

    def get_slice(self, dim, target, first, last):
        out = []
        for i in range(dim):
            if i+1 == target:
                out.append(slice(first, last))
            else:
                out.append(slice(None))
        return tuple(out)

    def calc_output(self, input_layer, dim, first, second):
        return input_layer[self.get_slice(dim, self.target_dim, first, second)]
        # Depricated Code:
        """
        self.dim = dim
        if dim == 2:
            if self.target_dim == 2:
                return input_layer[:, first:second]
        elif dim == 3:
            if self.target_dim == 2:
                return input_layer[:, first:second, :]
            elif self.target_dim == 3:
                return input_layer[:, :, first:second]
        elif dim == 4:
            if self.target_dim == 2:
                return input_layer[:, first:second, :, :]
            elif self.target_dim == 3:
                return input_layer[:, :, first:second, :]
            elif self.target_dim == 4:
                return input_layer[:, :, :, first:second]
        elif dim == 5:
            if self.target_dim == 2:
                return input_layer[:, first:second, :, :, :]
            elif self.target_dim == 3:
                return input_layer[:, :, first:second, :, :]
            elif self.target_dim == 4:
                return input_layer[:, :, :, first:second, :]
            elif self.target_dim == 5:
                return input_layer[:, :, :, :, first:second]
        elif dim == 6:
            if self.target_dim == 2:
                return input_layer[:, first:second, :, :, :, :]
            elif self.target_dim == 3:
                return input_layer[:, :, first:second, :, :, :]
            elif self.target_dim == 4:
                return input_layer[:, :, :, first:second, :, :]
            elif self.target_dim == 5:
                return input_layer[:, :, :, :, first:second, :]
            elif self.target_dim == 6:
                return input_layer[:, :, :, :, :, first:second]
        """

    def compute_output_shape(self, input_shape):
        branch = []
        for j in self.loc:
            shape = list(input_shape)
            shape[self.target_dim-1] = j
            branch.append(tuple(shape))
        return branch

    def get_config(self):
        config = {"loc": self.loc,
                  "dim": self.dim,
                  "target_dim": self.target_dim}
        base_config = super(SplitLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DuplicateLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, input_layer):
        return [input_layer, input_layer]

    def calc_output(self, input_layer):
        return [input_layer, input_layer]
