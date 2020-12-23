# -*- coding: utf-8 -*-

"""
================================================================================
                                 Layers Shorthand:
================================================================================
conv(filter: tuple<int, int>, filters=1, strides: {strides},
     padding: {padding}, activation="selu", use_bias=True)

conv3d(filter: tuple<int, int, int>, filters=1, strides: {strides},
       padding: {padding}, activation="selu", use_bias=True)

pool3d(filter: tuple<int, int, int>, strides: {strides}, padding: {padding})

pool(filter: tuple<int, int>, strides: {strides}, padding: {padding})

dense(size: int, activation="selu", use_bias=True)

dropout(rate: float, noise_shape=None, seed=None)

zero_padding(padding_size={padding_size})

split(sizes: tuple<ints>, target_dim: int)

reshape(shape: tuple<ints>)

gaussian_noise(noise=0.01)

activation(function: str)

input(shape: tuple<ints>)

softmax()

flatten()

duplicate()


Arguments description:
----------------------
{padding}: str
    Can only be one of: ("valid", # No padding
                         "same",  # The pattern is repeated in all directions)

{strides}: int/tuple<ints>
    An integer or tuple/list of 3 integers, specifying the strides of the
    convolution along the height and width. Can be a single integer to
    specify the same value for all spatial dimensions.

{padding_size}: int/tuple<ints>
    * In the case of a int: the same symmetric padding is applied to height and
    width.
    * In the case of 2 ints: interpreted as two different symmetric padding
    values for height and width: (symmetric_height_pad, symmetric_width_pad).
    * In the case of 2 tuples of 2 ints: interpreted as ((top_pad, bottom_pad),
    (left_pad, right_pad))

================================================================================
                   Example 1 (99.12% accuracy):

                 This is the neural network model:
                 ---------------------------------
                               Input
                                 ↓
                               Reshape
                                 ↓
                               Conv2D
                                 ↓
                               Pool2D
                                 ↓
                               Conv2D
                                 ↓
                               Pool2D
                                 ↓
                               Flatten
                                 ↓
                               Dropout
                                 ↓
                               Dense
                                 ↓
                               Softmax
                                 ↓
                               Output
================================================================================
# Set the seed (to make results reproducible):
from constants.set_seed import set_seed
set_seed(42)

from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np

# Read the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255 # Normalise the data
x_test = x_test/255   # Normalise the data

y_train = to_categorical(y_train, 10) # Convert it into 1 hot
y_test = to_categorical(y_test, 10)   # Convert it into 1 hot

model = [{"type": "input", "shape": (28, 28)},
         {"type": "resize", "shape": (28, 28, 1)},
         {"type": "conv", "filters": 32, "filter": (3, 3), "activation": "relu"},
         {"type": "pool", "filter": (2, 2)}, # Do a PoolMax2D
         {"type": "conv", "filters": 16, "filter": (3, 3), "activation": "relu"},
         {"type": "pool", "filter": (2, 2)}, # Do a PoolMax2D
         {"type": "flatten"},
         {"type": "dropout", "rate": 0.5},
         {"type": "dense", "size": 10},
         {"type": "softmax"}] # Apply softmax as the data is categorical

core = AICore(model, loss="categorical_crossentropy", learning_rate=0.001)
try:
    # Train the neural net on the data
    core.train(x_train, y_train, batch_size=100, epochs=15,
               validation_data=(x_test, y_test))
except KeyboardInterrupt as error:
    # If the user presses Ctrl+c: evaluate the model quickly and exit
    pass

# Evaluate the model
score = core.model.evaluate(x_test, y_test, verbose=1)
print("Loss:", score[0])
print("Accuracy %s%%"%(str(score[1]*100+0.05/10)[:5])) # Round the accuracy
================================================================================
               Example 1 Improved (99.13% accuracy):

                 This is the neural network model:
                 ---------------------------------
                                Input
                                /   \
                               /     \
                              /       \
                             /         \
                        Reshape        Flatten
                           ↓              ↓
                        Conv2D          Dense
                           ↓              ↓
                        Pool2D          Dense
                           ↓              ↓
                        Conv2D          Dense
                           ↓              /
                        Pool2D           /
                           ↓            /
                        Flatten        /
                              \       /
                               Dropout
                                  ↓
                                Dense
                                  ↓
                                Dense
                                  ↓
                               Softmax
                                  ↓
                                Output
================================================================================
# Set the seed (to make results reproducible):
from constants.set_seed import set_seed
set_seed(42)

from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np

# Read the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255 # Normalise the data
x_test = x_test/255   # Normalise the data

y_train = to_categorical(y_train, 10) # Convert it into 1 hot
y_test = to_categorical(y_test, 10)   # Convert it into 1 hot


conv = [{"type": "resize", "shape": (28, 28, 1)}, # Reshape
        {"type": "conv", "filters": 32, "filter": (3, 3), "activation": "relu"},
        {"type": "pool", "filter": (2, 2)}, # Do a PoolMax2D
        {"type": "conv", "filters": 16, "filter": (3, 3), "activation": "relu"},
        {"type": "pool", "filter": (2, 2)}, # Do a PoolMax2D
        {"type": "flatten"}] # Flatten it to make it easier to merge it

dense = [{"type": "flatten"}, # Dense layers only take flat layers as inputs
         {"type": "dense", "size": 400},
         {"type": "dense", "size": 200},
         {"type": "dense", "size": 100}]

model = [{"type": "input", "shape": (28, 28)},
         {"type": "duplicate"}, # Duplicate the layer and pass it into
         [conv, dense],    # 1 conv and 1 dense network
         {"type": "dropout", "rate": 0.5}, # A dropout layer
         {"type": "dense", "size": 100},
         {"type": "dense", "size": 10},
         {"type": "softmax"}] # Apply softmax as the data is categorical


core = AICore(model, loss="categorical_crossentropy", learning_rate=0.001)
try:
    # Train the neural net on the data
    core.train(x_train, y_train, batch_size=100, epochs=15,
               validation_data=(x_test, y_test))
except KeyboardInterrupt as error:
    # If the user presses Ctrl+c: evaluate the model quickly and exit
    pass

# Evaluate the model
score = core.model.evaluate(x_test, y_test, verbose=1)
print("Loss:", score[0])
print("Accuracy %s%%"%(str(score[1]*100+0.05/10)[:5])) # Round the accuracy

================================================================================
                                     Layers:
================================================================================
dropout
    rate (float)                   Between 0 and 1. Fraction of the input units to drop.
    noise_shape (tensor<1D>)       1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. For instance, if your inputs have shape (batch_size, timesteps, features)
                                   and you want the dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features).
    seed (int)                     A random seed.

conv
    filters (int=1)                The number of filters to have in that convolutional layer
    strides (int=1)                An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions.
    filter (int/list/tuple)        An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
    padding (str="zero")           One of ("valid", "same", "zero") (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same
                                   height/width dimension as the input. "zero" results in a padding of 0s in all directions
    activation (str="selu")        Activation function to use. To not use an activation function use `None`
    use_bias (bool=True)

conv3d
    filters (int=1)                The number of filters to have in that convolutional layer
    strides (int=1)                An integer or tuple/list of 3 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions.
    filter (int/list/tuple)        An integer or tuple/list of 3 integers, specifying the height and width of the 3D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
    padding (str="zero")           One of ("valid", "same", "zero") (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the
                                   same height/width dimension as the input. "zero" results in a padding of 0s in all directions
    activation (str="selu")        Activation function to use. To not use an activation function use `None`
    use_bias (bool=True)

pool
    filter (int/tuple<2 ints>)     Window size over which to take the maximum. (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.
    strides (int=1)                An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions.
    padding (str="zero")           One of ("valid", "same", "zero") (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same
                                   height/width dimension as the input. "zero" results in a padding of 0s in all directions. If "zero" is used, padding_size must be defined (look at zero_padding)

pool3d
    filter (int/tuple<3 ints>)     Window size over which to take the maximum. (2, 2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.
    strides (int=1)                An integer or tuple/list of 3 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions.
    padding (str="zero")           One of ("valid", "same", "zero") (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same
                                   height/width dimension as the input. "zero" results in a padding of 0s in all directions. If "zero" is used, padding_size must be defined (look at zero_padding)

zero_padding
    padding (int/tuple)            One of (int, tuple of 2 ints, tuple of 2 tuples of 2 ints)
                                     In the case of a int: the same symmetric padding is applied to height and width.
                                     In the case of 2 ints: interpreted as two different symmetric padding values for height and width: (symmetric_height_pad, symmetric_width_pad).
                                     In the case of 2 tuples of 2 ints: interpreted as ((top_pad, bottom_pad), (left_pad, right_pad))

reshape
    size (tuple<ints>)             How to reshape the tensor

split
    sizes (tuple<ints>)            The sizes of the tensors after the split. So split(loc=(2, 1))([14, 15, 16]) => [14, 15], [16]
    target_dim (int)               The dimention in which we are splitting

dense
    size (int)                     The number of outputs
    activation (str="selu")        The activation function to be applied
    use_bias (bool=True)           If the layer should use a bias.

gaussian_noise
    noise (float=0.01)

activation
    function (str)                 The activation function to be applied.

softmax

flatten

softmax
================================================================================
"""

# Add Graphviz so that keras doesn't fail
import os
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz 2.44.1/bin/"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
optimisations = {"layout_optimizer": True,
                 "constant_folding": True,
                 "shape_optimization": True,
                 "remapping": True,
                 "arithmetic_optimization": True,
                 "dependency_optimization": True,
                 "loop_optimization": True,
                 "function_optimization": True,
                 "debug_stripper": True,
                 "disable_model_pruning": True,
                 "scoped_allocator_optimization":True,
                 "pin_to_host_optimization": True,
                 "implementation_selector": True,
                 "min_graph_nodes": 0}
try: # Turn on all optimisations
    tf.config.optimizer.set_experimental_options(optimisations)
except:
    warnings.warn("Couldn't turn on optimisations.")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "100"

from keras.layers import Dense, Conv2D, Conv3D, Flatten, Input, Activation
from keras.layers import Reshape, MaxPool3D, ZeroPadding2D, Dropout
from keras.layers import Concatenate, GaussianNoise, MaxPool2D
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.optimizers import Adam

import numpy as np
import tempfile
import zipfile
import pickle
import copy
import sys

from .customlayers import SplitLayer, DuplicateLayer


class AICore:
    def __init__(self, modeldict, learning_rate=0.001, optimizer=None,
                 loss="categorical_crossentropy", load=False, ask_verify=True):
        if load == False:
            self.modeldict = modeldict
            self.learning_rate = learning_rate
            self.loss = loss
            if optimizer is None:
                self.optimizer = Adam(learning_rate=self.learning_rate)
            else:
                self.optimizer = optimizer
            self.init_neural_network()
            if ask_verify:
                self.ask_verify_model()
        else:
            if load != True:
                load = "save/autosave.sav"
            self.load(load, ask_verify)

    def __getstate__(self):
        _self = copy.deepcopy(self.__dict__)
        keys_to_pop = []
        for key, value in _self.items():
            if key == "model":
                _self[key] = self.get_model_state()
            else:
                try:
                    _ = pickle.dumps(value)
                except:
                    sys.stderr.write(("Warning: Couldn't save \"%s\" "+\
                                      "attribute.\n")%key)
                    keys_to_pop.append(key)
        for key in keys_to_pop:
            _self.pop(key)
        return _self

    def get_model_state(self):
        with tempfile.TemporaryDirectory() as basefolder:
            self.save(basefolder)
            zipfilename = basefolder+"\\files.zip"
            filesystem = zipfile.ZipFile(zipfilename, "x", zipfile.ZIP_STORED)
            rootlen = len(basefolder) + 1
            for base, subdirs, files in os.walk(basefolder):
                for file in files:
                    if file == "files.zip":
                        continue
                    filename = os.path.join(base, file)
                    filesystem.write(filename, filename[rootlen:])
            filesystem.close()
            with open(zipfilename, "br") as file:
                return file.read()

    def __setstate__(self, _self):
        for key, value in _self.items():
            if key == "model":
                self.set_model_state(value)
            else:
                setattr(self, key, value)

    def set_model_state(self, state):
        with tempfile.TemporaryDirectory() as basefolder:
            zipfilename = basefolder+"\\files.zip"
            with open(zipfilename, "bw") as file:
                file.write(state)

            filesystem = zipfile.ZipFile(zipfilename, "r", zipfile.ZIP_STORED)
            for member in filesystem.infolist():
                filefolder = os.path.join(basefolder, member.filename)
                abspath = os.path.realpath(filefolder)
                if abspath.startswith(os.path.realpath(basefolder)):
                    filesystem.extract(member, basefolder)
            filesystem.close()

            self.load(basefolder, ask_verify=False)

    def save(self, location="save/autosave.sav"):
        self.model.save(location)

    def load(self, location="save/autosave.sav", ask_verify=True):
        custom_objects = {"SplitLayer": SplitLayer,
                          "DuplicateLayer": DuplicateLayer}
        self.model = load_model(location, custom_objects=custom_objects)
        if ask_verify:
            self.ask_verify_model()

    def ask_verify_model(self):
        self.model.summary()
        self.plot_model()
        if input("? ") != "":
            exit()
        print("\n"*3)

    def plot_model(self, file="tmp/graph.png", dpi=300, shapes=True):
        try:
            plot_model(self.model, to_file=file, expand_nested=True,
                       dpi=dpi, show_shapes=shapes)
        except ImportError as error:
            print(error, file=sys.stderr)
            print("Couldn't save an image of the model because "\
                  "\"import pydot\" failed", file=sys.stderr)
        except OSError as error:
            print(error, file=sys.stderr)
            print("Couldn't save an image of the model because pydot"\
                  "failed to call GraphViz", file=sys.stderr)

    def init_neural_network(self):
        assert self.modeldict[0]["type"] == "input", "First layer has to "+\
                                            "be an input layer"
        input_shape = self.modeldict[0]["shape"]
        input_layer = Input(shape=input_shape)
        last_layer = input_layer

        for layer in self.modeldict[1:]:
            last_layer = self.add_layer(last_layer, layer)

        kwargs = {"optimizer": self.optimizer,
                  "loss": self.loss}
        self.model = Model(inputs=input_layer, outputs=last_layer)
        self.model.compile(**kwargs)
        self.model.build(input_shape=input_shape)

    def add_layer(self, input_layer, this):
        if isinstance(this, list):
            assert isinstance(input_layer, list), "Internal Error."
            input_layers = input_layer
            for i, split in enumerate(this):
                for layer in split:
                    input_layers[i] = self.add_layer(input_layers[i], layer)
            return input_layers

        if isinstance(input_layer, list):
            input_layer = self.add_merge(input_layer)

        layer_type = this.pop("type")
        if layer_type == "merge":
            return input_layer
        elif layer_type == "dropout":
            return self.add_dropout_layer(input_layer, **this)
        elif layer_type == "conv":
            return self.add_conv_layer(input_layer, **this)
        elif layer_type == "conv3d":
            return self.add_conv3d_layer(input_layer, **this)
        elif layer_type == "pool":
            return self.add_pool(input_layer, **this)
        elif layer_type == "pool3d":
            return self.add_pool3d(input_layer, **this)
        elif layer_type == "dense":
            return self.add_dense_layer(input_layer, **this)
        elif layer_type == "flatten":
            return self.add_flatten(input_layer, **this)
        elif layer_type == "split":
            return self.add_split(input_layer, **this)
        elif (layer_type == "reshape") or (layer_type == "resize"):
            return self.add_reshape(input_layer, **this)
        elif layer_type == "activation":
            return self.add_activation(input_layer, **this)
        elif layer_type == "softmax":
            return self.add_softmax(input_layer, **this)
        elif type == "gaussian_noise":
            return self.add_gaussian_noise(input_layer, **this)
        elif layer_type == "duplicate":
            return self.add_duplicate(input_layer)
        elif layer_type == "zero_padding":
            return self.add_zero_padding(input_layer, **this)
        else:
            msg = "This layer hasn't been implemented: "+str(layer_type)
            raise NotImplementedError(msg)

    def add_dropout_layer(self, input_layer, rate, noise_shape=None, seed=None, name=None):
        return Dropout(rate, noise_shape=noise_shape, seed=seed, name=name)(input_layer)

    def add_dense_layer(self, input_layer, size, activation="selu", use_bias=True, name=None):
        return Dense(units=size, activation=activation, use_bias=use_bias, name=name)(input_layer)

    def add_conv_layer(self, input_layer, filter, filters=1, use_bias=True, strides=1, padding="valid", activation="selu", name=None):
        return Conv2D(kernel_size=filter, filters=filters, strides=strides, activation=activation, padding=padding, name=name)(input_layer)

    def add_conv3d_layer(self, input_layer, filter, filters=1, use_bias=True, strides=1, padding="valid", activation="selu", name=None):
        return Conv3D(kernel_size=filter, filters=filters, strides=strides, activation=activation, padding=padding, name=name)(input_layer)

    def add_pool(self, input_layer, filter, strides=None, padding="valid", name=None):
        return MaxPool2D(pool_size=filter, strides=strides, padding=padding, name=name)(input_layer)

    def add_pool3d(self, input_layer, filter, strides=None, padding="valid", name=None):
        return MaxPool3D(pool_size=filter, strides=strides, padding=padding, name=name)(input_layer)

    def add_flatten(self, input_layer, name=None):
        return Flatten(name=name)(input_layer)

    def add_reshape(self, input_layer, shape, name=None):
        return Reshape(shape, name=name)(input_layer)

    def add_split(self, input_layer, sizes, target_dim):
        branches = SplitLayer(sizes=sizes, target_dim=target_dim)(input_layer)
        return branches

    def add_duplicate(self, input_layer):
        return DuplicateLayer()(input_layer)

    def add_merge(self, sub_layers):
        return Concatenate()(sub_layers)

    def add_gaussian_noise(self, input_layer, noise=0.01, name=None):
        return GaussianNoise(noise=noise, name=name)(input_layer)

    def add_activation(self, input_layer, function, name=None):
        return Activation(activation=function, name=name)(input_layer)

    def add_softmax(self, input_layer, name=None):
        return self.add_activation(input_layer, function="softmax", name=name)

    def add_zero_padding(self, input_layer, padding_size=None, name=None):
        return ZeroPadding2D(padding=padding_size, name=name)(input_layer)

    def train(self, questions, answers=None, **kwargs):
        return self.model.fit(questions, answers, **kwargs)

    def predict(self, questions):
        return self.model.predict(questions)

    def evaluate(self, questions, answers, **kwargs):
        return self.model.evaluate(questions, answers, **kwargs)

    def deepcopy(self):
        return copy.deepcopy(self)
