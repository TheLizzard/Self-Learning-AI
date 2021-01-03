import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Get the seed for the randomness to produce reproducible results
from constants import seed as _seed

from inspect import getsource, getframeinfo, stack
from inspect import getsource
from warnings import warn
import tensorflow as tf
from os import path
import pickle
import sys

from aibrain.ai import AI
from training.trainer import Trainer
from board.environment import Environment
from gui.plotwindow import ContinuousPlotWindow


class Logger:
    __slots__ = ("output", "items")

    def __init__(self, output=True):
        self.output = output
        self.items = []

    def append(self, item, force_output=False):
        self.items.append(item)
        if self.output or force_output:
            print(item)

    def dump(self, obj):
        padding = "=" * max(len(item) for item in self.items)
        obj.write(padding + "\n".join(self.items) + padding)


class App:
    def __init__(self, model, depth, ask_verify=True, sample_size=1000, **kwargs):
        self.ops = Logger()
        self.epoch = 0
        self.sample_size = sample_size

        self.AI = AI(model, ask_verify=ask_verify)
        self.AI.plot_model()
        self.trainer = Trainer(Environment, self.AI, depth=depth)

        self.plotwindow = ContinuousPlotWindow(fg="white", bg="black", geometry=(400, 400), dpi=100)
        self.plotwindow.set_xlabel("Number of games played", colour="white")
        self.plotwindow.set_ylabel("Loss", colour="white")
        self.plotwindow.set_title("sample_size = %s"%str(self.sample_size), colour="white")
        self.plotwindow.grid_lines(True, colour="grey", linestyle="--")
        self.plotwindow.set_format(colour="white", size=7)
        self.plotwindow.resize(300, 300)
        self.plotwindow.xlim(left=0)
        self.plotwindow.ylim(left=0)
        self.plotwindow.exit_when_done = True

    def set_seed(self, seed=42):
        _seed.set_seed(seed)

        self.ops.append("[debug] set_seed(%s)"%str(seed))

    def compile(self, **kwargs):
        self.trainer.compile(**kwargs)

        kwargs_text = self.dict_to_str(kwargs)
        self.ops.append("[debug] compile(%s)" % kwargs_text)

    def config(self, *args, **kwargs):
        text = self.list_to_str(args)
        if (len(args) != 0) and (len(kwargs) != 0):
            text += ", "
        text += self.dict_to_str(kwargs)
        self.ops.append("[debug] config(%s)"%text)

        return self.trainer.config(*args, **kwargs)

    def save(self, filename="saved/autosave.pcl"):
        _self = {}
        _self.update(self.__dict__)

        _self.update({"trainer": self.trainer.__getstate__(),
                      "losses": self.plotwindow.points})
        _self.pop("plotwindow")
        _self.pop("AI")

        encoded_data = pickle.dumps(_self)
        with open(filename, "wb") as file:
            file.write(encoded_data)

        self.ops.append("[debug] save(filename=%s)"%filename)

    def load(self, filename="saved/autosave.pcl", **kwargs):
        with open(filename, "rb") as file:
            encoded_data = file.read()
        _self = pickle.loads(encoded_data)
        losses = _self.pop("losses")
        self.__dict__.update(_self)

        self.trainer = Trainer(lambda: None, None, depth=None)
        self.trainer.__setstate__(_self["trainer"], **kwargs)
        self.AI = self.trainer.AI

        self.plotwindow.reset()
        for x, y in zip(*losses):
            self.plotwindow.add(x, y)

        text = "filename=" + str(filename) + ", " + self.dict_to_str(kwargs)
        self.ops.append("[debug] load(%s)"%text)

    def set_main(self, function):
        text = '"""\n'+getsource(function)+'"""'
        self.ops.append("[debug] run(%s)"%text)

        self.plotwindow.set_main(function)
        self.plotwindow.mainloop()

    def test(self, sample_size=None, debug=False):
        if sample_size is None:
            sample_size = self.sample_size
        loss = self.trainer.test(sample_size, debug=debug)
        if sample_size == self.sample_size:
            self.plotwindow.add(self.epoch, loss)
        else:
            warn("This test isn't going to be plotted as sample_size != "+str(self.sample_size))

        self.ops.append("[debug] test(sample_size=%s)[epoch=%s] => %s"%(str(sample_size), str(self.epoch), str(loss)))
    
    def test_all(self, debug=False):
        loss = self.trainer.test_all(debug=debug)
        if self.sample_size == "all":
            self.plotwindow.add(self.epoch, loss)
        else:
            warn("This test isn't going to be plotted as \"all\" != "+str(self.sample_size))

        self.ops.append("[debug] test_all()[epoch=%s] => %s"%(str(self.epoch), str(loss)))

    def train(self, worlds=1, epochs=1, debug=False):
        for epoch in range(epochs):
            for world in range(worlds):
                self.trainer.train(debug=debug)
            self.trainer.flush()
            self.epoch += 1

            self.ops.append("[debug] train(worlds=%s)"%str(worlds))

    def predict(self, *args, **kwargs):
        text = self.list_to_str(args)
        if (len(args) != 0) and (len(kwargs) != 0):
            text += ", "
        text += self.dict_to_str(kwargs)
        self.ops.append("[debug] predict(%s)"%text)

        return self.AI.predict(*args, **kwargs)

    def human_test(self, **kwargs):
        self.trainer.human_test(**kwargs)

    def exit(self, msg):
        dir, filename, lineno = self.get_caller()
        self.ops.append("exit(%s)[filename=%s, lineno=%s, dir=%s]"%(msg, filename, lineno, dir))
        self.ops.append(msg, force_output=True)

        exit()

    def dump_debug(self):
        self.ops.dump(sys.stdout)

    @staticmethod
    def dict_to_str(_dict):
        items = [str(key)+"="+str(value) for key, value in _dict.items()]
        return ", ".join(items)

    @staticmethod
    def list_to_str(_list):
        return ", ".join(_list)

    @staticmethod
    def get_caller():
        caller = getframeinfo(stack()[2][0])
        location = caller.filename
        filename = path.basename(location)
        dir = path.dirname(path.dirname(location))
        return dir, filename, caller.lineno


if __name__ == "__main__":
    dense = [{"type": "flatten"},
             {"type": "dense", "size": 100},
             {"type": "dense", "size": 500},
             {"type": "dense", "size": 500},
             {"type": "dense", "size": 500},
             {"type": "dense", "size": 500}]

    conv = [{"type": "resize", "shape": (3, 3, 3, 1)},
            {"type": "conv3d", "filters": 64, "filter": (3, 3, 2), "padding": "same"},
            {"type": "pool3d", "filter": (2, 2, 1), "padding": "same"},
            {"type": "flatten"},
            {"type": "dense", "size": 500},
            {"type": "dense", "size": 500}]

    policy = [{"type": "dense", "size": 250},
              {"type": "dense", "size": 100},
              {"type": "dense", "size": 9},
              {"type": "softmax", "name": "policy"}]

    value = [{"type": "dense", "size": 250},
             {"type": "dense", "size": 100},
             {"type": "dense", "size": 10},
             {"type": "dense", "size": 1, "activation": "tanh", "name": "value"}]

    model = [{"type": "input", "shape": (3, 3, 3)},
             {"type": "duplicate"},
             [conv, dense],
             {"type": "dense", "size": 1000},
             {"type": "dense", "size": 1000},
             {"type": "dense", "size": 500},
             {"type": "dropout", "rate": 0.1},
             {"type": "split", "sizes": (250, 250), "target_dim": 1},
             [policy, value]]

    @tf.function(experimental_compile=True)
    def loss_function_value(true, pred):
        return tf.reduce_sum(tf.pow(true-pred, 2))

    @tf.function(experimental_compile=True)
    def loss_function_policy(true, pred):
        return tf.math.negative(tf.nn.softmax_cross_entropy_with_logits(pred, true))

    def main():
        #app.load(custom_objects=custom_objects, compile=True)
        #app.test(10, debug=True)
        #app.test_all()
        #app.human_test(depth=2, debug=True)
        #exit()

        #app.config(method="optimizer.learning_rate.assign", args=(1e-5, ), kwargs={})
        for epoch in range(50):
            app.train(worlds=10, epochs=10)
            app.test_all()#debug=True
            if app.epoch%5 == 0:
                app.save()
        app.save()
        app.exit("End of main function.")

    loss_dict = {"value": loss_function_value,
                 "policy": loss_function_policy}
    custom_objects = {"loss_function_value": loss_function_value,
                      "loss_function_policy": loss_function_policy}

    app = App(model, depth=10, sample_size="all", ask_verify=True, debug=True)
    app.compile(loss=loss_dict, learning_rate=1e-5)
    app.set_main(main)
    app.exit("End of code.")