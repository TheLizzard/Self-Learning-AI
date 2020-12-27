# Get the seed for the randomness to produce reproducible results
from constants.set_seed import set_seed
set_seed(42)

from warnings import warn
import tensorflow as tf
import pickle

from aibrain.ai import AI
from training.trainer import Trainer
from board.environment import Environment
from gui.plotwindow import ContinuousPlotWindow


class App:
    def __init__(self, model, custom_objects={}, ask_verify=True, sample_size=1000, **kwargs):
        self.idx = 1
        self.sample_size = sample_size

        ai = AI(model, ask_verify=ask_verify, custom_objects=custom_objects)
        ai.plot_model()
        self.trainer = Trainer(Environment, ai)

        self.plotwindow = ContinuousPlotWindow(fg="white", bg="black", geometry=(400, 400), dpi=100)
        self.plotwindow.set_xlabel("Epoch number", colour="white")
        self.plotwindow.set_ylabel("Loss", colour="white")
        self.plotwindow.set_title("sample_size = %s"%str(self.sample_size), colour="white")
        self.plotwindow.set_format(colour="white", size=7)
        self.plotwindow.xlim(left=0)
        self.plotwindow.ylim(left=0)
        self.plotwindow.exit_when_done = True

    def compile(self, **kwargs):
        self.trainer.compile(**kwargs)

    def save(self, filename="saved/autosave.pcl"):
        _self = {}
        _self.update(self.__dict__)
        _self.pop("plotwindow")

        trainer = pickle.dumps(self.trainer.__getstate__())
        _self.update({"trainer": trainer, "losses": self.plotwindow.points[1]})

        encoded_data = pickle.dumps(_self)
        with open(filename, "wb") as file:
            file.write(encoded_data)

    def load(self, filename="saved/autosave.pcl", custom_objects={}):
        with open(filename, "rb") as file:
            encoded_data = file.read()
        _self = pickle.loads(encoded_data)
        state = pickle.loads(_self.pop("trainer"))
        self.trainer.__setstate__(state, custom_objects=custom_objects)
        losses = _self.pop("losses")
        self.__dict__.update(_self)
        self.plotwindow.reset()
        for x, y in enumerate(losses, start=1):
            self.plotwindow.add(x, y)

    def set_main(self, function):
        self.plotwindow.set_main(function)
        self.plotwindow.mainloop()

    def test(self, sample_size=None, debug=False):
        if sample_size is None:
            sample_size = self.sample_size
        loss = self.trainer.test(sample_size, debug=debug)
        if sample_size == self.sample_size:
            print("[debug]  testing_loss = "+str(loss))
            self.plotwindow.add(self.idx, loss)
            self.idx += 1
        else:
            print("[debug]  testing_loss = "+str(loss))+"\tsample_size = "+str(sample_size)
            warn("This test isn't going to be plotted as sample_size != "+str(self.sample_size))
    
    def test_all(self, debug=False):
        loss = self.trainer.test_all(debug=debug)
        if self.sample_size == "all":
            print("[debug]  testing_loss="+str(loss))
            self.plotwindow.add(self.idx, loss)
            self.idx += 1
        else:
            print("[debug]  testing_loss = "+str(loss))+"\tsample_size = \"all\""
            warn("This test isn't going to be plotted as \"all\" != "+str(self.sample_size))

    def train(self, worlds=1, debug=False):
        for world in range(worlds):
            self.trainer.train(debug=debug)
        self.trainer.flush()


if __name__ == "__main__":
    dense = [{"type": "flatten"},
             {"type": "dense", "size": 100},
             {"type": "dense", "size": 500},
             {"type": "dense", "size": 250}]

    conv = [{"type": "resize", "shape": (3, 3, 3, 1)},
            {"type": "conv3d", "filters": 32, "filter": (3, 3, 2), "padding": "same"},
            {"type": "flatten"},
            {"type": "dense", "size": 250}]

    policy = [{"type": "dense", "size": 250},
              {"type": "dense", "size": 50},
              {"type": "dense", "size": 9},
              {"type": "softmax", "name": "policy"}]

    value = [{"type": "dense", "size": 250},
             {"type": "dense", "size": 75},
             {"type": "dense", "size": 10},
             {"type": "dense", "size": 1, "activation": "tanh", "name": "value"}]

    model = [{"type": "input", "shape": (3, 3, 3)},
             {"type": "duplicate"},
             [conv, dense],
             {"type": "dense", "size": 500},
             {"type": "dropout", "rate": 0.5},
             {"type": "split", "sizes": (250, 250), "target_dim": 1},
             [policy, value]]

    @tf.function(experimental_compile=True)
    def loss_function_value(true, pred):
        return tf.reduce_sum(tf.pow(true-pred, 2))

    @tf.function(experimental_compile=True)
    def loss_function_policy(true, pred):
        return tf.math.negative(tf.nn.softmax_cross_entropy_with_logits(pred, true))

    loss_dict = {"value": loss_function_value,
                 "policy": loss_function_policy}

    def main():
        #app.load(custom_objects={"loss_function_value": loss_function_value, "loss_function_policy": loss_function_policy})
        #app.compile(loss=loss_dict, learning_rate=1e-4)
        print("[debug]  starting_test(<BaseTest>)")
        app.test_all()
        print("[debug]  test_ended(<BaseTest>)")
        for epoch in range(100):
            print("[debug]  starting_epoch(%s)"%str(epoch))
            app.train(worlds=5)
            print("[debug]  epoch_ended(%s)"%str(epoch))
            print("[debug]  starting_test(%s)"%str(epoch))
            app.test_all()#debug=True
            print("[debug]  test_ended(%s)"%str(epoch))
            if epoch%5 == 0:
                app.save()
        app.save()

    app = App(model, custom_objects={"loss":loss_dict}, sample_size="all", ask_verify=False)
    app.compile(loss=loss_dict, learning_rate=1e-4)
    app.set_main(main)