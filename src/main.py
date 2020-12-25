# Get the seed for the randomness to produce reproducible results
from constants.set_seed import set_seed
set_seed(42)

import tensorflow as tf

from aibrain.ai import AI
from training.trainer import Trainer
from board.environment import Environment
from gui.plotwindow import ContinuousPlotWindow

#import sys
#sys.setrecursionlimit(10000)


class App:
    def __init__(self, model, **kwargs):
        self.idx = 1

        self.AI = AI(model, **kwargs)
        self.AI.plot_model()
        self.trainer = Trainer(Environment, self.AI)

        self.plotwindow = ContinuousPlotWindow(fg="white", bg="black", geometry=(400, 400), dpi=100)
        self.plotwindow.set_xlabel("The epoch", colour="white")
        self.plotwindow.set_ylabel("The loss", colour="white")
        self.plotwindow.set_title("The AI loss from the tests", colour="white")
        self.plotwindow.set_format(colour="white", size=7)
        self.plotwindow.xlim(left=0)
        self.plotwindow.ylim(left=0)

    def set_main(self, function):
        self.plotwindow.set_main(function)
        self.plotwindow.mainloop()

    def test(self, sample_size=1000):
        loss = self.trainer.test(sample_size)
        print("[debug]  testing_loss="+str(loss))
        self.plotwindow.add(self.idx, loss)
        self.idx += 1

    def train(self, worlds=1):
        for world in range(worlds):
            self.trainer.train()
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

    @tf.function
    def loss_function_value(true, pred):
        return tf.reduce_sum(tf.pow(true-pred, 2))

    @tf.function
    def loss_function_policy(true, pred):
        return tf.math.negative(tf.nn.softmax_cross_entropy_with_logits(pred, true))

    loss_dict = {"value": loss_function_value,
                 "policy": loss_function_policy}

    def main():
        print("[debug]  starting_test(<BaseTest>)")
        app.test()
        print("[debug]  test_ended(<BaseTest>)")
        for epoch in range(100):
            print("[debug]  starting_epoch(%s)"%str(epoch))
            app.train()
            print("[debug]  epoch_ended(%s)"%str(epoch))
            print("[debug]  starting_test(%s)"%str(epoch))
            app.test()
            print("[debug]  test_ended(%s)"%str(epoch))

    app = App(model, loss=loss_dict, learning_rate=0.00001, ask_verify=False)
    app.set_main(main)