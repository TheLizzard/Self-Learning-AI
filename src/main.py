import tensorflow as tf

from training.trainer import Trainer
from environment import Environment
from aibrain.ai import AI

from test_creator import test_creator


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

    core = AI(model, loss=loss_dict, learning_rate=0.001, ask_verify=True)
    trainer = Trainer(Environment, core)

    for i in range(100):
        for epoch in range(10):
            trainer.train()
            trainer.flush()
            print("Done 1 more.")
            input(">>> ")
        trainer.test()
