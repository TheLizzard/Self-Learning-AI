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
              {"type": "softmax"}]

    value = [{"type": "dense", "size": 250},
             {"type": "dense", "size": 75},
             {"type": "dense", "size": 10},
             {"type": "dense", "size": 1, "activation": "tanh"}]

    model = [{"type": "input", "shape": (3, 3, 3)},
             {"type": "duplicate"},
             [conv, dense],
             {"type": "dense", "size": 500},
             {"type": "dropout", "rate": 0.5},
             {"type": "split", "sizes": (250, 250), "target_dim": 1},
             [policy, value]]

    def loss_function(true, pred):
        p_true, v_true = true
        p_pred, v_pred = pred
        return tf.reduce_sum(tf.pow(v_true-v_pred,2)) - tf.nn.softmax_cross_entropy_with_logits(p_pred, p_true)

    core = AI(model, loss=loss_function, learning_rate=0.000001, ask_verify=False)
    trainer = Trainer(Environment, core)

    for i in range(100):
        for epoch in range(100000):
            trainer.train()
            trainer.flush()
        trainer.test()
