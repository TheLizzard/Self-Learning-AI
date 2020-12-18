from training.trainer import Trainer
from board.board import TicTacToe as Game
from aibrain.ai import AI


if __name__ == "__main__":
    dense = [{"type": "flatten"},
             {"type": "dense", "size": 100},
             {"type": "dense", "size": 500},
             {"type": "dense", "size": 250}]

    conv = [{"type": "resize", "shape": (3, 3, 4, 1)},
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

    model = [{"type": "input", "shape": (3, 3, 4)},
             {"type": "duplicate"},
             [conv, dense],
             {"type": "dense", "size": 500},
             {"type": "dropout", "rate": 0.5},
             {"type": "split", "loc": (250, 250), "target_dim": 1},
             [value, policy]]

    core = AI(model, loss="categorical_crossentropy", learning_rate=0.001)
    Trainer(core, Game)
