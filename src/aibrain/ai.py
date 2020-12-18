# -*- coding: utf-8 -*-
from constants.set_seed import set_seed
set_seed(42)

from .core import AICore
import numpy as np
import copy


class AI:
    def __init__(self, model, **kwargs):
        self.AI = AICore(model, **kwargs)

    def train(self, questions, answers, verbose=0, **kwargs):
        history = self.AI.train(np.asarray(question), np.asarray(answer),
                                verbose=verbose, **kwargs)
        return history.history

    def train_single(self, question, answer, verbose=0, **kwargs):
        history = self.train(np.asarray([question]), np.asarray([answer]),
                             verbose=verbose, **kwargs)
        return {key: value[0] for key, value in history.items()}

    def predict(self, questions):
        return self.AI.predict(np.asarray(questions)).tolist()

    def predict_single(self, question):
        return self.predict(np.asarray([question]))[0]

    def deepcopy(self):
        return copy.deepcopy(self)

    def get_ready_for_pickle(self):
        return self.AI.get_ready_for_pickle()

    def save(self, *args, **kwargs):
        self.AI.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        self.AI.load(*args, **kwargs)
