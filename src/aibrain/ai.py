# -*- coding: utf-8 -*-
from .core import AICore
import numpy as np
import copy


class AI:
    def __init__(self, model, **kwargs):
        self.AI = AICore(model, **kwargs)

    def train(self, questions, answers=None, verbose=0, **kwargs):
        history = self.AI.train(questions, answers, verbose=verbose, **kwargs)
        return history.history

    def train_single(self, question, answer, verbose=0, **kwargs):
        history = self.train([question], [answer], verbose=verbose, **kwargs)
        return {key: value[0] for key, value in history.items()}

    def predict(self, questions):
        return self.AI.predict(questions)

    def predict_single(self, question):
        answers = self.AI.predict(np.asarray([question], dtype="float32"))
        return tuple(answer.tolist()[0] for answer in answers)

    def plot_model(self, *args, **kwargs):
        self.AI.plot_model(*args, **kwargs)

    def deepcopy(self):
        return copy.deepcopy(self)

    def get_ready_for_pickle(self):
        return self.AI.get_ready_for_pickle()

    def save(self, *args, **kwargs):
        self.AI.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        self.AI.load(*args, **kwargs)
