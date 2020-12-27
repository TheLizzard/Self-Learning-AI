# -*- coding: utf-8 -*-
from .core import AICore
import numpy as np
import copy


class AI:
    def __init__(self, model=None, custom_objects={}, ask_verify=True):
        self.AI = AICore(model, ask_verify=ask_verify, custom_objects=custom_objects)

    def __getstate__(self, **kwargs):
        return self.AI.__getstate__(**kwargs)

    def __setstate__(self, state, **kwargs):
        self.AI = AICore()
        self.AI.__setstate__(state, **kwargs)

    def compile(self, **kwargs):
        self.AI.compile(**kwargs)

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

    def ask_verify(self, *args, **kwargs):
        self.AI.ask_verify_model(*args, **kwargs)
