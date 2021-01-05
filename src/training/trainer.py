# -*- coding: utf-8 -*-
import numpy as np
import random
import copy
import sys

from aibrain.ai import AI
from board.alphabeta import alphabeta_values
from .datasets import TrainDataset, TestDataset


test_dataset = TestDataset()


"""
all `debug` vars are strings where:
    Var     Class       Method              Description
    "a"     Trainer     "ask_ai"            Prints what the AI is asked each time (this goes directly to keras)
    "t"     Trainer     "train"             Prints each training sample (question, answer) (requires input)
    "e"     Trainer     "test"              Prints the inputs and outputs of each test sample (requires input)
    "n"     Trainer     "normalise"         Prints each time a variable is normalised (eg. environment, value)

    "s"     App         system commands     When the App wants to execute system commands like `exit`
    "p"     App         "predict"           When the App calls predict on the model
    "l"     App         "load"              When the App whats to load a model
    "v"     App         "save"              When the App whats to save the model
    "i"     App         model commands      When the App calls other model commands like `compile`, `config`
    "b"     App         "test"              Prints the loss when the App conducts a test

use like:
    debug=""
    debug="eatn"
    debug="ae"
"""


class Trainer:
    def __init__(self, environment, AI, depth, train_last="all", debug=""):
        self.training_data = TrainDataset(number_inputs=1, number_outputs=2)
        self.environment = environment
        self.train_last = train_last
        self.depth = depth
        self.debug = debug
        self.AI = AI
        self.reset()

    def __getstate__(self):
        return {"training_data": self.training_data,
                "environment": self.environment,
                "AI": self.AI.__getstate__(),
                "depth": self.depth,
                "current_environment": self.current_environment}

    def __setstate__(self, _self, **kwargs):
        self.AI = AI()
        self.AI.__setstate__(_self.pop("AI"), **kwargs)
        self.__dict__.update(_self)

    def reset(self):
        self.current_environment = self.environment()

    def compile(self, **kwargs):
        self.AI.compile(**kwargs)

    def human_test(self, depth):
        env = self.environment()
        while not env.over:
            print(env)
            print("Predicted: "+str(self.amplify(env, depth=depth)))
            _in = input(">>> ")
            if _in.isdigit():
                env.act(int(_in))
            elif _in.lower() == "pop":
                env.undo_action()
            else:
                break

    def test_all(self):
        global test_dataset
        error = 0
        for case, correct_value in test_dataset:
            error += self.test_value_case(case, correct_value)
        return error

    def test(self, sample_size=1000):
        global test_dataset
        error = 0
        samples = random.sample(test_dataset.data, sample_size)
        for sample in samples:
            case, correct_value = sample
            error += self.test_value_case(case, correct_value)
        return error

    def test_value_case(self, environment, correct_value):
        value = self.ask_ai_value(environment, normalise=False)
        if "e" in self.debug:
            print("[debug][e]   environment="+str(environment)+"    correct_value="+str(correct_value)+"  value="+str(value))
            input("[debug][e] input>>> ")
        return (correct_value-value)**2

    def train(self):
        last_done = False
        while (not self.current_environment.over) or (not last_done):
            amplified_v, amplified_p = self.amplify(self.current_environment)
            policy = self.add_missing([p+0.1 for p in amplified_p], self.current_environment)

            neg_environment = self.normalise_environment(self.current_environment, reverse=True)
            environment = self.normalise_environment(self.current_environment)
            amplified_p = self.add_missing(amplified_p, self.current_environment)

            if "t" in self.debug:
                print("[debug][t]   environment="+str(environment)+"    amplified_p="+str(amplified_p)+"  amplified_v="+str(amplified_v))
                input("[debug][t] input>>> ")

            self.training_data.add(environment, amplified_p, amplified_v)
            self.training_data.add(neg_environment, amplified_p, -amplified_v)

            if not self.current_environment.over:
                action = self.current_environment.random_action_from_policy(policy)
                self.current_environment.act(action)
            else:
                last_done = True
        self.reset()

    def flush(self, **kwargs):
        self.AI.train(self.training_data, **kwargs)
        if self.train_last != "all":
            self.training_data.flush(slice(-int(self.train_last), None, None))

    def normalise_environment(self, environment, reverse=False):
        if "n" in self.debug:
            print("[debug][n]   environment="+str(environment)+"    player="+str(environment.player)+"  reverse="+str(reverse))
        if reverse:
            if environment.player:
                xs, os, ns = environment.state_as_list
                return [os, xs, ns]
            else:
                return environment.state_as_list

        if environment.player:
            return environment.state_as_list
        else:
            xs, os, ns = environment.state_as_list
            return [os, xs, ns]

    def normalise_value(self, environment, value):
        if "n" in self.debug:
            print("[debug][n]   environment="+str(environment)+"    player="+str(environment.player)+"  value="+str(value))
        if environment.player:
            return value
        else:
            return -value

    def amplify(self, environment, depth=None):
        if depth is None:
            depth = self.depth
        amplified_vs = alphabeta_values(environment, eval=self.ask_ai_value, depth=depth)
        amplified_vs = [self.normalise_value(environment, value) for value in amplified_vs]
        amplified_p = [(i+1)/2 for i in amplified_vs] # Note: `(i+1)/2` converts the score from [-1, 1] to [0, 1]
        if environment.player:
            amplified_v = max(amplified_vs)
        else:
            amplified_v = min(amplified_vs)
        return amplified_v, amplified_p

    def add_missing(self, policy, environment):
        """
        Sometimes the policy wouldn't be 9 items long
        but to have a numpy array of policies we need
        all of them to have the same size.
        """
        # Must be in order that they are considered in the alphabeta
        all_actions = (7, 8, 9, 4, 5, 6, 1, 2, 3)
        extended_policy = []
        legal_actions = tuple(environment.legal_actions)
        idx = 0
        for action in all_actions:
            if action in legal_actions:
                extended_policy.append(policy[idx])
                idx += 1
            else:
                extended_policy.append(0)
        return extended_policy

    def ask_ai(self, environment, normalise=True):
        if normalise:
            question = self.normalise_environment(environment)
        else:
            question = environment.state_as_list
        question = np.asarray(question)
        policy, value = self.AI.predict_single(question)
        value = value[0]
        if "a" in self.debug:
            print("[debug][a]   question="+str(question)+"    policy="+str(policy)+"  value="+str(value))
        if normalise:
            return self.normalise_value(environment, value), policy
        else:
            return value, policy

    def ask_ai_value(self, environment, normalise=True):
        return self.ask_ai(environment, normalise=normalise)[0]

    def config(self, *args, **kwargs):
        return self.AI.config(*args, **kwargs)