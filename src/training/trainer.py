# -*- coding: utf-8 -*-
import tkinter as tk
import pickle
import random
import copy
import sys

from board.alphabeta import alphabeta_values


with open("tests.tst", "rb") as file:
    test_dataset = pickle.loads(file.read())
test_dataset_list = list(test_dataset.items())
del test_dataset


class Trainer:
    def __init__(self, environment, AI):
        self.environment = environment
        self.AI = AI
        self.training_data = []
        self.reset()

    def full_test(self):
        global test_dataset_list
        error = 0
        for case, correct_value in test_dataset_list:
            error += self.test_value_case(case, correct_value)
        print(error)

    def test(self, sample_size=1000):
        global test_dataset_list
        error = 0
        samples = random.sample(test_dataset_list, sample_size)
        for sample in samples:
            case, correct_value = sample
            error += self.test_value_case(case, correct_value)
        print(error)

    def test_value_case(self, case, correct_value):
        value = self.ask_ai_value(case)
        return (correct_value-value)**2

    def reset(self):
        self.current_environment = self.environment()

    def train(self):
        while not self.environment.over:
            amplified_v, amplified_p = self.amplify(self.AI, self.current_environment)
            env = self.normalise_environment(self.current_environment)
            amplified_v = self.normalise_value(self.current_environment, amplified_v)
            self.training_data.append((env, amplified_p, amplified_v))
            action = self.current_environment.random_action_from_policy(amplified_p)
            self.current_environment.act(action)
        self.reset()

    def flush(self):
        while len(self.training_data) > 0:
            environment, amplified_p, amplified_v = self.training_data.pop(0)
            self.AI.train_single(environment, [amplified_p, amplified_v])

    def normalise_environment(self, environment):
        if environment.player:
            return environment.state_as_list
        else:
            xs, os, ns = environment.state_as_list
            return [os, xs, ns]

    def normalise_value(self, environment, value):
        if environment.player:
            return value
        else:
            return -value

    def amplify(self, ai, environment):
        amplified_vs = alphabeta_values(environment, eval=self.ask_ai_value, depth=3)
        amplified_p = [(i+1)/2 for i in amplified_vs]
        amplified_v = max(amplified_vs)
        return amplified_v, amplified_p

    def ask_ai(self, environment):
        policy, value = self.AI.predict_single(self.normalise_environment(environment))
        return self.normalise_value(environment, value[0]), policy

    def ask_ai_value(self, environment):
        return self.ask_ai(environment)[0]