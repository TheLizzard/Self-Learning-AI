from keras.utils import Sequence
import numpy as np
import pickle


class TrainDataset(Sequence):
    __slots__ = ("data", )

    def __init__(self, number_vars=2):
        """
        `number_vars` shows how many vars there are going to be:
            2 if there is one input and one output
            3 if there is one input and two outputs
            4 if there are two inputs and two outputs
        """
        self.number_vars = number_vars
        self.data = []
        for var in range(number_vars):
            self.data.append([])

    def add(self, *vars):
        msg = "The number of variables given doesn't match what it should be."
        assert len(vars) == len(self.data), msg
        for i, var in enumerate(vars):
            self.data[i].append(np.asarray(var))

    def __getitem__(self, idx):
        question, *answers = tuple(np.asarray(d) for d in self.data)
        return question, list(answers)

    def __len__(self):
        return 1


class TestDataset:
    __slots__ = ("data", "idx")

    def __init__(self, filename="tests.tst"):
        self.idx = 0
        with open(filename, "rb") as file:
            self.data = pickle.loads(file.read())

    def __iter__(self):
        self.idx = 0
        return self.data

    def __next__(self):
        if len(self.data) == self.idx:
            raise StopIteration
        else:
            self.idx += 1
            return self[self.idx-1]

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1