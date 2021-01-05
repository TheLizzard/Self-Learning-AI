from keras.utils import Sequence
import numpy as np
import pickle


class TrainDataset(Sequence):
    def __init__(self, number_inputs=1, number_outputs=1):
        """
        `number_vars` shows how many vars there are going to be:
            2 if there is one input and one output
            3 if there is one input and two outputs
            4 if there are two inputs and two outputs
        """
        self.number_inputs = number_inputs
        self.number_outputs = number_outputs
        self.number_vars = number_inputs+number_outputs
        self.data = []
        for var in range(self.number_vars):
            self.data.append([])

    def add(self, *vars):
        msg = "The number of variables given doesn't match what it should be."
        assert len(vars) == self.number_vars, msg
        for i, var in enumerate(vars):
            self.data[i].append(np.asarray(var, dtype="float32"))

    def flush(self, slice=None):
        if slice is None:
            self.data = []
            for var in range(self.number_vars):
                self.data.append([])
        else:
            self.data = [_list[slice] for _list in self.data]

    def __getitem__(self, idx):
        all = list(np.asarray(d, dtype="float32") for d in self.data)
        questions = all[:self.number_inputs]
        answers = all[self.number_inputs:]
        if self.number_inputs == 1:
            questions = questions[0]
        if self.number_outputs == 1:
            answers = answers[0]
        return questions, answers

    def __len__(self):
        return 1


class TestDataset:
    __slots__ = ("data", "idx")

    def __init__(self, filename="tests.tst"):
        with open(filename, "rb") as file:
            self.data = pickle.loads(file.read())

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)