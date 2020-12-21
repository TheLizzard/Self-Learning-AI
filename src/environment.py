from numpy.random import choice
import warnings
import copy

from board.board import TicTacToe


WARNED_DEEPCOPY = False


class Environment(TicTacToe):
    def act(self, action):
        super().push(action)

    def undo_action(self):
        return super().pop()

    @property
    def over(self):
        return super().game_over

    @property
    def legal_actions(self):
        return super().legal_moves

    @property
    def state_as_list(self):
        board = self.state.board
        xs = [[], [], []]
        os = [[], [], []]
        ns = [[], [], []]
        for i, value in enumerate(board):
            if value == 1:
                xs[i//3].append(1)
                os[i//3].append(0)
                ns[i//3].append(0)
            elif value == -1:
                xs[i//3].append(0)
                os[i//3].append(1)
                ns[i//3].append(0)
            elif value == 0:
                xs[i//3].append(0)
                os[i//3].append(0)
                ns[i//3].append(1)
        return [xs, os, ns]
    
    def random_action_from_policy(self, probability_distribution):
        # We normalised the `probability_distribution` to make sure it has
        # 9 elements so we can use the full `all_actions`
        all_actions = (7, 8, 9, 4, 5, 6, 1, 2, 3)
        probability_distribution = self.normalise_distribution(probability_distribution)
        return choice(all_actions, 1, p=probability_distribution).tolist()[0]

    def deepcopy(self):
        global WARNED_DEEPCOPY
        if not WARNED_DEEPCOPY:
            WARNED_DEEPCOPY = True
            warnings.warn("This is a very slow method!")
        return copy.deepcopy(self)

    def normalise_distribution(self, distribution):
        _sum = sum(distribution)
        return [i/_sum for i in distribution]