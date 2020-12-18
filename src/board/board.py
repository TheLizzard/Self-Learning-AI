from random import randint
from .errors import IllegalMove
import copy


WINNING_MOVES = [[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8],
                 [0, 3, 6],
                 [1, 4, 7],
                 [2, 5, 8],
                 [0, 4, 8],
                 [2, 4, 6]]

MOVE_TO_IDX = {7: 0,
               8: 1,
               9: 2,
               4: 3,
               5: 4,
               6: 5,
               1: 6,
               2: 7,
               3: 8}

IDX_TO_MOVE = {value: key for key, value in MOVE_TO_IDX.items()}


class TicTacToeState:
    def __init__(self, board=[0 for i in range(9)], player=True):
        self.board = board
        self.player = player
        self.update_game_over()

    def __repr__(self):
        return str(self)

    def __str__(self):
        out = "TicTacToeState(\n\t"
        for i, cell in enumerate(self.board, 1):
            if cell == -1:  out += "O"
            elif cell == 0: out += "•"
            elif cell == 1: out += "X"

            if i%3 == 0: out += "\n\t"
            else: out += " "
        return out[:-1]+")"

    def get(self, key):
        return self.board[key]

    def set(self, key, value):
        self.board[key] = value

    def deepcopy(self):
        return copy.deepcopy(self)

    @property
    def legal_moves(self):
        for idx in range(9):
            if self.board[idx] == 0:
                yield self.idx_to_move(idx)

    def update_game_over(self):
        """
        Updates the <TicTacToeState>.game_over attribute.
        """
        self.update_winner()
        if (self.winner is not None) or (0 not in self.board):
            self.game_over = True
        else:
            self.game_over = False

    def update_winner(self):
        """
        Updates the <TicTacToeState>.winner attribute, where:
            None    means draw
            False   means black/O won
            True    means white/X won
        """
        for a, b, c in WINNING_MOVES:
            if self.get(a) == self.get(b) == self.get(c) != 0:
                if self.get(a) == 1:
                    self.winner = True
                elif self.get(a) == -1:
                    self.winner = False
                self.game_over = True
                return None
        self.winner = None

    def inverse(self):
        """
        Takes the inverse of the position (switches the Xs and Os) and
        returns the TicTacToeState after that.
        """
        new_board = []
        for cell in self.board:
            if cell == -1:
                new_board.append(1)
            elif cell == 0:
                new_board.append(0)
            elif cell == 1:
                new_board.append(-1)
        return TicTacToeState(new_board, not self.player)

    def push(self, move):
        """
        Pushes a move and returns the TicTacToeState after the push.
        Note: It doesn't check for move legality.
        """
        idx = self.move_to_idx(move)
        if self.get(idx) == 0:
            new_board = self.copy_list(self.board)
            if self.player:
                new_board[idx] = 1
            else:
                new_board[idx] = -1
            return TicTacToeState(new_board, not self.player)
        else:
            raise IllegalMove(self, move)

    @staticmethod
    def copy_list(_list):
        """
        Returns a copy of the list
        """
        new_list = []
        for i in _list:
            new_list.append(i)
        return new_list

    @staticmethod
    def move_to_idx(move):
        return MOVE_TO_IDX[move]

    @staticmethod
    def idx_to_move(idx):
        return IDX_TO_MOVE[idx]


class TicTacToe:
    def __init__(self):
        self.state = TicTacToeState()
        self.move_stack = []
        self.states_stack = []

    def __str__(self):
        return str(self.state).replace("TicTacToeState", "TicTacToe")

    def push(self, move):
        """
        Pushes a move and pushes the old state to the
        <TicTacToe>.states_stack.
        """
        self.states_stack.append(self.state)
        self.state = self.state.push(move)
        self.move_stack.append(move)

    def pop(self):
        """
        Undoes a move by poping from the <TicTacToe>.states_stack
        """
        if len(self.move_stack) == 0:
            raise IllegalPop()
        self.state = self.states_stack.pop()
        return self.move_stack.pop()

    def random_move(self):
        """
        Returns a random move (all moves have equal weights)
        """
        available_moves = tuple(self.legal_moves())
        return available_moves[randint(0, len(available_moves)-1)]

    def deepcopy(self):
        return copy.deepcopy(self)

    @property
    def game_over(self):
        return self.state.game_over

    @property
    def winner(self):
        return self.state.winner

    @property
    def stack(self):
        return self.move_stack

    @property
    def legal_moves(self):
        return self.state.legal_moves

    @property
    def player(self):
        return self.state.player


if __name__ == "__main__":
    moves = (7, 5, 3)
    board = TicTacToe()
    for move in moves:
        board.push(move)
    print(board)
    print(board.winner)


"""
winners = [
			[0,1,2,3],
			[1,2,3,4],
			[2,3,4,5],
			[3,4,5,6],
			[7,8,9,10],
			[8,9,10,11],
			[9,10,11,12],
			[10,11,12,13],
			[14,15,16,17],
			[15,16,17,18],
			[16,17,18,19],
			[17,18,19,20],
			[21,22,23,24],
			[22,23,24,25],
			[23,24,25,26],
			[24,25,26,27],
			[28,29,30,31],
			[29,30,31,32],
			[30,31,32,33],
			[31,32,33,34],
			[35,36,37,38],
			[36,37,38,39],
			[37,38,39,40],
			[38,39,40,41],

			[0,7,14,21],
			[7,14,21,28],
			[14,21,28,35],
			[1,8,15,22],
			[8,15,22,29],
			[15,22,29,36],
			[2,9,16,23],
			[9,16,23,30],
			[16,23,30,37],
			[3,10,17,24],
			[10,17,24,31],
			[17,24,31,38],
			[4,11,18,25],
			[11,18,25,32],
			[18,25,32,39],
			[5,12,19,26],
			[12,19,26,33],
			[19,26,33,40],
			[6,13,20,27],
			[13,20,27,34],
			[20,27,34,41],

			[3,9,15,21],
			[4,10,16,22],
			[10,16,22,28],
			[5,11,17,23],
			[11,17,23,29],
			[17,23,29,35],
			[6,12,18,24],
			[12,18,24,30],
			[18,24,30,36],
			[13,19,25,31],
			[19,25,31,37],
			[20,26,32,38],

			[3,11,19,27],
			[2,10,18,26],
			[10,18,26,34],
			[1,9,17,25],
			[9,17,25,33],
			[17,25,33,41],
			[0,8,16,24],
			[8,16,24,32],
			[16,24,32,40],
			[7,15,23,31],
			[15,23,31,39],
			[14,22,30,38],
			]
"""
