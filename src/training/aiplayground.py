# -*- coding: utf-8 -*-
import copy
from AlphaBeta import AlphaBeta


class PlayGround:
    def __init__(self, board_resetter, AIs, threshold):
        ids = [id(AI) for AI in AIs]
        for AI in AIs:
            assert ids.count(id(AI)) == 1, "Use <AI.AI object>.deepcopy()"
        self.AIs = AIs
        self.threshold = threshold
        self.board_resetter = board_resetter

    def start(self):
        AI_1, AI_2 = self.AIs
        progress = self.play(AI_1, AI_2)
        progress -= self.play(AI_2, AI_1)
        progress /= 2
        if progress >= self.threshold:
            return 1
        return 0

    def play(self, AI_1, AI_2):
        board = self.board_resetter()
        while not board.game_over:
            move = self.get_move(AI_1, board)
            board.push(move)
            if not board.game_over:
                move = self.get_move(AI_2, board)
                board.push(move)
        winner = board.winner
        if winner == 0:
            return 1
        if winner is None:
            return 0
        return -1

    def get_move(self, AI, board):
        kwargs = {"depth": 2,
                  "evaluator": self.predict,
                  "dampingFactor": 1}
        ab = AlphaBeta(board, **kwargs)
        ab.AI = AI
        return ab.start()["move"]

    def predict(self, board, ab):
        return ab.AI.predict(board.get_state())[0]
                
