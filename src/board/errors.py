class IllegalMove(Exception):
    def __init__(self, board, move):
        text = "The move %s is illegal in this poistion %s."
        super().__init__(text%(str(move), str(board)))


class IllegalPop(Exception):
    def __init__(self):
        text = "There are no moves to undo."
        super().__init__(text)
