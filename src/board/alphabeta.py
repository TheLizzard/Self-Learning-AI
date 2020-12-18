# -*- coding: utf-8 -*-
def alphabeta_move(node, depth, eval, α=float("-inf"), β=float("inf")):
    if depth == 0:
        return eval(node)
    elif node.game_over:
        if node.winner is None:
            return 0
        elif node.winner is True:
            return 1
        elif node.winner is False:
            return -1
    elif node.player:
        value = float("-inf")
        best_move = None
        for move in node.legal_moves:
            node.push(move)
            new_value = alphabeta(node, depth-1, eval, α, β)
            if new_value > value:
                value = new_value
                best_move = move
            node.pop()
            α = max(α, value)
            if α >= β:
                break
        return value, best_move
    else:
        value = float("inf")
        best_move = None
        for move in node.legal_moves:
            node.push(move)
            new_value = alphabeta(node, depth-1, eval, α, β)
            if new_value < value:
                value = new_value
                best_move = move
            node.pop()
            β = min(β, value)
            if α >= β:
                break
        return value, best_move

def alphabeta(node, depth, eval, α=float("-inf"), β=float("inf")):
    if depth == 0:
        return eval(node)
    elif node.game_over:
        if node.winner is None:
            return 0
        elif node.winner is True:
            return 1
        elif node.winner is False:
            return -1
    elif node.player:
        value = float("-inf")
        for move in node.legal_moves:
            node.push(move)
            value = max(value, alphabeta(node, depth-1, eval, α, β))
            node.pop()
            α = max(α, value)
            if α >= β:
                break
        return value
    else:
        value = float("inf")
        for move in node.legal_moves:
            node.push(move)
            value = min(value, alphabeta(node, depth-1, eval, α, β))
            node.pop()
            β = min(β, value)
            if α >= β:
                break
        return value


if __name__ == "__main__":
    from board import TicTacToe

    def eval(node):
        pass

    board = TicTacToe()
    while not board.game_over:
        result = alphabeta_move(board, 20, eval)
        print(board, board.player, result, sep="\t")
        _in = input("? ")
        if _in.isdigit():
            board.push(int(_in))
        else:
            board.pop()
