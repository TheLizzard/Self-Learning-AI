# -*- coding: utf-8 -*-
def alphabeta_values(node, depth, eval):
    values = []
    if depth == 0:
        return eval(node)
    elif node.game_over:
        if node.winner is None:
            value = 0
        else:
            value = node.winner*2-1
        return [value for i in range(9)]
    elif node.player:
        for move in node.legal_moves:
            node.push(move)
            values.append(alphabeta(node, depth-1, eval))
            node.pop()
    else:
        for move in node.legal_moves:
            node.push(move)
            values.append(alphabeta(node, depth-1, eval))
            node.pop()
    return values

def alphabeta(node, depth, eval, α=float("-inf"), β=float("inf")):
    if depth == 0:
        return eval(node)
    elif node.game_over:
        if node.winner is None:
            return 0
        else:
            return node.winner*2-1
    elif node.player:
        value = float("-inf")
        for move in node.legal_moves:
            node.push(move)
            value = max(value, alphabeta(node, depth-1, eval, α, β))
            if value == 1:
                node.pop()
                return 1
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
            if value == -1:
                node.pop()
                return -1
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
        result = alphabeta_values(board, 20, eval)
        print(tuple(board.legal_moves))
        print(board, board.player, result, sep="\t")
        _in = input("? ")
        if _in.isdigit():
            board.push(int(_in))
        else:
            board.pop()
