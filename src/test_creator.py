import pickle

from environment import Environment


RESULTS = {}


def test_creator():
    env = Environment()
    _test_creator(env)

def _test_creator(env):
    if env.over:
        if env.winner is None:
            value = 0
        else:
            value = env.winner*2-1
        RESULTS.update({env.deepcopy(): value})
        return value
    if env.player:
        best_value = float("-inf")
        for actions in env.legal_actions:
            env.push(actions)
            value = _test_creator(env)
            best_value = max(value, best_value)
            #RESULTS.update({env.deepcopy(): value})
            env.pop()
        return best_value
    else:
        best_value = float("inf")
        for move in env.legal_moves:
            env.push(move)
            value = _test_creator(env)
            best_value = min(value, best_value)
            #RESULTS.update({env.deepcopy(): value})
            env.pop()
        return best_value


if __name__ == "__main__":
    test_creator()
    with open("tests.tst", "wb") as file:
        _list = list(RESULTS.items())
        print(_list[:3])
        file.write(pickle.dumps(_list))