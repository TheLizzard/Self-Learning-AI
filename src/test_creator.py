import pickle

from environment import Environment


RESULTS = {}


def test_creator():
    env = Environment()
    _test_creator(env)

def _test_creator(env):
    if env.over:
        if env.winner is None:
            return 0
        else:
            return env.winner*2-1
    if env.player:
        best_value = float("-inf")
        for actions in env.legal_actions:
            env.push(actions)
            value = _test_creator(env)
            best_value = max(value, best_value)
            RESULTS.update({env.deepcopy(): value})
            env.pop()
        return best_value
    else:
        best_value = float("inf")
        for move in env.legal_moves:
            env.push(move)
            value = _test_creator(env)
            best_value = min(value, best_value)
            RESULTS.update({env.deepcopy(): value})
            env.pop()
        return best_value


if __name__ == "__main__":
    test_creator()
    with open("tests.tst", "wb") as file:
        file.write(pickle.dumps(RESULTS))
    idx = 0
    for env, value in RESULTS.items():
        print(env, "\t", value)
        if idx == 3:
            break
        else:
            idx += 1