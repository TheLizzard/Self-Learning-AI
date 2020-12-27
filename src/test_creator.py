import pickle

from board.environment import Environment


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


## All of the ending positions get saved to "tests.tst"
if __name__ == "__main__":
    from constants.set_seed import set_seed
    set_seed(42)

    import random

    # Calculate the data
    test_creator()
    _list = list(RESULTS.items())

    # Shuffle the data
    random.shuffle(_list)
    print(len(_list))

    # Take only the first 1000 items of the data
    _list = _list[:1000]
    print(_list[:3])

    # Save the data to a file
    with open("tests.tst", "wb") as file:
        file.write(pickle.dumps(_list))