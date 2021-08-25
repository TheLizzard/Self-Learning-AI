import pickle
import random

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


def create_full_test():
    test_creator()
    _list = list(RESULTS.items())
    print("full test size =", len(_list))
    # Shuffle the data
    random.shuffle(_list)
    # Save to "full_test.tst"
    with open("full_test.tst", "wb") as file:
        file.write(pickle.dumps(_list))

def create_test_sample(sample):
    with open("full_test.tst", "rb") as file:
        _list = pickle.loads(file.read())
    print("full test size =", len(_list))
    # Shuffle the data
    random.shuffle(_list)
    # Take the sample
    _list = _list[:sample]
    # Save to "test.tst"
    with open("test.tst", "wb") as file:
        file.write(pickle.dumps(_list))


## All of the ending positions get saved to "tests.tst"
if __name__ == "__main__":
    from constants.seed import set_seed
    set_seed(42)

    create_test_sample(sample=2000)
