from aibrain.core import AICore

model = [{"type": "input", "shape": (1, )},
         {"type": "dense", "size": 2},
         {"type": "split", "sizes": (1, 1), "target_dim": 1}]
core = AICore(model)
print(core.predict([[0]]))