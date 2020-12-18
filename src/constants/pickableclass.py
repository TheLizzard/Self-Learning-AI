import pickle


class Unknown:
    def __init__(self, class_name):
        self.class_name = class_name

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<UnknowObject of type %s>"%self.class_name


class PickableClass:
    def __getstate__(self):
        _self = {}
        for key, value in self.__dict__.items():
            try:
                pickle.dumps(getattr(self, key))
                _self.update({key: value})
            except:
                _self.update({key: Unknown(type(value))})
        return pickle.dumps(_self)

    def __setstate__(self, state):
        _self = pickle.loads(state)
        for key, value in _self.items():
            setattr(self, key, value)
