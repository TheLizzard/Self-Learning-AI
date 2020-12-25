class partial:
    __slots__ = ("function", "args", "kwargs")

    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        output = str(self.function)+"("

        args = ", ".join(map(str, self.args))
        if args != "":
            output += "%s"%args

        kwargs = str(self.kwargs)[1:-1]
        if kwargs != "":
            if args != "":
                output += ", "
            output += "%s"%kwargs

        return output+")"

    def __call__(self, *args, **kwrags):
        return self.function(*self.args, *args, **self.kwargs, **kwrags)