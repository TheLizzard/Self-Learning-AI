from functools import partial
from time import sleep
import tkinter as tk
import threading
import sys

from .draggablewindow import DraggableWindow
#from .threadsafe.partial import partial
from .graphing import ScatterPlot


class ContinuousPlotWindow:
    def __init__(self, fg="#000000", bg="#f0f0ed", geometry=(400, 400), dpi=100, exit_when_done=False):
        self.exit_when_done = exit_when_done
        self.plotted_points = -1
        self.points = [[], []]
        self.kwargs = {}
        self.set_format()
        self.ops = []

        self.root = DraggableWindow()
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)
        self.plot = ScatterPlot(self.root, fg=fg, bg=bg, geometry=geometry, dpi=dpi)
        self.plot.grid(row=1, column=1)

    def __getstate__(self):
        self_dict = {"exit_when_done": self.exit_when_done,
                     "points": self.points,
                     "kwargs": self.kwargs,
                     "plot": self.plot.__getstate__()}
        return self_dict

    def __setstate__(self, _self):
        self.__dict__.update(_self)
        self.root = DraggableWindow()
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)
        self.plot = ScatterPlot(self.root, **self.plot.__setstate__())
        self.plot.grid(row=1, column=1)

    def destroy(self): # Do not call
        self.root.quit()
        self.root.destroy()

    def _mainloop(self): # Do not call
        self.flush_ops()
        self.root.after(100, self._mainloop)

    def flush_ops(self): # Do not call
        while len(self.ops) > 0:
            op = self.ops.pop(0)
            op()
        if len(self.points[0]) != self.plotted_points:
            self.plotted_points = len(self.points[0])
            self.plot.reset()
            self.plot.add(*self.points, **self.kwargs)
            self.plot.update()

    # The caller can only call the methods bellow:
    def mainloop(self):
        self.root.after(100, self._mainloop)
        self.root.mainloop()
        if self.exit_when_done:
            exit()

    def reset(self):
        self.ops.append(partial(self.plot.reset))

    def set_main(self, function):
        t = threading.Thread(target=function, daemon=True)
        t.start()

    def set_format(self, colour=None, size=2, marker="o"):
        """
        Setts the format of the data points
            colour: str     Default=
            size: int
            marker: str
        """
        self.kwargs = {"colour": colour, "size": size, "marker": marker}

    def add(self, x, y):
        """
        Adds a data point to the graph. Arguments:
            x: float
            y: float
        Note: To change the points' apperance look at
              `ContinuousPlotWindow.set_format`
        """
        self.points[0].append(x)
        self.points[1].append(y)

    def xlim(self, left=None, right=None):
        """
        Setts the x-axis minimum and maximum. It has these args:
            left: int/float     # The minimum for the x-axis
            right: int/float    # The maximum for the x-axis
        If they are None then the default ones are used.
        """
        self.plot.xlim(left=left, right=right)

    def ylim(self, left=None, right=None):
        """
        Setts the y-axis minimum and maximum. It has these args:
            left: int/float     # The minimum for the y-axis
            right: int/float    # The maximum for the y-axis
        If they are None then the default ones are used.
        """
        self.plot.ylim(left=left, right=right)

    def set_xlabel(self, text, fontsize=10, colour="black"):
        """
        Setts the x-axis label. These are the arguments:
            text: str
            fontsize: int
            colour: string
        """
        self.plot.set_xlabel(text, fontsize=fontsize, colour=colour)

    def set_ylabel(self, text, fontsize=10, colour="black"):
        """
        Setts the y-axis label. These are the arguments:
            text: str
            fontsize: int
            colour: string
        """
        self.plot.set_ylabel(text, fontsize=fontsize, colour=colour)

    def set_title(self, text, fontsize=15, colour="black"):
        """
        Setts the title of the plot. These are the arguments:
            text: str
            fontsize: int
            colour: string
        """
        self.plot.set_title(text, fontsize=fontsize, colour=colour)

    def resize(self, width, height, dpi=None):
        """
        Resizes the Widget to fit the width and height given.
        """
        self.ops.append(partial(self.plot.resize, width=width, height=height, dpi=dpi))

    def grid_lines(self, show=True, colour=None, **kwargs):
        """
        If show is True it will display the grid lines.
        kwargs include:
            linewidth: float
            linestyle: str    can be one of: {"-", "--", "-.", ":"}
        """
        self.plot.grid_lines(show=show, colour=colour, **kwargs)