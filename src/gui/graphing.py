from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from functools import partial
from copy import deepcopy
import tkinter as tk
import random


COLOURS = ["red", "blue", "green", "black",
           "purple", "pink", "cyan", "lime"]
_COLOURS = deepcopy(COLOURS)


class ScatterPlot(tk.Canvas):
    def __init__(self, master, fg="#000000", bg="#f0f0ed", geometry=(400, 400), dpi=100):
        self.resized = False
        self.width = geometry[0]
        self.height = geometry[1]

        self.xboundaries = [None, None]
        self.yboundaries = [None, None]
        self.xlabel = {}
        self.ylabel = {}
        self.title = {}
        self.set_xlabel(text=None, fontsize=10, colour=fg)
        self.set_ylabel(text=None, fontsize=10, colour=fg)
        self.set_title(text=None, fontsize=15, colour=fg)
        self.fg = fg
        self.bg = bg
        self.show_grid_lines = False
        self.grid_lines_kwargs = {}

        self.figure, self.axis = plt.subplots()
        self.axis.grid()
        self.axis.set_axisbelow(True)
        self.figure.set_size_inches(self.width/dpi, self.height/dpi)
        self.figure.set_dpi(dpi)

        super().__init__(master, height=self.height, width=self.width,
                         borderwidth=0, bg=bg)

    def __getstate__(self):
        _self = deepcopy(self.__dict__)
        _self.pop("figure")
        _self.pop("axis")
        return _self

    def __setstate__(self, _self):
        self.__dict__.update(_self)

    def xlim(self, left=None, right=None):
        """
        Setts the x-axis minimum and maximum. It has these args:
            left: int/float     # The minimum for the x-axis
            right: int/float    # The maximum for the x-axis
        If they are None then the default ones are used.
        """
        if left is not None:
            self.xboundaries[0] = left
        if right is not None:
            self.xboundaries[1] = right

    def ylim(self, left=None, right=None):
        """
        Setts the y-axis minimum and maximum. It has these args:
            left: int/float     # The minimum for the y-axis
            right: int/float    # The maximum for the y-axis
        If they are None then the default ones are used.
        """
        if left is not None:
            self.yboundaries[0] = left
        if right is not None:
            self.yboundaries[1] = right

    def add(self, x, y, colour=None, size=2, marker="o"):
        """
        Plots a scatter graph of the x against the y arguments. These are
        the args:
            x: list/tuple/np.vector
            y: list/tuple/np.vector
            size: int                # the size of the data points
            marker: str              # The marker to be used default = "o"
                                     # Doesn't work.
        Note: This method doesn't show anything to the screen. Call
        <ScatterPlot>.update() to update the display
        """
        if colour is None:
            global COLOURS, _COLOURS
            if len(COLOURS) == 0:
                COLOURS = deepcopy(_COLOURS)
            idx = random.randint(0, len(COLOURS)-1)
            colour = COLOURS.pop(idx)

        kwargs = {"x": x,
                  "y": y,
                  "c": colour,
                  "s": size,
                  "marker": marker}
        self.axis.scatter(**kwargs)

    def update(self, filename="tmp/last_graph_frame.png", format="png"):
        """
        Updates what is shown on the screen. It takes thses args:
            filename: str   # Where you want to save the picture
                            # Default = "tmp/last_graph_frame.png"
            format: str     # The type of the file
                            # Default = "png"
        """
        self._config()
        # Save the picture
        self.figure.savefig(filename, format=format, transparent=True)
        # Open the picture, resize it and display it
        try:
            print("Opening")
            pilimage = Image.open(filename)
            print("Opened")
        except PermissionError as error:
            print("Error")
            raise error
        if self.resized:
            self.config(height=self.height, width=self.width)
            pilimage = pilimage.resize((self.width, self.height), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(pilimage)
        self.create_image(self.width/2, self.height/2, image=self.image)
        print("Closing")
        pilimage.close()
        print("Closed")

    def set_xlabel(self, text=None, fontsize=None, colour=None):
        """
        Setts the x-axis label. These are the arguments:
            text: str
            fontsize: int
            colour: string
        """
        if text is not None:
            self.xlabel.update({"text": text})
        if fontsize is not None:
            self.xlabel.update({"fontsize": fontsize})
        if colour is not None:
            self.xlabel.update({"colour": colour})

    def set_ylabel(self, text=None, fontsize=None, colour=None):
        """
        Setts the y-axis label. These are the arguments:
            text: str
            fontsize: int
            colour: string
        """
        if text is not None:
            self.ylabel.update({"text": text})
        if fontsize is not None:
            self.ylabel.update({"fontsize": fontsize})
        if colour is not None:
            self.ylabel.update({"colour": colour})

    def set_title(self, text=None, fontsize=None, colour=None):
        """
        Setts the title of the plot. These are the arguments:
            text: str
            fontsize: int
            colour: string
        """
        if text is not None:
            self.title.update({"text": text})
        if fontsize is not None:
            self.title.update({"fontsize": fontsize})
        if colour is not None:
            self.title.update({"colour": colour})

    def resize(self, width, height, dpi=None):
        """
        Resizes the Widget to fit the width and height given.
        """
        self.width = width
        self.height = height
        self.resized = True

    def reset(self):
        """
        Removes all of the data points from the screen.
        """
        plt.cla()

    def grid_lines(self, show=True, colour=None, **kwargs):
        """
        If show is True it will display the grid lines.
        kwargs include:
            linewidth: float
            linestyle: str    can be one of: {"-", "--", "-.", ":"}
        """
        if (colour is None) and ("color" not in self.grid_lines_kwargs):
            colour = self.fg
        self.show_grid_lines = show
        self.grid_lines_kwargs.update(kwargs)
        if colour is not None:
            self.grid_lines_kwargs.update({"color": colour})

    def _config(self):
        # Set all of the axis labels
        if self.xlabel["text"] is not None:
            self.axis.set_xlabel(self.xlabel["text"], color=self.xlabel["colour"], fontdict={"fontsize": self.xlabel["fontsize"]})
        if self.ylabel["text"] is not None:
            self.axis.set_ylabel(self.ylabel["text"], color=self.ylabel["colour"], fontdict={"fontsize": self.ylabel["fontsize"]})
        if self.title["text"] is not None:
            self.axis.set_title(self.title["text"], color=self.title["colour"], fontdict={"fontsize": self.title["fontsize"]})
        
        # Set the axis colour:
        self.axis.tick_params(labelcolor=self.fg, color=self.fg)
        self.axis.spines["left"].set_color(self.fg)
        self.axis.spines["right"].set_color(self.fg)
        self.axis.spines["top"].set_color(self.fg)
        self.axis.spines["bottom"].set_color(self.fg)

        # Set x and y boundaries
        self.axis.set_ylim(*self.xboundaries)
        self.axis.set_xlim(*self.yboundaries)

        # Add grid lines
        if self.show_grid_lines:
            self.axis.grid(True, **self.grid_lines_kwargs)
        else:
            self.axis.grid(False)

        # Make everything neat and tidy
        self.figure.tight_layout()


if __name__ == "__main__":
    from time import sleep

    root = tk.Tk()
    root.resizable(False, False)
    root.title("This is the window title")

    plot = ScatterPlot(root, fg="white", bg="black", geometry=(400, 400), dpi=100)
    plot.grid(row=1, column=1)
    plot.resize(300, 300)
    plot.grid_lines(show=True, colour="grey", linestyle="--")

    plot.set_xlabel("This is the x-axis label")
    plot.set_ylabel("This is the y-axis label")
    plot.set_title("This is the title")

    plot.xlim(left=0)
    plot.ylim(left=0)
    plot.update()

    idx = 1
    points = [[], []]

    def destroy():
        root.quit()
        root.destroy()

    def mainloop():
        global idx, points
        points[0].append(idx)
        points[1].append(idx**2)
        plot.reset()
        plot.add(*points, size=10, colour="red")
        plot.update()
        idx += 1
        root.after(2000, mainloop)

    root.protocol("WM_DELETE_WINDOW", destroy)
    root.after(0, mainloop)
    root.mainloop()