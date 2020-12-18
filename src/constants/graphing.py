from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from copy import deepcopy
import tkinter as tk
import random


COLOURS = ["red", "blue", "green", "black",
           "purple", "pink", "cyan", "lime"]
__COLOURS = deepcopy(COLOURS)


class ScatterPlot(tk.Canvas):
    def __init__(self, master, bg="#f0f0ed", geometry=(400, 400), dpi=100):
        self.width = geometry[0]
        self.height = geometry[1]
        self.dpi = dpi
        self.dpi_ratio = self.dpi/self.width

        self.xboundaries = [None, None]
        self.yboundaries = [None, None]

        self.figure, self.axis = plt.subplots()
        self.axis.grid()
        self.axis.set_axisbelow(True)
        self.figure.set_size_inches(self.width/dpi, self.height/dpi)
        self.figure.set_dpi(dpi)

        super().__init__(master, height=self.height, width=self.width,
                         borderwidth=0)

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
            global COLOURS, __COLOURS
            if len(COLOURS) == 0:
                COLOURS = deepcopy(__COLOURS)
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
        self.config(height=self.height, width=self.width)
        # Set x and y boundaries
        self.axis.set_ylim(*self.xboundaries)
        self.axis.set_xlim(*self.yboundaries)
        # Make everything neat and tidy
        self.figure.tight_layout()
        # Save the picture
        self.figure.savefig(filename, format=format, transparent=True)
        # Open the picture and display it
        pilimage = Image.open(filename)
        self.image = ImageTk.PhotoImage(pilimage)
        self.create_image(self.width/2, self.height/2, image=self.image)

    def set_xlabel(self, text, fontsize=10, colour="black"):
        """
        Setts the x-axis label. These are the arguments:
            text: str
            fontsize: int
            colour: string
        """
        fontdict = {"fontsize": fontsize}
        self.axis.set_xlabel(text, fontdict=fontdict, color=colour)

    def set_ylabel(self, text, fontsize=10, colour="black"):
        """
        Setts the y-axis label. These are the arguments:
            text: str
            fontsize: int
            colour: string
        """
        fontdict = {"fontsize": fontsize}
        self.axis.set_ylabel(text, fontdict=fontdict, color=colour)

    def set_title(self, text, fontsize=15, colour="black"):
        """
        Setts the title of the plot. These are the arguments:
            text: str
            fontsize: int
            colour: string
        """
        fontdict = {"fontsize": fontsize}
        self.axis.set_title(text, fontdict=fontdict, color=colour)

    def resize(self, width, height, dpi=None):
        """
        Resizes the Widget to fit the width and height given.
        """
        self.width = width
        self.height = height
        if dpi is None:
            self.dpi = self.dpi_ratio*width
        else:
            self.dpi = dpi
        self.figure.set_size_inches(width/self.dpi, height/self.dpi)
        self.figure.set_dpi(self.dpi)


#class ScatterPlotWindow(ScatterPlot):
#    def __init__(self, *args, **kwargs):
#        self.root = tk.Tk()
#        super().__init__(self.root, *args, **kwargs)
#        self.aspect_ratio = self.width/self.height
#        self.grid(row=1, column=1)
#        self.root.bind("<Configure>", self.update_size)
#
#    def update_size(self, event):
#        current_width = self.width
#        current_height = self.height
#
#        desired_width = event.width-4
#        desired_height = event.height-4
#
#        if current_width == desired_width:
#            if current_height == desired_height:
#                return None
#
#        new_height = min(desired_height, desired_width/self.aspect_ratio)
#        new_width = new_height*self.aspect_ratio
#
#        super().resize(width=new_width, height=new_height)
#        super().update()
#        print(event.width)


if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)

    plot = ScatterPlot(root, geometry=(400, 400), dpi=100)
    plot.grid(row=1, column=1)

    datax = [0, 1, 2, 3, 4, 5, 6]
    datay = [0, 1, 4, 9, 16, 25, 36]

    plot.add(datax, datay, size=10, colour="red")
    plot.set_xlabel("This is the x-axis label")
    plot.set_ylabel("This is the y-axis label")
    plot.set_title("This is the title")

    plot.xlim(left=0)
    plot.ylim(left=0)
    plot.update()
