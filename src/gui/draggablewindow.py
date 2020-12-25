import tkinter as tk


class DraggableWindow(tk.Tk):
    """
    A draggable window.
        Controls:
            right click and drag      to move the window
            ctrl+w                    to close the window
            left click                to reset the window to (0, 0)
            scroll wheel click        to move window to the closest corner of the screen
    """
    def __init__(self):
        super().__init__()
        self._offsetx = 0
        self._offsety = 0

        self.topmost()
        super().bind("<Control-w>", self.kill)
        super().bind("<Button-1>", self.clickwin)
        super().bind("<B1-Motion>", self.dragwin)
        super().bind("<Button-2>", self.move_closest_corner)
        super().bind("<Button-3>", self.reset_position)

    def topmost(self):
        super().attributes("-topmost", True)
        try:
            super().attributes("-type", "splash") # Linux
        except:
            super().overrideredirect(True) # Windows

    def dragwin(self, event):
        x = super().winfo_pointerx() - self._offsetx
        y = super().winfo_pointery() - self._offsety
        super().geometry("+%d+%d"%(x, y))

    def clickwin(self, event):
        self._offsetx = event.x+event.widget.winfo_rootx()-super().winfo_rootx()
        self._offsety = event.y+event.widget.winfo_rooty()-super().winfo_rooty()

    def reset_position(self, event):
        super().geometry("+0+0")

    def move_closest_corner(self, event):
        screen_width = super().winfo_screenwidth()
        screen_height = super().winfo_screenheight()

        root_width = super().winfo_width()
        root_height = super().winfo_height()

        centre_x = root_width/2+super().winfo_rootx()
        centre_y = root_height/2+super().winfo_rooty()

        if screen_width/2 < centre_x:
            geometry = "+%i"%int(screen_width-root_width)
        else:
            geometry = "+0"

        if screen_height/2 < centre_y:
            geometry += "+%i"%int(screen_height-root_height)
        else:
            geometry += "+0"

        super().geometry(geometry)

    def kill(self, event=None):
        super().quit()
        super().destroy()


if __name__ == "__main__":
    root = DraggableWindow()
    label = tk.Label(root, text="-"*20+"Label"+"-"*20)
    label.pack()
    root.mainloop()