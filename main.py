import tkinter as tk
from pathfinder.gui import PathfindingApp

def main():
    root = tk.Tk()
    app = PathfindingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()