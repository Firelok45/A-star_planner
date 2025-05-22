import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import numpy as np
from .algorithms import A_star_final
from .map_utils import create_default_map, add_slow_rectangles


class PathfindingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pathfinding Visualizer")

        # Параметры по умолчанию
        self.base_map = create_default_map()
        self.current_map = np.copy(self.base_map)
        self.start_coord = (2, 2)
        self.end_coord = (20, 14)
        self.speed = 0.7
        self.evr = 2  # Euclidean по умолчанию

        # Параметры для замедляющих прямоугольников
        self.bottom_rect = (9, 12, 6, 4)
        self.top_rect = (9, 1, 6, 3)
        self.mode = 1  # Режим по умолчанию (без замедления)

        self.setup_ui()
        self.draw_map()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Map display
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Start coordinates
        ttk.Label(control_frame, text="Start X:").grid(row=0, column=0, padx=2, pady=2)
        self.start_x_entry = ttk.Entry(control_frame, width=5)
        self.start_x_entry.grid(row=0, column=1, padx=2, pady=2)
        self.start_x_entry.insert(0, str(self.start_coord[0]))
        self.start_x_entry.bind('<KeyRelease>', lambda e: self.update_coords())

        ttk.Label(control_frame, text="Start Y:").grid(row=0, column=2, padx=2, pady=2)
        self.start_y_entry = ttk.Entry(control_frame, width=5)
        self.start_y_entry.grid(row=0, column=3, padx=2, pady=2)
        self.start_y_entry.insert(0, str(self.start_coord[1]))
        self.start_y_entry.bind('<KeyRelease>', lambda e: self.update_coords())

        # End coordinates
        ttk.Label(control_frame, text="End X:").grid(row=1, column=0, padx=2, pady=2)
        self.end_x_entry = ttk.Entry(control_frame, width=5)
        self.end_x_entry.grid(row=1, column=1, padx=2, pady=2)
        self.end_x_entry.insert(0, str(self.end_coord[0]))
        self.end_x_entry.bind('<KeyRelease>', lambda e: self.update_coords())

        ttk.Label(control_frame, text="End Y:").grid(row=1, column=2, padx=2, pady=2)
        self.end_y_entry = ttk.Entry(control_frame, width=5)
        self.end_y_entry.grid(row=1, column=3, padx=2, pady=2)
        self.end_y_entry.insert(0, str(self.end_coord[1]))
        self.end_y_entry.bind('<KeyRelease>', lambda e: self.update_coords())

        # Speed
        ttk.Label(control_frame, text="Slow Speed:").grid(row=0, column=4, padx=2, pady=2)
        self.speed_entry = ttk.Entry(control_frame, width=5)
        self.speed_entry.grid(row=0, column=5, padx=2, pady=2)
        self.speed_entry.insert(0, str(self.speed))

        # Heuristic
        ttk.Label(control_frame, text="Heuristic:").grid(row=1, column=4, padx=2, pady=2)
        self.heuristic_var = tk.StringVar(value="Euclidean")
        heuristic_menu = ttk.OptionMenu(control_frame, self.heuristic_var,
                                        "Euclidean", "Manhattan", "Chebyshev", "Euclidean")
        heuristic_menu.grid(row=1, column=5, padx=2, pady=2)

        # Mode selection
        ttk.Label(control_frame, text="Map Mode:").grid(row=0, column=6, padx=2, pady=2)
        self.mode_var = tk.StringVar(value="No slow zones")
        mode_menu = ttk.OptionMenu(control_frame, self.mode_var,
                                    "No slow zones",
                                   "Bottom zone only",
                                   "Both zones",
                                   "No slow zones")
        mode_menu.grid(row=0, column=7, padx=2, pady=2)
        self.mode_var.trace('w', self.update_map_mode)

        # Buttons
        ttk.Button(control_frame, text="Find Path", command=self.find_path).grid(row=1, column=6, columnspan=2, padx=5,
                                                                                 pady=2)

    def update_coords(self):
        """Обновляем координаты при изменении в полях ввода"""
        try:
            self.start_coord = (
                int(self.start_x_entry.get()),
                int(self.start_y_entry.get())
            )
            self.end_coord = (
                int(self.end_x_entry.get()),
                int(self.end_y_entry.get())
            )
            self.draw_map()
        except ValueError:
            pass  # Игнорируем неполные числа

    def update_map_mode(self, *args):
        """Обновляем режим карты при изменении выбора"""
        mode_map = {
            "No slow zones": 1,
            "Bottom zone only": 2,
            "Both zones": 3,
            "Top zone only": 4
        }
        self.mode = mode_map[self.mode_var.get()]
        self.update_map()

    def update_map(self):
        """Обновляем карту с учетом текущего режима"""
        try:
            self.speed = float(self.speed_entry.get())

            heuristic = self.heuristic_var.get()
            if heuristic == "Manhattan":
                self.evr = 0
            elif heuristic == "Chebyshev":
                self.evr = 1
            else:  # Euclidean
                self.evr = 2

            # Применяем текущий режим к карте
            if self.mode == 1:
                self.current_map = np.copy(self.base_map)  # Без замедляющих зон
            else:
                self.current_map = add_slow_rectangles(
                    self.base_map,
                    self.bottom_rect,
                    self.top_rect,
                    self.mode
                )

            self.draw_map()
        except ValueError as e:
            print(f"Invalid input: {e}")

    def draw_map(self):
        """Отрисовываем карту с текущими маркерами"""
        self.ax.clear()
        cmap = ListedColormap(['black', 'gray', 'white', 'green'])
        self.ax.imshow(self.current_map, cmap=cmap, vmin=0, vmax=3)

        # Mark start and end
        self.ax.scatter(self.start_coord[0], self.start_coord[1], c='red', s=100, label='Start')
        self.ax.scatter(self.end_coord[0], self.end_coord[1], c='blue', s=100, label='End')
        self.ax.legend(loc='upper right')

        self.canvas.draw()

    def find_path(self):
        """Запускаем поиск пути"""
        self.update_map()  # Обновляем параметры

        result_map, total_weight = A_star_final(
            self.current_map,
            self.start_coord,
            self.end_coord,
            self.speed,
            self.evr
        )

        self.ax.clear()
        cmap = ListedColormap(['black', 'gray', 'white', 'green'])
        self.ax.imshow(result_map, cmap=cmap, vmin=0, vmax=3)

        # Mark start and end
        self.ax.scatter(self.start_coord[0], self.start_coord[1], c='red', s=100, label='Start')
        self.ax.scatter(self.end_coord[0], self.end_coord[1], c='blue', s=100, label='End')

        # Add weight info
        self.ax.scatter([], [], c='white', alpha=0, label=f'Path Weight: {round(total_weight, 2)}')
        self.ax.legend(loc='upper right')

        self.canvas.draw()