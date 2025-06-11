import tkinter as tk
from tkinter import font

import random
import time

from utils import debug

class NimTrainingVisualizer(tk.Tk):
    """
    A Tkinter-based visualizer for tracking training progress in Nim.
    
    This visualizer displays:
    - A blue line representing the win rate over training episodes.
    - A red circle marking the maximum win rate point achieved so far.
    - An orange line showing the epsilon decay (or exploration rate) over episodes.
    
    Axes are labeled and numbered, with the episode count on the x-axis
    and values (0 to 1) on the y-axis. Live values are displayed below the plot.
    
    Parameters:
    -----------
    total_episodes : int
        The total number of training episodes planned.
    max_sticks : int
        The maximum number of sticks in the Nim game (used for context, not for plotting).
    """

    def __init__(self, total_episodes, max_sticks):
        super().__init__()
        self.debug = debug()
        
        if self.debug:
            print("[DEBUG] Initializing NimTrainingVisualizer")
            
        self.title("Nim Training Progress")
        self.geometry("900x500")
        self.resizable(False, False)

        self.total_episodes = total_episodes
        self.max_sticks = max_sticks
        
        # Calculate update intervals so that plotting remains efficient
        self.update_every = max(1, total_episodes // 100)
        self.save_every = self.update_every
        
        if self.debug:
            print(f"[DEBUG] Total episodes: {total_episodes}, Update interval: {self.update_every}")
        
        self.win_rates = []   # List to store win rate values
        self.epsilons = []    # List to store epsilon values
        
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.font_axis = font.Font(family="Arial", size=10)
        self.font_values = font.Font(family="Arial", size=12, weight="bold")

        self.margin = 60
        self.width = 800
        self.height = 350

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self, textvariable=self.status_var, font=self.font_values)
        self.status_label.pack(pady=5)

        # Set up window close protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.draw_axes()
        
        # Process any pending GUI events to show the window
        self.update_idletasks()

    def draw_axes(self):
        """
        Draw X and Y axes with labels and tick marks.
        Y-axis from 0 to 1, X-axis labeled with episode numbers.
        """
        self.canvas.delete("all")

        m = self.margin
        w = self.width
        h = self.height

        # Draw Y-axis
        self.canvas.create_line(m, m, m, m + h, width=2)
        # Draw Y-axis ticks and labels at 0.00, 0.25, 0.50, 0.75, 1.00
        for i in range(5):
            y = m + h - i * (h / 4)
            val = i * 0.25
            self.canvas.create_line(m - 5, y, m, y, width=2)
            self.canvas.create_text(m - 30, y, text=f"{val:.2f}", font=self.font_axis)

        # Draw X-axis
        self.canvas.create_line(m, m + h, m + w, m + h, width=2)
        self.canvas.create_text(m + w / 2, m + h + 40, text="Episodes", font=self.font_axis)

        # Add legend
        legend_x = m + w - 150
        legend_y = m + 20
        self.canvas.create_line(legend_x, legend_y, legend_x + 20, legend_y, fill="blue", width=2)
        self.canvas.create_text(legend_x + 30, legend_y, text="Win Rate", anchor="w", font=self.font_axis)
        
        self.canvas.create_line(legend_x, legend_y + 20, legend_x + 20, legend_y + 20, fill="orange", width=2)
        self.canvas.create_text(legend_x + 30, legend_y + 20, text="Epsilon", anchor="w", font=self.font_axis)
        
        self.canvas.create_oval(legend_x + 5, legend_y + 35, legend_x + 15, legend_y + 45, outline="red", width=2)
        self.canvas.create_text(legend_x + 30, legend_y + 40, text="Max Win Rate", anchor="w", font=self.font_axis)

    def update(self, win_rate, epsilon):
        """
        Add new data points for win rate and epsilon, update status text,
        and redraw the plot. This method is called externally by the training script.
        
        Parameters:
        -----------
        win_rate : float
            Current win rate value (0 to 1).
        epsilon : float
            Current epsilon value (exploration rate, 0 to 1).
        """
        self.win_rates.append(win_rate)
        self.epsilons.append(epsilon)

        ep_count = len(self.win_rates) * self.update_every
        self.status_var.set(
            f"Episodes: {ep_count} / {self.total_episodes} | "
            f"Last Win Rate: {win_rate:.3f} | Epsilon: {epsilon:.3f}"
        )
        self.draw_plot()
        
        # Process GUI events to keep window responsive
        self.update_idletasks()
        
        if self.debug:
            print(f"[DEBUG] Updated plot: Win Rate = {win_rate:.3f}, Epsilon = {epsilon:.3f}")

    def update_plot(self, win_rate, epsilon):
        """
        Legacy method name - redirects to update() for backward compatibility.
        """
        self.update(win_rate, epsilon)

    def draw_plot(self):
        """
        Draw the win rate and epsilon lines with axis, labels,
        and highlight the maximum win rate point.
        """
        m = self.margin
        w = self.width
        h = self.height
        points = len(self.win_rates)

        self.draw_axes()

        if points == 0:
            if self.debug:
                print("[DEBUG] No points to plot")
            return

        # Draw X-axis labels at max 10 evenly spaced points
        max_labels = 10
        if points == 1:
            indices = [0]
        else:
            steps = min(points - 1, max_labels)
            indices = [int(i * (points - 1) / steps) for i in range(steps + 1)]

        for idx in indices:
            if points == 1:
                x = m + w / 2
            else:
                x = m + idx * (w / (points - 1))
            episode_num = (idx + 1) * self.update_every  # Fixed to show correct episode numbers
            self.canvas.create_line(x, m + h, x, m + h + 5)
            self.canvas.create_text(x, m + h + 25, text=str(episode_num), font=self.font_axis)

        # Draw win rate line in blue
        if points > 1:
            for i in range(1, points):
                x1 = m + (i - 1) * (w / (points - 1))
                y1 = m + h - (self.win_rates[i - 1] * h)
                x2 = m + i * (w / (points - 1))
                y2 = m + h - (self.win_rates[i] * h)
                self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
        else:
            x = m + w / 2
            y = m + h - (self.win_rates[0] * h)
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="blue", outline="")

        # Mark the maximum win rate point with a red circle
        if self.win_rates:  # Check if we have data
            max_val = max(self.win_rates)
            max_idx = self.win_rates.index(max_val)
            max_x = m + max_idx * (w / (points - 1)) if points > 1 else m + w / 2
            max_y = m + h - max_val * h
            self.canvas.create_oval(max_x - 6, max_y - 6, max_x + 6, max_y + 6, outline="red", width=3, fill="")

        # Draw epsilon line in orange
        if len(self.epsilons) > 1:
            for i in range(1, points):
                x1 = m + (i - 1) * (w / (points - 1))
                y1 = m + h - (self.epsilons[i - 1] * h)
                x2 = m + i * (w / (points - 1))
                y2 = m + h - (self.epsilons[i] * h)
                self.canvas.create_line(x1, y1, x2, y2, fill="orange", width=2)
        elif len(self.epsilons) == 1:
            x = m + w / 2
            y = m + h - (self.epsilons[0] * h)
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="orange", outline="")

    def close(self):
        """
        Close the visualizer window.
        """        
        if self.debug:
            print("[DEBUG] Closing visualizer")
        self.destroy()

    def on_closing(self):
        """
        Handle window close event.
        """
        self.close()


def create_training_visualizer(total_episodes, max_sticks=10):
    """
    Factory function to create a visualizer for training.
    Returns the visualizer instance ready to receive updates.
    """
    visualizer = NimTrainingVisualizer(total_episodes, max_sticks)
    return visualizer


def run_training_demo():
    """
    Demonstration function showing how to use the visualizer with simulated training data.
    This is for testing purposes only.
    """
    print("Running training demo - this is for testing only!")
    visualizer = create_training_visualizer(total_episodes=1000, max_sticks=10)
    
    # Simulate training updates
    def simulate_training():
        for i in range(100):  # 100 updates for 1000 episodes (every 10 episodes)
            win_rate = min(0.95, i / 100.0 + 0.1 + random.uniform(-0.1, 0.1))  # Gradually improving
            epsilon = max(0.01, 1.0 - (i / 100.0))  # Decaying epsilon
            
            visualizer.update(win_rate, epsilon)
            
            if i % 10 == 0:
                print(f"Demo update {i}: Win Rate = {win_rate:.3f}, Epsilon = {epsilon:.3f}")
            
            # Small delay to see the updates
            time.sleep(0.1)
        
        print("Demo completed - close window to exit")
    
    # Start simulation after window is shown
    visualizer.after(1000, simulate_training)
    
    # Start the GUI event loop
    try:
        visualizer.mainloop()
    except tk.TclError:
        pass  # Window was closed


if __name__ == "__main__":
    run_training_demo()