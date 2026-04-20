"""Basic usage demo."""

import tkinter as tk
import numpy as np
from tkfastscatter import ScatterWidget

root = tk.Tk()
root.title("tkfastscatter — 250k points")
root.geometry("1000x750")

widget = ScatterWidget(root, fps=60)
widget.pack(fill="both", expand=True)

rng = np.random.default_rng(42)

# 250k points in a twisted torus-ish shape
N = 250_000
theta = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
phi   = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
r     = (2.0 + np.cos(phi)).astype(np.float32)

x = (r * np.cos(theta)).astype(np.float32)
y = (r * np.sin(theta)).astype(np.float32)
z = (np.sin(phi)).astype(np.float32)

pts = np.stack([x, y, z], axis=1)               # (N, 3)
scalars = np.sqrt(x**2 + y**2).astype(np.float32)  # color by radial distance

widget.set_points(pts, scalars=scalars, colormap="plasma", point_size=3.0)

# Controls: left-drag = orbit  |  middle-drag = pan  |  scroll = zoom  |  double-click = reset
root.mainloop()
