# cortex/viz_hex.py
# Minimal plotting of activations on a hex grid.

import numpy as np
import matplotlib.pyplot as plt

def axial_to_xy(q: int, r: int, size: float = 1.0):
    x = size * (1.5 * q)
    y = size * (np.sqrt(3) * (r + q / 2.0))
    return x, y

def plot_hex_activations(hexes, activations, edges=None, title=None):
    xs, ys = [], []
    for h in hexes:
        x, y = axial_to_xy(h.q, h.r)
        xs.append(x); ys.append(y)
    xs = np.array(xs); ys = np.array(ys)
    act = np.array(activations)

    plt.figure()
    sc = plt.scatter(xs, ys, s=60, c=act)
    plt.colorbar(sc)
    if edges:
        for (i, j, w) in edges:  # (region_i, region_j, weight in [0,1])
            x1, y1 = xs[i], ys[i]; x2, y2 = xs[j], ys[j]
            plt.plot([x1, x2], [y1, y2], alpha=float(w))
    if title: plt.title(title)
    plt.axis('equal'); plt.tight_layout()
    plt.show()
