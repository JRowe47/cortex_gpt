# viz_hex.py
import matplotlib.pyplot as plt
import numpy as np

def axial_to_xy(q, r, size=1.0):
    x = size * (3/2 * q)
    y = size * (np.sqrt(3)/2 * (2*r + q))
    return x, y

def plot_hex_activations(hexes, activations, edges=None, title=None):
    xs, ys = [], []
    for h in hexes:
        x, y = axial_to_xy(h.q, h.r)
        xs.append(x); ys.append(y)
    xs = np.array(xs); ys = np.array(ys)
    act = np.array(activations)

    plt.figure()
    plt.scatter(xs, ys, s=60, c=act)
    if edges:
        for (i, j, w) in edges:    # (region_i, region_j, weight in [0,1])
            x1, y1 = xs[i], ys[i]; x2, y2 = xs[j], ys[j]
            plt.plot([x1, x2], [y1, y2], alpha=float(w))
    if title: plt.title(title)
    plt.axis('equal'); plt.tight_layout()
    plt.show()
