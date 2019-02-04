import itertools
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime


def plot_scores(model, items, resolution=None, figsize=None, timestamps=False):
    colors = itertools.cycle(plt.cm.tab10(np.linspace(0, 1, 10)))
    if resolution is None:
        first = min(obs.t for obs in model.observations)
        last = max(obs.t for obs in model.observations)
        resolution = 100 / (last - first)
    fig, ax = plt.subplots(figsize=figsize)
    for name in items:
        color = next(colors)
        first = min(obs.t for obs in model.item[name].observations)
        last = max(obs.t for obs in model.item[name].observations)
        ts = np.linspace(first, last, num=int(resolution*(last - first)))
        ms, vs = model.item[name].predict(ts)
        std = np.sqrt(vs)
        if timestamps:
            ts = [datetime.fromtimestamp(t) for t in ts]
        ax.plot(ts, ms, color=color, label=name)
        ax.fill_between(ts, ms-std, ms+std, color=color, alpha=0.1)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.5)
    ax.legend()
    return fig, ax
