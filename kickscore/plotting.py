import itertools
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_scores(
    model: Any,  # TODO should be `Model`, but this introduces circular dependency.
    items: Sequence[str],
    resolution: float | None = None,
    figsize: float | None = None,
    timestamps: bool = False,
) -> tuple[Figure, Axes]:
    colors = itertools.cycle(plt.cm.tab10(np.linspace(0, 1, 10)))  # pyright: ignore[reportAttributeAccessIssue]
    if resolution is None:
        first = min(obs.t for obs in model.observations)
        last = max(obs.t for obs in model.observations)
        resolution = 100 / (last - first)
    fig, ax = plt.subplots(figsize=figsize)
    for name in items:
        color = next(colors)
        ts, _, _ = model.item[name].scores
        first = min(ts)
        last = max(ts)
        ts = np.linspace(first, last, num=int(resolution * (last - first)))
        ms, vs = model.item[name].predict(ts)
        std = np.sqrt(vs)
        if timestamps:
            ts = [datetime.fromtimestamp(t) for t in ts]
        ax.plot(ts, ms, color=color, label=name)  # pyright: ignore[reportArgumentType]
        ax.fill_between(ts, ms - std, ms + std, color=color, alpha=0.1)  # pyright: ignore[reportArgumentType]
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.5)
    ax.legend()
    return fig, ax
