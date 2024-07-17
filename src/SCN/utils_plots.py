import os

import matplotlib.animation
import matplotlib.axes
import matplotlib.colors as colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def _gradient_line(y: np.ndarray, forget: bool = False) -> LineCollection:
    """
    Create a LineCollection object with a gradient of transparency for a trajectory y

    Parameters
    ----------
    y: np.ndarray (2, time_steps)
        Trajectory to plot.

    forget: bool, default=False
        If True, the line will disappear with time.

    Returns
    -------
    lc: LineCollection
        LineCollection object with the gradient of transparency.
    """
    fsteps = 1000
    points = np.array([y[0, :], y[1, :]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(list(segments), colors="black", norm=colors.Normalize(0, 1))
    alphas = (
        np.concatenate([np.zeros(y.shape[1] - fsteps), np.linspace(0, 1, fsteps)])
        if forget and y.shape[1] > fsteps
        else np.linspace(0, 1, y.shape[1])
    )
    lc.set_alpha(list(alphas))
    lc.set_linewidth(2)
    return lc


def _line_closest_point(x0: float, y0: float, a: float, b: float, c: float):
    """
    Find the closest point to the line ax + by + c = 0 from the point (x0, y0)

    Parameters
    ----------
    x0: float
        x-coordinate of the point

    y0: float
        y-coordinate of the point

    a: float
        Coefficient of x in the line equation

    b: float
        Coefficient of y in the line equation

    c: float
        Constant term in the line equation

    Returns
    -------
    x: float
        x-coordinate of the closest point

    y: float
        y-coordinate of the closest point
    """

    denom = a**2 + b**2
    x = (b * (b * x0 - a * y0) - a * c) / denom
    y = (a * (-b * x0 + a * y0) - b * c) / denom

    return x, y


def _save_fig(fig: matplotlib.figure.Figure, name: str) -> None:
    """
    Save the figure.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        Figure to save.

    name: str
        Name of the file to save.
    """

    path = "./plots/"
    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(path + name, dpi=300, bbox_inches="tight", pad_inches=0.1)


def _save_ani(
    ani: matplotlib.animation.Animation, name: str, anim_freq: int = 10
) -> None:
    """
    Save the animation.

    Parameters
    ----------
    ani: matplotlib.animation.Animation
        Animation to save.

    name: str
        Name of the file to save.

    anim_freq: int, default=10
        Frame rate of the animation.
    """

    path = "./animations/"
    if not os.path.exists(path):
        os.makedirs(path)

    # TODO: make the path relative
    plt.rcParams["animation.ffmpeg_path"] = "/Program Files/ffmpeg/bin/ffmpeg"
    ani.save(
        path + name,
        writer="ffmpeg",
        fps=anim_freq,
    )


def _get_colors(N: int, W: np.ndarray) -> list:
    """
    Get the colors for the neurons.

    Non-daleian: cycling default colors
    Daelian: cool for I, hot for E

    Parameters
    ----------
    N: int
        Number of neurons.

    W: np.ndarray (N, N)
        Synaptic weights.

    Returns
    -------
    colors: list
        List of colors for the neurons.
    """

    excitatory = np.all(W >= 0, axis=0)
    inhibitory = np.all(W <= 0, axis=0)

    daleian = np.all(np.logical_or(excitatory, inhibitory))

    if daleian:
        cmapI = plt.get_cmap("winter")
        cmapE = plt.get_cmap("autumn")

        colors = [0] * N
        exc_idx = np.argwhere(excitatory).flatten()
        inh_idx = np.argwhere(inhibitory).flatten()
        colorsI = [cmapI(i) for i in np.linspace(0, 1, len(inh_idx))]
        colorsE = [cmapE(i) for i in np.linspace(0, 1, len(exc_idx))]
        colors = [colorsI.pop(0) if i in inh_idx else colorsE.pop(0) for i in range(N)]
    else:
        cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [cmap[i % 10] for i in range(N)]

    return colors
