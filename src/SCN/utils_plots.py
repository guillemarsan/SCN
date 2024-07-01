import matplotlib.axes as mpl_axes
import matplotlib.colors as colors
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


def _trick_axis(ax: mpl_axes.Axes, x0: np.ndarray) -> None:
    """
    Move the axis so the (0,0) is at x0.

    Parameters
    ----------
    ax: mpl_axes.Axes
        Axis object to modify.

    x0: np.ndarray (2,)
        Point to move the (0,0) to.
    """
    # Axis ticks
    lm = np.ceil((x0[0] - 1) / 0.25) * 0.25
    nxticks = (
        np.arange(lm, lm + 2, 0.25)
        if (x0[0] - 1) % 0.25 != 0
        else np.arange(lm, lm + 2.0001, 0.25)
    )
    xticks = nxticks - x0[0]
    ax.set_xticks(xticks)
    ax.set_xticklabels(nxticks)

    lm = np.ceil((x0[1] - 1) / 0.25) * 0.25
    nyticks = (
        np.arange(lm, lm + 2, 0.25)
        if (x0[1] - 1) % 0.25 != 0
        else np.arange(lm, lm + 2.0001, 0.25)
    )
    yticks = nyticks - x0[1]
    ax.set_yticks(yticks)
    ax.set_yticklabels(nyticks)
