import os

import matplotlib.animation
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _gradient_line(y: np.ndarray, forget: bool = False) -> LineCollection:
    """
    Create a LineCollection object with a gradient of transparency for a trajectory y

    Parameters
    ----------
    y: np.ndarray (2 or 3, time_steps)
        Trajectory to plot.

    forget: bool, default=False
        If True, the line will disappear with time.

    Returns
    -------
    lc: LineCollection
        LineCollection object with the gradient of transparency.
    """

    fsteps = 1000
    if y.shape[0] == 2:
        points = np.array([y[0, :], y[1, :]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(list(segments), colors="black", norm=Normalize(0, 1))
    else:
        points = np.array([y[0, :], y[1, :], y[2, :]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(list(segments), linewidths=2, norm=Normalize(0, 1))

    alphas = (
        np.concatenate([np.zeros(y.shape[1] - fsteps), np.linspace(0, 1, fsteps)])
        if forget and y.shape[1] > fsteps
        else np.linspace(0, 1, y.shape[1])
    )

    if y.shape[0] == 2:
        lc.set_alpha(list(alphas))
        lc.set_linewidth(2)
    elif y.shape[0] == 3:
        colors = np.zeros((len(segments), 4))  # Create an array for RGBA colors
        colors[:, :3] = 0  # Set RGB to black
        for i in range(len(segments)):
            colors[i, -1] = alphas[i]  # Set the alpha channel
        lc.set_color([tuple(color) for color in colors])

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
    p: np.ndarray (2,)
        Coordinates of the closest point
    """

    denom = a**2 + b**2
    x = (b * (b * x0 - a * y0) - a * c) / denom
    y = (a * (-b * x0 + a * y0) - b * c) / denom

    return np.array([x, y])


def _plane_closest_point(
    x0: float, y0: float, z0: float, a: float, b: float, c: float, d: float
):
    """
    Find the closest point to the plane ax + by + cz + d = 0 from the point (x0, y0, z0)

    Parameters
    ----------
    x0: float
        x-coordinate of the point

    y0: float
        y-coordinate of the point

    z0: float
        z-coordinate of the point

    a: float
        Coefficient of x in the plane equation

    b: float
        Coefficient of y in the plane equation

    c: float
        Coefficient of z in the plane equation

    d: float
        Constant term in the plane equation

    Returns
    -------
    p: np.ndarray (3,)
        Coordinates of the closest point
    """

    denom = a**2 + b**2 + c**2
    numer = a * x0 + b * y0 + c * z0 + d
    k = numer / denom
    x = x0 - a * k
    y = y0 - b * k
    z = z0 - c * k

    return np.array([x, y, z])


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

    mockW = W.copy()
    mockW[np.abs(mockW) < 1e-10] = 0
    excitatory = np.all(mockW >= 0, axis=0)
    inhibitory = np.all(mockW <= 0, axis=0)

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


def _compute_arrow_segments(
    point: np.ndarray, vector: np.ndarray, scale: float
) -> list:
    # Calculate the arrowhead segments

    shaft_end = point + vector * scale
    arrow_length = np.linalg.norm(vector) * 0.28 * scale
    arrow_width = arrow_length * 0.27

    # Calculate the direction vectors for the arrowhead
    direction = vector / np.linalg.norm(vector)
    perp_vector1 = np.cross(direction, np.array([1, 0, 0]))
    if np.linalg.norm(perp_vector1) == 0:
        perp_vector1 = np.cross(direction, np.array([0, 1, 0]))
    perp_vector1 /= np.linalg.norm(perp_vector1)

    arrowhead1 = shaft_end - arrow_length * direction + arrow_width * perp_vector1
    arrowhead2 = shaft_end - arrow_length * direction - arrow_width * perp_vector1

    # Set the segments for the shaft and arrowheads
    segments = [
        [point, shaft_end],
        [shaft_end, arrowhead1],
        [shaft_end, arrowhead2],
    ]
    return segments
