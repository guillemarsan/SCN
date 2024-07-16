import matplotlib.axes
import numpy as np

from .utils_plots import _gradient_line


def _plot_traj(
    ax: matplotlib.axes.Axes,
    traj: np.ndarray,
    gradient: bool = True,
) -> list:
    """
    Plot a trajectory.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis to plot the trajectory.

    traj: np.ndarray (2, time_steps)
        Trajectory to plot.

    gradient: bool, default = True
        If True, the trajectory is plotted with a gradient transparency.

    Returns
    -------
    artists: list
        List of artists to update the plot.
    """

    artists = []

    if gradient:
        final = ax.scatter(traj[0, -1], traj[1, -1], c="black")
        middle = _gradient_line(traj)
        ax.add_collection(middle)
        ini = ax.scatter(traj[0, 0], traj[1, 0], edgecolor="grey", facecolors="none")
    else:
        final = ax.scatter(traj[0, -1], traj[1, -1], c="blue")
        middle = ax.plot(traj[0, :], traj[1, :], color="blue", alpha=0.5)[0]
        ini = ax.scatter(traj[0, 0], traj[1, 0], edgecolor="blue", facecolors="none")

    artists.append(ini)
    artists.append(middle)
    artists.append(final)
    return artists


def _animate_traj(
    ax: matplotlib.axes.Axes,
    artists: list,
    traj: np.ndarray,
    gradient: bool = True,
) -> None:
    """
    Animate a trajectory.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis to plot the trajectory.

    artists: list
        List of artists to update the plot.

    traj: np.ndarray (2, time_steps)
        New trajectory to plot.

    gradient: bool, default = True
        If True, the trajectory is plotted with a gradient transparency.
    """

    if traj.ndim == 1:
        traj = traj[:, np.newaxis]

    if gradient:
        artists[0].set_offsets(traj[:, 0])
        artists[1].remove()
        gtraj = _gradient_line(traj, forget=True)
        ax.add_collection(gtraj)
        artists[1] = gtraj
        artists[2].set_offsets(traj[:, -1])
    else:
        artists[0].set_offsets(traj[:, 0])
        artists[1].set_xdata(traj[0, :])
        artists[1].set_ydata(traj[1, :])
        artists[2].set_offsets(traj[:, -1])


def _animate_spiking(
    artists: list,
    spiking: np.ndarray,
) -> None:
    """
    Animate the spiking effect of the network.

    Parameters
    ----------
    artists: list
        List of artists to update the plot.

    spiking: np.ndarray(int)
        Neurons spiking in this frame. Index starting at 1. -n if the neuron needs to be restored.
    """

    # spike effect
    for n in spiking:
        # restore normal look
        if n < 0:  # this weird indexing is to differentiate -0 from 0
            artists[-(n + 1)][1].set_linewidth(3)
        # spike look
        else:
            artists[n - 1][1].set_linewidth(10)


def _animate_axis(ax: matplotlib.axes.Axes, x0: np.ndarray, xf: np.ndarray) -> None:
    """
    Move the axis so x0 looks to be at xf.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis object to modify.

    x0: np.ndarray (2,)
        Point to move to xf.

    xf: np.ndarray (2,)
        Point to move x0 to.
    """
    # Axis ticks
    lm = np.ceil((xf[0] - 1) / 0.25) * 0.25
    nxticks = (
        np.arange(lm, lm + 2, 0.25)
        if (xf[0] - 1) % 0.25 != 0
        else np.arange(lm, lm + 2.0001, 0.25)
    )
    xticks = nxticks + (x0[0] - xf[0])
    ax.set_xticks(xticks)
    ax.set_xticklabels(nxticks)

    lm = np.ceil((xf[1] - 1) / 0.25) * 0.25
    nyticks = (
        np.arange(lm, lm + 2, 0.25)
        if (xf[1] - 1) % 0.25 != 0
        else np.arange(lm, lm + 2.0001, 0.25)
    )
    yticks = nyticks + (x0[1] - xf[1])
    ax.set_yticks(yticks)
    ax.set_yticklabels(nyticks)
