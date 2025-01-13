import matplotlib.axes
import matplotlib.collections
import matplotlib.quiver
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .utils_plots import _compute_arrow_segments, _gradient_line


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
        if traj.shape[0] == 2:
            final = ax.scatter(traj[0, -1], traj[1, -1], c="black")
            middle = _gradient_line(traj)
            ax.add_collection(middle)
            ini = ax.scatter(
                traj[0, 0], traj[1, 0], edgecolor="grey", facecolors="none"
            )
        else:
            assert isinstance(ax, Axes3D)
            final = ax.scatter(traj[0, -1], traj[1, -1], traj[2, -1], c="black")
            middle = _gradient_line(traj)
            ax.add_collection3d(middle)
            ini = ax.scatter(
                traj[0, 0], traj[1, 0], traj[2, 0], edgecolor="grey", facecolors="none"
            )
    else:
        if traj.shape[0] == 2:
            final = ax.scatter(traj[0, -1], traj[1, -1], c="blue")
            middle = ax.plot(traj[0, :], traj[1, :], color="blue", alpha=0.5)[0]
            ini = ax.scatter(
                traj[0, 0], traj[1, 0], edgecolor="blue", facecolors="none"
            )
        else:
            final = ax.scatter(traj[0, -1], traj[1, -1], traj[2, -1], c="blue")
            middle = ax.plot(
                traj[0, :], traj[1, :], traj[2, :], color="blue", alpha=0.5
            )[0]
            ini = ax.scatter(
                traj[0, 0], traj[1, 0], traj[2, 0], edgecolor="blue", facecolors="none"
            )

    artists.append(ini)
    artists.append(middle)
    artists.append(final)
    return artists


def _plot_small_vector(
    ax: matplotlib.axes.Axes,
    point: np.ndarray,
    vector: np.ndarray,
) -> matplotlib.quiver.Quiver:
    """
    Plot a small vector (e.g. leak)

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis to plot the vector.

    point: np.ndarray (2,) or (3,)
        Point to start the vector.

    vector: np.ndarray (2,) or (3,)
        Vector to plot.

    Returns
    -------
    artists: matplotlib.quiver.Quiver
        Artists to update the plot.
    """

    if point.shape[0] == 2:
        arrow = ax.quiver(
            point[0],
            point[1],
            vector[0],
            vector[1],
            scale=6,
            scale_units="xy",
            angles="xy",
            color="grey",
            alpha=0.3,
        )
    else:
        scale_factor = 0.2
        vector *= scale_factor
        arrow = ax.quiver(
            point[0],
            point[1],
            point[2],
            vector[0],
            vector[1],
            vector[2],
            color="grey",
            alpha=0.3,
        )

    return arrow


def _plot_big_vector(
    ax: matplotlib.axes.Axes,
    point: np.ndarray,
    vector: np.ndarray,
    color: str | np.ndarray = "grey",
    on: bool = True,
) -> matplotlib.quiver.Quiver:
    """
    Plot a big vector (e.g. neurons)

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis to plot the vector.

    point: np.ndarray (2,) or (3,)
        Point to start the vector.

    vector: np.ndarray (2,) or (3,)
        Vector to plot.

    color: str or np.ndarray, default = "grey"
        Color of the vector.

    on: bool, default = True
        If True, the vector will be plotted. If False, it will disappear.

    Returns
    -------
    artists: matplotlib.quiver.Quiver
        Artists to update the plot.
    """

    if point.shape[0] == 2:
        arrow = ax.quiver(
            point[0],
            point[1],
            vector[0],
            vector[1],
            scale=6,
            scale_units="xy",
            angles="xy",
            color=color,
            alpha=1 if on else 0,
        )
    else:
        arrow = ax.quiver(
            point[0],
            point[1],
            point[2],
            vector[0],
            vector[1],
            vector[2],
            color=color,
            alpha=1 if on else 0,
        )

    return arrow


def _plot_scatter(
    ax: matplotlib.axes.Axes,
    points: np.ndarray,
    marker: str = "o",
    size: int = 1,
    zorder: int = 10,
) -> matplotlib.collections.PathCollection:
    """
    Plot a scatter of points.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis to plot the vector.

    points: np.ndarray (2 or 3, n_points)
        Points to plot.

    marker: str, default = "o"
        Marker to use in the scatter plot.

    size: int, default = 10
        Size of the markers.

    zorder: int, default = 1
        Zorder of the markers.

    Returns
    -------
    artists: matplotlib.collections.PathCollection
        Artists to update the plot.
    """

    # Get the size of the plot in pixels
    assert ax.figure is not None
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width, height = bbox.width * ax.figure.dpi, bbox.height * ax.figure.dpi

    # Calculate marker size relative to the plot size
    relative_size = min(width, height) * 0.25 * size  # Adjust the factor as needed

    if points.shape[0] == 2:
        scatter = ax.scatter(
            points[0, :],
            points[1, :],
            marker=marker,
            facecolor="white",
            edgecolor="black",
            s=relative_size,
            zorder=zorder,
        )
    else:
        scatter = ax.scatter(
            points[0, :],
            points[1, :],
            points[2, :],
            marker=marker,
            facecolor="white",
            edgecolor="black",
            s=relative_size,  # type: ignore
            zorder=zorder,
        )

    return scatter


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
        if traj.shape[0] == 2:
            artists[0].set_offsets(traj[:, 0])
            artists[1].remove()
            gtraj = _gradient_line(traj, forget=True)
            ax.add_collection(gtraj)
            artists[1] = gtraj
            artists[2].set_offsets(traj[:, -1])
        elif traj.shape[0] == 3:
            assert isinstance(ax, Axes3D)
            artists[0]._offsets3d = (
                np.array([traj[0, 0]]),
                np.array([traj[1, 0]]),
                np.array([traj[2, 0]]),
            )
            artists[1].remove()
            gtraj = _gradient_line(traj, forget=True)
            ax.add_collection3d(gtraj)
            artists[1] = gtraj
            artists[2]._offsets3d = (
                np.array([traj[0, -1]]),
                np.array([traj[1, -1]]),
                np.array([traj[2, -1]]),
            )
    else:
        if traj.shape[0] == 2:
            artists[0].set_offsets(traj[:, 0])
            artists[1].set_xdata(traj[0, :])
            artists[1].set_ydata(traj[1, :])
            artists[2].set_offsets(traj[:, -1])
        elif traj.shape[0] == 3:
            artists[0]._offsets3d = (
                np.array([traj[0, 0]]),
                np.array([traj[1, 0]]),
                np.array([traj[2, 0]]),
            )
            artists[1].set_data(traj[0, :], traj[1, :])
            artists[1].set_3d_properties(traj[2, :])
            artists[2]._offsets3d = (
                np.array([traj[0, -1]]),
                np.array([traj[1, -1]]),
                np.array([traj[2, -1]]),
            )


def _animate_small_vector(
    artists: matplotlib.quiver.Quiver,
    point: np.ndarray,
    vector: np.ndarray,
) -> None:
    """
    Animate a small vector (e.g. leak).

    Parameters
    ----------
    artists: matplotlib.quiver.Quiver
        Artists to update the plot.

    point: np.ndarray (2,)
        Point to start the vector.

    vector: np.ndarray (2,)
        Vector to plot.
    """

    if point.shape[0] == 2:
        artists.set_offsets([point[0], point[1]])
        artists.set_UVC(vector[0], vector[1])
    elif point.shape[0] == 3:
        assert isinstance(artists, Line3DCollection)
        artists._offsets3d = (  # type: ignore
            np.array([point[0]]),
            np.array([point[1]]),
            np.array([point[2]]),
        )
        scale = 0.2
        artists.set_segments(
            [
                [
                    [point[0], point[1], point[2]],
                    [
                        point[0] + scale * vector[0],
                        point[1] + scale * vector[1],
                        point[2] + scale * vector[2],
                    ],
                ]
            ]
        )


def _animate_big_vector(
    artists: matplotlib.quiver.Quiver,
    point: np.ndarray,
    vector: np.ndarray | None = None,
    on: bool = True,
) -> None:
    """
    Animate a big vector (e.g. neurons).

    Parameters
    ----------
    artists: matplotlib.quiver.Quiver
        Artists to update the plot.

    point: np.ndarray (2,)
        Point to start the vector.

    vector: np.ndarray (2,)
        Vector to plot. If None, the vector will just be translated.

    on: bool, default = True
        If True, the vector will be plotted. If False, it will disappear.
    """

    if point.shape[0] == 2:
        artists.set_offsets([point[0], point[1]])
        if vector is not None:
            artists.set_UVC(vector[0], vector[1])
    elif point.shape[0] == 3:
        assert isinstance(artists, Line3DCollection)
        artists._offsets3d = (  # type: ignore
            np.array([point[0]]),
            np.array([point[1]]),
            np.array([point[2]]),
        )

        if vector is not None:
            artists.set_segments(_compute_arrow_segments(point, vector, 1))

    if not on:
        artists.set_alpha(0)
    else:
        artists.set_alpha(1)


def _animate_scatter(
    artists: matplotlib.collections.PathCollection, points: np.ndarray
) -> None:
    """
    Animate a scatter of points.

    Parameters
    ----------
    artists: matplotlib.collections.PathCollection
        Artists to update the plot.

    points: np.ndarray (2, n_points)
        Points to plot.
    """
    if points.shape[0] == 2:
        artists.set_offsets(points.T)
    else:
        artists._offsets3d = (  # type: ignore
            np.array(points[0, :]),
            np.array(points[1, :]),
            np.array(points[2, :]),
        )


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
            (
                artists[-(n + 1)][1].set_linewidth(3)
                if len(artists[-(n + 1)]) == 3
                else artists[-(n + 1)][0].set_alpha(0.2)
            )
        # spike look
        else:
            (
                artists[n - 1][1].set_linewidth(10)
                if len(artists[n - 1]) == 3
                else artists[n - 1][0].set_alpha(0.7)
            )


def _animate_axis(ax: matplotlib.axes.Axes, x0: np.ndarray, xf: np.ndarray) -> None:
    """
    Move the axis so x0 looks to be at xf.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis object to modify.

    x0: np.ndarray (2,) or (3,)
        Point to move to xf.

    xf: np.ndarray (2,) or (3,)
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

    if x0.shape[0] == 3:
        lm = np.ceil((xf[2] - 1) / 0.25) * 0.25

        nzticks = (
            np.arange(lm, lm + 2, 0.25)
            if (xf[2] - 1) % 0.25 != 0
            else np.arange(lm, lm + 2.0001, 0.25)
        )
        zticks = nzticks + (x0[2] - xf[2])
        assert isinstance(ax, Axes3D)
        ax.set_zticks(zticks)  # type: ignore
        ax.set_zticklabels(nzticks)  # type: ignore
