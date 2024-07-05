import time
from typing import Self

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from . import boundary
from .low_rank_LIF import Low_rank_LIF
from .utils_plots import _gradient_line, _line_closest_point, _save_fig, _trick_axis


class Autoencoder(Low_rank_LIF):
    r"""
    Autoencoder model. Subclass of the Low-rank LIF model with :math:`\mathbf{F} = -\mathbf{E} = \mathbf{D}^\top`.

    :math:`N` neurons, :math:`d_i = d_o` signal dimensions.

    This leads to the voltage equation:
    :math:`\mathbf{V}(t) = \mathbf{D}^\top (\mathbf{x}(t) - \mathbf{D} \mathbf{r}(t))`.

    See Also
    --------
    :class:`~SCN.low_rank_LIF.Low_rank_LIF` : Parent model Low_rank_LIF.

    Notes
    -----
    Blabla.

    References
    ----------
    Calaim, Nuno, Florian Alexander Dehmelt, Pedro J Gonçalves, and Christian K Machens.
    “Robustness in Spiking Networks: a Geometric Perspective,” 2021.
    https://doi.org/10.1101/2020.06.15.148338.


    Denève, Sophie, and Christian K. Machens. “Efficient Codes and Balanced Networks.”
    Nature Neuroscience 19, no. 3 (February 23, 2016): 375–82. https://doi.org/10.1038/nn.4243.

    Boerlin, Martin, Christian K. Machens, and Sophie Denève. “Predictive Coding of Dynamical
    Variables in Balanced Spiking Networks.” PLoS Computational Biology 9, no. 11 (November 2013).
    https://doi.org/10.1371/journal.pcbi.1003258.

    Examples
    --------
    >>> from SCN import Autoencoder
    >>> from SCN import Simulation
    >>> net.plot()
    ...
    >>> sim = Simulation()
    >>> x = np.tile([[0.5], [1]], (1, 10000))
    >>> sim.run(net, x)
    >>> sim.animate()
    """

    D: np.ndarray
    r"Weights of the network. :math:`d_o = d_i \times N`"

    F: np.ndarray
    r"Forward weights of the network. :math:`N \times d_i`. In Autoencoder, :math:`F= D^\top`"

    E: np.ndarray
    r"Encoding weights of the network. :math:`N \times d_o`. In Autoencoder, :math:`E= -D^\top`"

    W: np.ndarray
    r"Recurrent weights of the network. :math:`N \times N`. The weights are low-rank, i.e. :math:`W = -D^\top D`"

    lamb: float
    "Leak timescale of the network."

    T: np.ndarray
    r"Thresholds of the neurons. :math:`N \times 1`"

    def __init__(
        self,
        D: np.ndarray,
        T: int | float | np.ndarray | None = None,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
    ) -> None:
        r"""
        Constructor with specific parameters.

        Parameters
        ----------
        D : ndarray of shape (do, N)
            Weights of the network. The columns need to be normalized.

        T : ndarray of shape (N,)
            Threshold of the neurons. If None, :math:`T_i = 1/2`.

        lamb : float, default=1
            Leak timescale of the network.

        spike_scale : int, float or ndarray, default=1
            Scale of the spikes.
        """

        assert all(np.abs(np.linalg.norm(D, axis=0) - 1) < 1e-10), (
            "The columns of D need to be normalized, if you want to change the boundary"
            + "use T, if you want to change the spike size use spike_scale"
        )

        if spike_scale is not None:
            assert (
                isinstance(spike_scale, (int, float))
                or spike_scale.shape[0] == D.shape[1]
            ), "spike_scale is a single value or an array with as many values as neurons"

        F = D.T
        E = -D.T

        if T is None:
            T = 0.5
        elif isinstance(T, (int, float)):
            T = T * np.ones(D.shape[1])

        D = spike_scale * D

        assert T is not None and not isinstance(T, (int, float))
        super().__init__(F=F, E=E, D=D, T=T, lamb=lamb)

    @classmethod
    def init_random(  # type: ignore[reportIncompatibleMethodOverride]
        cls,
        d: int = 2,
        N: int = 10,
        seed: int | None = None,
        T: int | float | np.ndarray | None = None,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
    ) -> Self:
        r"""
        Random initialization of the Autoencoder network.
        (see :func:`~SCN.boundary._sphere_random`)

        Parameters
        ----------
        d : float, default=1
            Input dimensions.

        N : float, default=10
            Number of neurons.

        seed : int or None, default=None
            Seed for the random number generator.

        T : ndarray of shape (N,)
            Threshold of the neurons. If None, :math:`T_i = 1/2`.

        lamb : float, default=1
            Leak timescale of the network.

        spike_scale : int, float or ndarray, default=1
            Scale of the spikes.

        Returns
        -------
        net: Autoencoder
            Autoencoder network with random decoders.
        """

        # random parameters
        D = boundary._sphere_random(d=d, N=N, seed=seed)

        return cls(D, T, lamb, spike_scale)

    @classmethod
    def init_cube(
        cls,
        d: int = 2,
        one_quadrant: bool = False,
        T: int | float | np.ndarray | None = None,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
    ) -> Self:
        r"""
        Hypercube initialization of the Autoencoder network.

        Parameters
        ----------
        d : float, default=1
            Input dimensions.

        one_quadrant : bool, default=False
            If True, the weights are in the first quadrant.

        T : ndarray of shape (N,)
            Threshold of the neurons. If None, :math:`T_i = 1/2`.

        lamb : float, default=1
            Leak timescale of the network.

        spike_scale : int, float or ndarray, default=1
            Scale of the spikes.

        Returns
        -------
        net: Autoencoder
            Autoencoder network with hypercube decoders.
        """

        # hypercube parameters
        D = boundary._cube(d=d, one_quadrant=one_quadrant)

        return cls(D, T, lamb, spike_scale)

    @classmethod
    def init_2D_spaced(
        cls,
        N: int = 10,
        angle_range: list | None = None,
        T: int | float | np.ndarray | None = None,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
    ) -> Self:
        r"""
        Regularly spaced 2D initialization of the Autoencoder network.

        :math:`N` neurons spaced regularly between `angle_range[0]` and `angle_range[1]`.
        The decoders are :math:`\mathbf{D}_i = (-\cos(\alpha_i), -\sin(\alpha_i))`.

        Parameters
        ----------
        N: int, default=10
            Number of neurons.

        angle_range : list, default=None
            Range of angles for the neurons. If None, the range is :math:`[0, 2 \pi]`.

        T : ndarray of shape (N,)
            Threshold of the neurons. If None, :math:`T_i = 1/2`.

        lamb : float, default=1
            Leak timescale of the network.

        spike_scale : int, float or ndarray, default=1
            Scale of the spikes.

        Returns
        -------
        net: Autoencoder
            Autoencoder network with regularly spaced decoders.
        """

        # evenly spaced circular parameters
        D = boundary._2D_circle_spaced(N=N, angle_range=angle_range)

        return cls(D, T, lamb, spike_scale)

    @classmethod
    def init_2D_random(
        cls,
        N: int = 10,
        angle_range: list | None = None,
        seed: int | None = None,
        T: int | float | np.ndarray | None = None,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
    ) -> Self:
        r"""
        Randomly spaced 2D initialization of the Autoencoder network.

        :math:`N` neurons spaced randomly between `angle_range[0]` and `angle_range[1]`.
        The decoders are :math:`\mathbf{D}_i = (-\cos(\alpha_i), -\sin(\alpha_i))`.

        Parameters
        ----------
        N: int, default=10
            Number of neurons.

        angle_range : list, default=None
            Range of angles for the neurons. If None, the range is :math:`[0, 2 \pi]`.

        seed : int or None, default=None
            Seed for the random number generator.

        T : ndarray of shape (N,)
            Threshold of the neurons. If None, :math:`T_i = 1/2`.

        lamb : float, default=1
            Leak timescale of the network.

        spike_scale : int, float or ndarray, default=1
            Scale of the spikes.

        Returns
        -------
        net: Autoencoder
            Autoencoder network with randomly spaced decoders.
        """

        # randomly spaced circular parameters
        D = boundary._2D_circle_random(N=N, angle_range=angle_range, seed=seed)

        return cls(D, T, lamb, spike_scale)

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        inverse: bool = False,
        save: bool = True,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list]:
        """
        Plot the network: bounding box (and trajectories)

        If x and y are passed, this is also plotted as trajectories in the bounding box.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes to plot to. If None, a new figure is created.

        x : ndarray of shape (di, time_steps), default=None
            Input trajectory to plot.

        y : ndarray of shape (do, time_steps), default=None
            Output trajectory to plot.

        inverse : bool, default=False
            If True, the network is plotted centered at (0,0) and the axis moved to x0.

        save : bool, default=True
            If True, the figure is saved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure of the plot.

        ax : matplotlib.axes.Axes
            Axes of the plot.

        artists : list
            List of artists in the plot.
        """

        if ax is None:
            ax = plt.figure(figsize=(10, 10)).gca()

        if x is None:
            x = np.zeros((self.di, 1))
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y is not None and y.ndim == 1:
            y = y[:, np.newaxis]

        if inverse:
            x = x - x[:, -1:]
            y = y - x[:, -1:]

        # Bounding box
        artists = self._draw_bbox(x0=x[:, -1], ax=ax)

        # X Trajectory
        xfinal = ax.scatter(x[0, -1], x[1, -1], c="blue")
        xtraj = ax.plot(x[0, :], x[1, :], color="blue", alpha=0.5)[0]
        xini = ax.scatter(x[0, 0], x[1, 0], edgecolor="blue", facecolors="none")
        artists.append([xini, xtraj, xfinal])

        # Y Trajectory
        if y is not None:

            yfinal = ax.scatter(y[0, -1], y[1, -1], c="black")
            ytraj = _gradient_line(y)
            ax.add_collection(ytraj)
            yini = ax.scatter(y[0, 0], y[1, 0], edgecolor="grey", facecolors="none")
            artists.append([yini, ytraj, yfinal])

        if inverse:
            _trick_axis(ax, x[:, -1])

        fig = ax.get_figure()
        assert fig is not None
        if save:
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            _save_fig(fig, time_stamp + "-autoencoder.png")

        return fig, ax, artists

    def _animate(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        artists: list,
        spiking: np.ndarray,
    ) -> None:
        """
        Animate the network by modifying the artists.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot to.

        x : ndarray of shape (di, time_steps)
            Input trajectory to plot.

        y : ndarray of shape (do, time_steps)
            Output trajectory to plot.

        artists : list
            List of artists to modify.

        spiking : ndarray(int)
            Neurons spiking in this frame. Index starting at 1. -n if the neuron needs to be restored.
        """

        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]

        xinv = x - x[:, -1:]
        yinv = y - x[:, -1:]

        # x traj
        artists[-2][0].set_offsets(xinv[:, 0])
        artists[-2][1].set_xdata(xinv[0, :])
        artists[-2][1].set_ydata(xinv[1, :])

        # y traj
        artists[-1][0].set_offsets(yinv[:, 0])

        artists[-1][1].remove()
        ytraj = _gradient_line(yinv, forget=True)
        ax.add_collection(ytraj)
        artists[-1][1] = ytraj

        artists[-1][2].set_offsets(yinv[:, -1])

        # move axes
        _trick_axis(ax, x[:, -1])

        # spike effect
        for n in spiking:
            # restore normal look
            if n < 0:  # this weird indexing is to differentiate -0 from 0
                artists[-(n + 1)][1].set_linewidth(3)
            # spike look
            else:
                artists[n - 1][1].set_linewidth(10)

    def _draw_bbox(
        self,
        x0: np.ndarray,
        ax: matplotlib.axes.Axes,
        artists: list | None = None,
    ) -> list:
        """
        Plot the bounding box of an Autoencoder network.

        Parameters
        ----------
        x0 : ndarray of shape (di,)
            Center of the bounding box.

        ax : matplotlib.axes.Axes
            Axes to plot to.

        artists : list, default=None
            List of artists to modify. If None, new artists are created.

        Returns
        -------
        artists : list
            List of artists in the plot. Either the same as input or the new ones.
        """
        first_frame = artists is None

        if first_frame:
            artists = []
        # plot the network
        if self.di == 2:

            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            def line_func(y1: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
                return (-a * y1 - c) / b

            y1x = np.linspace(x0[0] - 1, x0[0] + 1, 100)
            y2x = np.linspace(x0[1] - 1, x0[1] + 1, 100)
            for n in range(self.N):
                a = self.F[n, 0]
                b = self.F[n, 1]
                c = self.T[n] - self.F[n, :] @ x0
                yo = (
                    line_func(y1x, a, b, c)
                    if np.abs(a) < np.abs(b)
                    else line_func(y2x, b, a, c)
                )
                y1 = y1x if np.abs(a) < np.abs(b) else yo
                y2 = yo if np.abs(a) < np.abs(b) else y2x

                # polygon (to optimize: no redraw)
                if not first_frame:
                    artists[n][0].remove()
                if np.abs(a) < np.abs(b):
                    poly = ax.fill_between(
                        y1,
                        y2,
                        y2=x0[1] - np.sign(b),
                        color=colors[n],
                        interpolate=True,
                        alpha=0.2,
                    )
                else:
                    poly = ax.fill_betweenx(
                        y2,
                        y1,
                        x2=x0[0] - np.sign(a),
                        color=colors[n],
                        interpolate=True,
                        alpha=0.2,
                    )
                if not first_frame:
                    artists[n][0] = poly

                # line
                line = None
                if first_frame:
                    line = ax.plot(y1, y2, linewidth=3, c=colors[n])[0]
                else:
                    artists[n][1].set_xdata(y1)
                    artists[n][1].set_ydata(y2)

                # quiver
                quiver = None
                q0, q1 = _line_closest_point(x0[0], x0[1], a, b, c)
                if first_frame:
                    quiver = ax.quiver(
                        q0,
                        q1,
                        self.D[0, n],
                        self.D[1, n],
                        color=colors[n],
                        scale=5,
                        scale_units="xy",
                        angles="xy",
                    )
                else:
                    artists[n][2].set_offsets([q0, q1])
                    artists[n][2].set_UVC(self.D[0, n], self.D[1, n])

                if first_frame:
                    artists.append([poly, line, quiver])

            ax.set_xlim(x0[0] - 1, x0[0] + 1)
            ax.set_ylim(x0[1] - 1, x0[1] + 1)
            ax.set_aspect("equal")

        return artists
