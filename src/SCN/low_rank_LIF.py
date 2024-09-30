import time

import matplotlib.axes
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt

from SCN import plot

from .utils_plots import _get_colors, _line_closest_point, _save_fig


class Low_rank_LIF:
    r"""
    Low-rank LIF model.

    :math:`N` neurons, :math:`d_i` input dimensions, :math:`d_o` output dimensions.

    :math:`\dot{\mathbf{V}}(t) = -\lambda \mathbf{V}(t) + \mathbf{F} \mathbf{c}(t) + \mathbf{W} \mathbf{s}(t)`
    where :math:`\lambda \in \mathbb{R}_+, \mathbf{V} \in \mathbb{R}^N, \mathbf{c} \in \mathbb{R}^{d_i},
    \mathbf{F} \in \mathbb{R}^{N \times d_i}, \mathbf{s} \in \mathbb{R}^N` is the spike train and
    :math:`\mathbf{W} \in \mathbb{R}^{N \times N}` fulfills :math:`\text{rank}(\mathbf{W}) = d_o < N`,
    i.e. :math:`W=ED` with :math:`\mathbf{E} \in \mathbb{R}^{N \times d_o}` and
    :math:`\mathbf{D} \in \mathbb{R}^{d_o \times N}`.

    This leads to the voltage equation:
    :math:`\mathbf{V}(t) = \mathbf{F} \mathbf{x}(t) + \mathbf{W} \mathbf{r}(t)` where
    :math:`\dot{\mathbf{x}}(t) = - \lambda \mathbf{x}(t) + \mathbf{c}(t)` and
    :math:`\dot{\mathbf{r}}(t) = - \lambda \mathbf{r}(t) + \mathbf{s}(t)`.

    See Also
    --------
    :class:`~SCN.autoencoder.Autoencoder` : Autoencoder specialization.

    References
    ----------
    Podlaski, W. F., & Machens, C. K. (2024). Approximating nonlinear functions with latent
    boundaries in low-rank excitatory-inhibitory spiking networks. Neural Computation, 36(5), 803-857.
    https://doi.org/10.1162/neco_a_01658

    """

    F: np.ndarray
    r"Forward weights of the network. :math:`N \times d_i`"

    E: np.ndarray
    r"Encoding weights of the network. :math:`N \times d_o`"

    D: np.ndarray
    r"Decoding weights of the network. :math:`d_o \times N`"

    W: np.ndarray
    r"Recurrent weights of the network. :math:`N \times N`. The weights are low-rank, i.e. :math:`W = ED`"

    lamb: float
    "Leak timescale of the network."

    T: np.ndarray
    r"Thresholds of the neurons. :math:`N \times 1`"

    def __init__(
        self,
        F: np.ndarray,
        E: np.ndarray,
        D: np.ndarray,
        T: np.ndarray,
        lamb: float = 1,
    ) -> None:
        """
        Constructor with specific parameters.

        Parameters
        ----------
        F : ndarray of shape (N, di)
            Forward weights of the network.

        E : ndarray of shape (N, do)
            Encoding weights of the network.

        D : ndarray of shape (do, N)
            Decoding weights of the network.

        T : ndarray of shape (N,)
            Threshold of the neurons.

        lamb : float, default=1
            Leak timescale of the network.

        """

        # dimensions
        self.di = F.shape[1]
        self.N = F.shape[0]
        self.do = E.shape[1]

        # asserts
        assert E.shape[0] == self.N, "E first dimension should be equal to N"
        assert D.shape[0] == self.do, "D first dim. should be equal to dim. of output"
        assert D.shape[1] == self.N, "D second dimension should be equal to N"
        assert T.shape[0] == self.N, "T first dimension should be equal to N"
        assert lamb > 0, "lamb should be positive"

        # parameters
        self.F = F
        self.E = E
        self.D = D
        self.T = T
        self.W = E @ D
        self.lamb = lamb

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        y_op: np.ndarray | None = None,
        y_op_lim: np.ndarray | None = None,
        save: bool = True,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list]:
        """
        Plot the network: bounding boundary (and trajectories)

        If x and y are passed, this is also plotted as trajectories in the bounding box.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes to plot to. If None, a new figure is created.

        x : ndarray of shape (di, time_steps), default=None
            Input trajectory to plot.

        y : ndarray of shape (do, time_steps), default=None
            Output trajectory to plot.

        y_op : ndarray of shape (do, time_steps), default=None
            Solution to the optimization problem with x(t) as input.

        y_op_lim : ndarray of shape (do, time_steps), default=None
            Solution to the optimization problem with x(t) as input, in the limit of small spikes.

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

        # Inhibitory standard
        centered = np.array([0, -1])
        x0 = x[:, -1]

        artists = []

        # plot the network
        if self.do == 2:
            artists = self._draw_bbox_2D(centered, x0, ax)
            # Y Trajectory
            if y is not None:
                artists_y = plot._plot_traj(ax, y, gradient=True)
                artists.append(artists_y)
                artists_leak = plot._plot_vector(ax, y[:, -1], -y[:, -1])
                artists.append(artists_leak)

            # y_op point
            if y_op is not None:
                artists_y_op = plot._plot_scatter(ax, y_op, marker="D")
                artists.append(artists_y_op)

            # y_op_lim point
            if y_op_lim is not None:
                artists_y_op_lim = plot._plot_scatter(ax, y_op_lim, marker="*", size=3)
                artists.append(artists_y_op_lim)
        else:
            raise NotImplementedError("Only 2D Latents vis. is implemented for now")

        fig = ax.get_figure()
        assert fig is not None
        if save:
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            _save_fig(fig, time_stamp + "-bounding-box.png")

        return fig, ax, artists

    def plot_rate_space(
        self,
        x: np.ndarray | None = None,
        ax: matplotlib.axes.Axes | None = None,
        r: np.ndarray | None = None,
        r_op: np.ndarray | None = None,
        r_op_lim: np.ndarray | None = None,
        save: bool = True,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list]:
        """
        Plot the network in rate space: boundaries (and trajectories). Only for N = 2 neurons.

        If r is passed, this is also plotted as a trajectory.

        Parameters
        ----------

        x : ndarray of shape (di, time_steps), default=None
            Input to the network.

        ax : matplotlib.axes.Axes, default=None
            Axes to plot to. If None, a new figure is created.

        r : ndarray of shape (N, time_steps), default=None
            Rates trajectory to plot.

        r_op : ndarray of shape (N, time_steps), default=None
            Solution to the optimization problem with x(t) as input.

        r_op_lim : ndarray of shape (N, time_steps), default=None
            Solution to the optimization problem with x(t) as input, in the limit of small spikes.

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
        if r is not None and r.ndim == 1:
            r = r[:, np.newaxis]

        x0 = x[:, -1]

        artists = []

        # plot the network
        if self.N == 2:
            artists = self._draw_rate_space_2D(x0, ax)
            # r Trajectory
            if r is not None:
                artists_r = plot._plot_traj(ax, r, gradient=True)
                artists.append(artists_r)
                artists_leak = plot._plot_vector(ax, r[:, -1], -r[:, -1])
                artists.append(artists_leak)

            # r_op point
            if r_op is not None:
                artists_r_op = plot._plot_scatter(ax, r_op, marker="D")
                artists.append(artists_r_op)

            # r_op_lim point
            if r_op_lim is not None:
                artists_r_op_lim = plot._plot_scatter(ax, r_op_lim, marker="*", size=3)
                artists.append(artists_r_op_lim)
        else:
            raise NotImplementedError("Only N=2 rate vis. is implemented for now")

        fig = ax.get_figure()
        assert fig is not None
        if save:
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            _save_fig(fig, time_stamp + "-rate-space.png")

        return fig, ax, artists

    def _animate(
        self,
        ax: matplotlib.axes.Axes,
        artists: list,
        x: np.ndarray,
        y: np.ndarray,
        y_op: np.ndarray | None = None,
        y_op_lim: np.ndarray | None = None,
        input_change: bool = False,
        spiking: np.ndarray | None = None,
    ) -> None:
        """
        Animate the network by modifying the artists.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot to.

        artists : list
            List of artists to modify.

        x : ndarray of shape (di, time_steps)
            Input trajectory to plot.

        y : ndarray of shape (do, time_steps)
            Output trajectory to plot.

        y_op : ndarray of shape (do, time_steps), default=None
            Solution to the optimization problem with x(t) as input.

        y_op_lim : ndarray of shape (do, time_steps), default=None
            Solution to the optimization problem with x(t) as input, in the limit of small spikes.

        input_change: bool, default=False
            If True, the input has changed.

        spiking : ndarray(int), default=None
            Neurons spiking in this frame. Index starting at 1. -n if the neuron needs to be restored.
        """

        offset = 0
        if y_op is not None:
            offset += 1
        if y_op_lim is not None:
            offset += 1

        plot._animate_traj(ax, artists[-2 - offset], y)
        plot._animate_vector(artists[-1 - offset], y[:, -1], -y[:, -1])
        if spiking is not None:
            plot._animate_spiking(artists, spiking)
        if input_change:
            centered = np.array([0, -1])
            x0 = x[:, -1]

            self._draw_bbox_2D(centered, x0, ax, artists)

            if y_op is not None:
                plot._animate_scatter(artists[-offset], y_op[:, -1:])
            if y_op_lim is not None:
                plot._animate_scatter(artists[-1], y_op_lim[:, -1:])

    def _animate_rate_space(
        self,
        ax: matplotlib.axes.Axes,
        artists: list,
        x: np.ndarray,
        r: np.ndarray,
        r_op: np.ndarray | None = None,
        r_op_lim: np.ndarray | None = None,
        input_change: bool = False,
        spiking: np.ndarray | None = None,
    ) -> None:
        """
        Animate the rate space by modifying the artists.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot to.

        artists : list
            List of artists to modify.

        x : ndarray of shape (di, time_steps)
            Input trajectory to plot.

        r : ndarray of shape (N, time_steps)
            Rate trajectory to plot.

        r_op : ndarray of shape (N, time_steps)
            Solution to the optimization problem with x(t) as input.

        r_op_lim : ndarray of shape (N, time_steps)
            Solution to the optimization problem with x(t) as input, in the limit of small spikes.

        input_change: bool, default=False
            If True, the input has changed.

        spiking : ndarray(int), default=None
            Neurons spiking in this frame. Index starting at 1. -n if the neuron needs to be restored.
        """

        offset = 0
        if r_op is not None:
            offset += 1
        if r_op_lim is not None:
            offset += 1

        plot._animate_traj(ax, artists[-2 - offset], r)
        plot._animate_vector(artists[-1 - offset], r[:, -1], -r[:, -1])
        if spiking is not None:
            plot._animate_spiking(artists, spiking)
        if input_change:
            x0 = x[:, -1]
            self._draw_rate_space_2D(x0, ax, artists)
            if r_op is not None:
                plot._animate_scatter(artists[-offset], r_op[:, -1:])
            if r_op_lim is not None:
                plot._animate_scatter(artists[-1], r_op_lim[:, -1:])

    def _draw_bbox_2D(
        self,
        centered: np.ndarray,
        x0: np.ndarray,
        ax: matplotlib.axes.Axes,
        artists: list | None = None,
    ) -> list:
        """
        Draw the bounding box visualization of the network.

        Parameters
        ----------
        centered : ndarray of shape (2,)
            Center of the bounding box.

        x0 : ndarray of shape (di,)
            Input of the network.

        ax : matplotlib.axes.Axes
            Axes to plot the network.

        artists : list, default = None
            List of artists to update the plot. If None, new artists are created.

        Returns
        -------
        artists : list
            List of artists to update the plot.

        """

        first_frame = artists is None

        if first_frame:
            artists = []

        colors = _get_colors(self.N, self.W)

        def line_func(y1: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return (-a * y1 - c) / b

        # TODO: Revisit where to center the plot
        y1x = np.linspace(centered[0] - 1, centered[0] + 1, 100)
        y2x = np.linspace(centered[1] - 1, centered[1] + 1, 100)
        for n in range(self.N):
            # TODO: This could be all that changes (a,b,c) so maybe this is where you need to separate
            a = self.E[n, 0]
            b = self.E[n, 1]
            c = -self.T[n] + self.F[n, :] @ x0
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
                    y2=centered[1] + np.sign(b),
                    color=colors[n],
                    interpolate=True,
                    alpha=0.2,
                    zorder=n,
                )
            else:
                poly = ax.fill_betweenx(
                    y2,
                    y1,
                    x2=centered[0] + np.sign(a),
                    color=colors[n],
                    interpolate=True,
                    alpha=0.2,
                    zorder=n,
                )
            if not first_frame:
                artists[n][0] = poly

            # line
            line = None
            if first_frame:
                line = ax.plot(y1, y2, linewidth=3, c=colors[n], zorder=n)[0]
            else:
                artists[n][1].set_xdata(y1)
                artists[n][1].set_ydata(y2)

            # quiver
            quiver = None
            q0, q1 = _line_closest_point(centered[0], centered[1], a, b, c)
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
                    zorder=n,
                )
            else:
                artists[n][2].set_offsets([q0, q1])
                artists[n][2].set_UVC(self.D[0, n], self.D[1, n])
                artists[n][2].set_zorder(n)

            if first_frame:
                artists.append([poly, line, quiver])

        ax.set_xlim(centered[0] - 1, centered[0] + 1)
        ax.set_ylim(centered[1] - 1, centered[1] + 1)
        ax.set_ylabel("y2")
        ax.set_xlabel("y1")
        ax.set_aspect("equal")

        return artists

    def _draw_rate_space_2D(
        self,
        x0: np.ndarray,
        ax: matplotlib.axes.Axes,
        artists: list | None = None,
    ) -> list:
        """
        Draw the rate space visualization of the network. For N = 2 neurons.

        Parameters
        ----------

        x0 : ndarray of shape (di,)
            Input of the network.

        ax : matplotlib.axes.Axes
            Axes to plot the network.

        artists : list, default = None
            List of artists to update the plot. If None, new artists are created.

        Returns
        -------
        artists : list
            List of artists to update the plot.

        """

        first_frame = artists is None

        if first_frame:
            artists = []

        colors = _get_colors(self.N, self.W)

        def line_func(y1: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return (-a * y1 - c) / b

        maxinter = 1
        a = np.zeros(self.N)
        b = np.zeros(self.N)
        c = np.zeros(self.N)
        for n in range(self.N):
            a[n] = self.W[n, 0]
            b[n] = self.W[n, 1]
            c[n] = -self.T[n] + self.F[n, :] @ x0
            if np.abs(a[n]) < 1e-3:
                maxinter = np.max([-c[n] / b[n], maxinter])
            elif np.abs(b[n]) < 1e-3:
                maxinter = np.max([-c[n] / a[n], maxinter])
            else:
                maxinter = np.max([-c[n] / a[n], -c[n] / b[n], maxinter])

        y1x = np.linspace(0, maxinter + 1, 100)
        y2x = np.linspace(0, maxinter + 1, 100)
        for n in range(self.N):
            diag = np.abs(a[n]) < np.abs(b[n])
            yo_p = (
                line_func(y1x, a[n], b[n], c[n])
                if diag
                else line_func(y2x, b[n], a[n], c[n])
            )
            # yo_p[yo_p < 0] = np.nan
            y1 = y1x if diag else yo_p
            y2 = yo_p if diag else y2x
            y1[y1 < 0] = 0
            y2[y2 < 0] = 0

            # polygon (to optimize: no redraw)
            if not first_frame:
                artists[n][0].remove()

            if diag:
                if (n == 0) * (self.W[n, n] > 0) or (n == 1) * (self.W[n, n] < 0):
                    poly = ax.fill_between(
                        y1,
                        0,
                        y2,
                        color=colors[n],
                        interpolate=True,
                        alpha=0.2,
                        zorder=n,
                    )
                else:
                    poly = ax.fill_between(
                        y1,
                        y2,
                        maxinter + 1,
                        color=colors[n],
                        interpolate=True,
                        alpha=0.2,
                        zorder=n,
                    )
            else:
                if (n == 0) * (self.W[n, n] > 0) or (n == 1) * (self.W[n, n] < 0):
                    poly = ax.fill_betweenx(
                        y2,
                        y1,
                        maxinter + 1,
                        color=colors[n],
                        interpolate=True,
                        alpha=0.2,
                        zorder=n,
                    )
                else:
                    poly = ax.fill_betweenx(
                        y2,
                        0,
                        y1,
                        color=colors[n],
                        interpolate=True,
                        alpha=0.2,
                        zorder=n,
                    )

            if not first_frame:
                artists[n][0] = poly

            # line
            line = None
            y1[y1 == 0] = np.nan
            y2[y2 == 0] = np.nan
            if first_frame:
                line = ax.plot(y1, y2, linewidth=3, c=colors[n], zorder=n)[0]
            else:
                artists[n][1].set_xdata(y1)
                artists[n][1].set_ydata(y2)

            # quiver
            quiver = None
            q0, q1 = _line_closest_point(0, 0, a[n], b[n], c[n])
            if q0 < 0 or q1 < 0:
                q0, q1 = _line_closest_point(0, maxinter + 1, a[n], b[n], c[n])
            if first_frame:
                quiver = ax.quiver(
                    q0,
                    q1,
                    not n,
                    n,
                    color=colors[n],
                    scale=5,
                    scale_units="xy",
                    angles="xy",
                    zorder=n,
                )
            else:
                artists[n][2].set_offsets([q0, q1])
                artists[n][2].set_UVC(not n, n)
                artists[n][2].set_zorder(n)

            if first_frame:
                artists.append([poly, line, quiver])

        ax.hlines(0, 0, maxinter + 1, color="k")
        ax.vlines(0, 0, maxinter + 1, color="k")
        ax.set_xlim(-0.02 * maxinter, maxinter + 1)
        ax.set_ylim(-0.02 * maxinter, maxinter + 1)
        ax.set_ylabel("r2")
        ax.set_xlabel("r1")
        ax.set_aspect("equal")

        return artists
