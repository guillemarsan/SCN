import time
from typing import Self

import matplotlib.axes
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from SCN import plot

from .plot import _animate_big_vector, _plot_big_vector
from .utils_neuro import _canon_symmetric
from .utils_plots import (
    _get_colors,
    _line_closest_point,
    _plane_closest_point,
    _save_fig,
)


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

    @classmethod
    def init_optim(
        cls,
        Q: np.ndarray,
        E: np.ndarray,
        T: int | float | np.ndarray = 0.5,
        di: int = 1,
        Fseed: int | None = None,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
    ) -> Self:
        r"""
        Optimization initialization of the Low-rank LIF network.

        Network associated with the optimization problem :math: `\text{optim}^Q_y \frac{1}{2} y^\top Q y \; \text{s.t.} \;
        Ey \leq T - Fx`.

        Parameters
        ----------

        Q: np.ndarray
            Quadratic term of the optimization problem. Symmetric matrix.

        E: np.ndarray
            Linear coefficients of the problem constraints. Encoding weights of the network.

        T: int | float | np.ndarray, default=0.5
            Bias of the problem constraints. Threshold of the neurons.

        di: int, default=1
            Input dimensions.

        Fseed: int | None, default=None
            Seed for the random number generator for determining the sign of :math:`\mathbf{F}`.

        lamb : float, default=1
            Leak timescale of the network.

        spike_scale : int, float or ndarray, default=1
            Scale of the spikes.

        Returns
        -------
        net: low_rank_LIF
            low_rank_LIF network that is associated with the given optimization problem.
        """

        assert np.allclose(Q, Q.T), "Q needs to be symmetric"
        assert (
            E.shape[1] == Q.shape[0]
        ), "E needs to have as many columns as the dimension of Q"
        assert (
            type(T) is not np.ndarray or T.shape[0] == E.shape[0]
        ), "T has to be the same size as the number of constraints"

        N = E.shape[0]

        A, S = _canon_symmetric(Q)
        negdef = -1 if np.all(np.diag(S) < 0) else 1
        EAinv = E @ np.linalg.inv(A)
        Coup = np.diag(
            1 - 2 * np.all(np.isclose(EAinv[:, np.diag(S) == 1], 0, atol=1e-7), axis=1)
        )

        D = spike_scale * -negdef * np.linalg.inv(Q) @ E.T @ Coup

        if Fseed is not None:
            np.random.seed(Fseed)
        F = np.random.choice([-1, 1], (N, di))

        if type(T) in {int, float}:
            T = np.full(N, T)

        assert type(T) is np.ndarray
        return cls(F, E, D, T, lamb)

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        x: np.ndarray | None = None,
        I: np.ndarray | None = None,
        y: np.ndarray | None = None,
        y_op: np.ndarray | None = None,
        y_op_lim: np.ndarray | None = None,
        latent_bias: bool = False,
        centered: np.ndarray | None = None,
        save: bool = True,
    ) -> tuple[
        matplotlib.figure.Figure | matplotlib.figure.SubFigure,
        matplotlib.axes.Axes,
        list,
    ]:
        """
        Plot the network: bounding boundary (and trajectories)

        If x and y are passed, this is also plotted as trajectories in the bounding box.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes to plot to. If None, a new figure is created.

        x : ndarray of shape (di, time_steps), default=None
            Input trajectory to plot.

        I : ndarray of shape (N, time_steps), default=None
            Input current to the neurons.

        y : ndarray of shape (do, time_steps), default=None
            Output trajectory to plot.

        y_op : ndarray of shape (do, time_steps), default=None
            Solution to the optimization problem with x(t) as input.

        y_op_lim : ndarray of shape (do, time_steps), default=None
            Solution to the optimization problem with x(t) as input, in the limit of small spikes.

        latent_bias : bool, default=False
            If True, the bounding box is centered considering the input current I.

        centered : np.ndarray | None, default=None
            Center of the geometry for the plot. If None, it is estimated automatically.

        save : bool, default=True
            If True, the figure is saved.

        Returns
        -------
        fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
            Figure of the plot.

        ax : matplotlib.axes.Axes
            Axes of the plot.

        artists : list
            List of artists in the plot.
        """

        assert centered is None or centered.shape == (
            self.do,
        ), "centered should be of shape (do,) or None"

        if ax is None:
            if self.do == 2:
                ax = plt.figure(figsize=(10, 10)).gca()
            elif self.do == 3:
                plt.figure(figsize=(10, 10))
                ax = plt.subplot(111, projection="3d")
            else:
                raise NotImplementedError("Only 2D and 3D network vis. is possible")

        if x is None:
            x = np.zeros((self.di, 1))
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y is not None and y.ndim == 1:
            y = y[:, np.newaxis]

        if I is None:
            I = np.zeros((self.N, 1))
        if I.ndim == 1:
            I = I[:, np.newaxis]

        x0 = x[:, -1]
        I0 = I[:, -1]

        if latent_bias:
            Ty = self.T - self.F @ x0
        else:
            Ty = self.T - self.F @ x0 - I0

        # Inhibitory standard
        if centered is None:
            negT = Ty.copy()
            negT[negT > 0] = 0
            centered = np.linalg.lstsq(self.E, negT, rcond=None)[0]

        artists = []

        # plot the network
        if self.do in {2, 3}:
            artists = (
                self._draw_bbox_2D(centered, Ty, ax)
                if self.do == 2
                else self._draw_bbox_3D(centered, Ty, ax)
            )
            # Y Trajectory
            if y is not None:
                artists_y = plot._plot_traj(ax, y, gradient=True)
                artists.append(artists_y)
                if not latent_bias:
                    artists_leak = plot._plot_small_vector(ax, y[:, -1], -y[:, -1])
                else:
                    bias = np.linalg.lstsq(self.E, I0, rcond=None)[0]
                    artists_leak = plot._plot_small_vector(
                        ax, y[:, -1], -y[:, -1] + bias
                    )
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
            raise NotImplementedError("Only 2D and 3D Latents vis. is possible")

        fig = ax.get_figure()
        assert fig is not None
        if save:
            assert type(fig) is matplotlib.figure.Figure
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            _save_fig(fig, time_stamp + "-bounding-box.png")

        return fig, ax, artists

    def plot_rate_space(
        self,
        x: np.ndarray | None = None,
        I: np.ndarray | None = None,
        ax: matplotlib.axes.Axes | None = None,
        r: np.ndarray | None = None,
        r_op: np.ndarray | None = None,
        r_op_lim: np.ndarray | None = None,
        save: bool = True,
    ) -> tuple[
        matplotlib.figure.Figure | matplotlib.figure.SubFigure,
        matplotlib.axes.Axes,
        list,
    ]:
        """
        Plot the network in rate space: boundaries (and trajectories). Only for N = 2 or 3 neurons.

        If r is passed, this is also plotted as a trajectory.

        Parameters
        ----------

        x : ndarray of shape (di, time_steps), default=None
            Input to the network.

        I : ndarray of shape (N, time_steps), default=None
            Input current to the neurons.

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
        fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
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

        if I is None:
            I = np.zeros((self.N, 1))
        if I.ndim == 1:
            I = I[:, np.newaxis]

        x0 = x[:, -1]
        I0 = I[:, -1]

        artists = []

        # plot the network
        if self.N in {2, 3}:
            artists = (
                self._draw_rate_space_2D(x0, I0, ax)
                if self.N == 2
                else self._draw_rate_space_3D(x0, I0, ax)
            )
            # r Trajectory
            if r is not None:
                artists_r = plot._plot_traj(ax, r, gradient=True)
                artists.append(artists_r)
                artists_leak = plot._plot_small_vector(ax, r[:, -1], -r[:, -1])
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
            raise NotImplementedError(
                "Only N=2 or 3 neurons rate vis. plot is possible"
            )

        fig = ax.get_figure()
        assert fig is not None
        if save:
            assert type(fig) is matplotlib.figure.Figure
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            _save_fig(fig, time_stamp + "-rate-space.png")

        return fig, ax, artists

    def plot_vol_space(
        self,
        x: np.ndarray | None = None,
        I: np.ndarray | None = None,
        ax: matplotlib.axes.Axes | None = None,
        V: np.ndarray | None = None,
        voltage_bias: str = "",
        save: bool = True,
    ) -> tuple[
        matplotlib.figure.Figure | matplotlib.figure.SubFigure,
        matplotlib.axes.Axes,
        list,
    ]:
        """
        Plot the network in voltage space: boundaries (and trajectories). Only for N = 2 or 3 neurons.

        If V is passed, this is also plotted as a trajectory.

        Parameters
        ----------

        x : ndarray of shape (di, time_steps), default=None
            Input to the network.

        I : ndarray of shape (N, time_steps), default=None
            Input current to the neurons.

        ax : matplotlib.axes.Axes, default=None
            Axes to plot to. If None, a new figure is created.

        V : ndarray of shape (N, time_steps), default=None
            Voltages trajectory to plot.

        voltage_bias : str, default=""
            If "", the thresholds change with Fx + I.
            If "F", the thresholds change with I only.
            If "I", the thresholds change with Fx only.
            If "FI", the thresholds do not change with Fx + I but the voltage is biased.

        save : bool, default=True
            If True, the figure is saved.

        Returns
        -------
        fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
            Figure of the plot.

        ax : matplotlib.axes.Axes
            Axes of the plot.

        artists : list
            List of artists in the plot.
        """

        assert voltage_bias in {
            "",
            "F",
            "I",
            "FI",
        }, 'voltage_bias should be in {"", "F", "I", "FI"}'

        if ax is None:
            ax = plt.figure(figsize=(10, 10)).gca()

        if x is None:
            x = np.zeros((self.di, 1))
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if V is not None and V.ndim == 1:
            V = V[:, np.newaxis]

        if I is None:
            I = np.zeros((self.N, 1))
        if I.ndim == 1:
            I = I[:, np.newaxis]

        x0 = x[:, -1]
        I0 = I[:, -1]

        artists = []

        # plot the network
        if self.N in {2, 3}:
            match voltage_bias:
                case "":
                    Tv = self.T - self.F @ x0 - I0
                case "F":
                    Tv = self.T - I0
                case "I":
                    Tv = self.T - self.F @ x0
                case "FI":
                    Tv = self.T
                case _:
                    raise ValueError('voltage_bias should be in {"", "F", "I", "FI"}')

            artists = (
                self._draw_vol_space_2D(Tv, ax)
                if self.N == 2
                else self._draw_vol_space_3D(Tv, ax)
            )
            # V Trajectory
            if V is not None:
                artists_V = plot._plot_traj(ax, V, gradient=True)
                artists.append(artists_V)
                artists_leak = plot._plot_small_vector(ax, V[:, -1], -V[:, -1])
                artists.append(artists_leak)
        else:
            raise NotImplementedError(
                "Only N=2 or 3 neurons voltage vis. plot is possible"
            )

        fig = ax.get_figure()
        assert fig is not None
        if save:
            assert type(fig) is matplotlib.figure.Figure
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            _save_fig(fig, time_stamp + "-voltage-space.png")

        return fig, ax, artists

    def _animate(
        self,
        ax: matplotlib.axes.Axes,
        artists: list,
        x: np.ndarray,
        I: np.ndarray,
        y: np.ndarray,
        y_op: np.ndarray | None = None,
        y_op_lim: np.ndarray | None = None,
        latent_bias: bool = False,
        input_change: bool = False,
        current_change: bool = False,
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

        I : ndarray of shape (N, time_steps)
            Input current to the neurons.

        y : ndarray of shape (do, time_steps)
            Output trajectory to plot.

        y_op : ndarray of shape (do, time_steps), default=None
            Solution to the optimization problem with x(t) as input.

        y_op_lim : ndarray of shape (do, time_steps), default=None
            Solution to the optimization problem with x(t) as input, in the limit of small spikes.

        latent_bias : bool, default=False
            If True, the bounding box is centered considering the input current I.

        input_change: bool, default=False
            If True, the input has changed.

        current_change: bool, default=False
            If True, the input current has changed.

        spiking : ndarray(int), default=None
            Neurons spiking in this frame. Index starting at 1. -n if the neuron needs to be restored.
        """

        offset = 0
        if y_op is not None:
            offset += 1
        if y_op_lim is not None:
            offset += 1

        plot._animate_traj(ax, artists[-2 - offset], y)
        if not latent_bias:
            plot._animate_small_vector(artists[-1 - offset], y[:, -1], -y[:, -1])
        else:
            I0 = I[:, -1]
            bias = np.linalg.lstsq(self.E, I0, rcond=None)[0]
            plot._animate_small_vector(artists[-1 - offset], y[:, -1], -y[:, -1] + bias)

        if input_change or current_change:
            x0 = x[:, -1]
            I0 = I[:, -1]

            if latent_bias:
                Ty = self.T - self.F @ x0
            else:
                Ty = self.T - self.F @ x0 - I0

            negT = Ty.copy()
            negT[negT > 0] = 0
            centered = np.linalg.lstsq(self.E, negT, rcond=None)[0]

            if self.do == 2:
                self._draw_bbox_2D(centered, Ty, ax, artists)
            else:
                self._draw_bbox_3D(centered, Ty, ax, artists)

            if y_op is not None:
                plot._animate_scatter(artists[-offset], y_op[:, -1:])
            if y_op_lim is not None:
                plot._animate_scatter(artists[-1], y_op_lim[:, -1:])

        if spiking is not None and len(spiking) > 0:
            plot._animate_spiking(artists, spiking)

    def _animate_rate_space(
        self,
        ax: matplotlib.axes.Axes,
        artists: list,
        x: np.ndarray,
        I: np.ndarray,
        r: np.ndarray,
        r_op: np.ndarray | None = None,
        r_op_lim: np.ndarray | None = None,
        input_change: bool = False,
        current_change: bool = False,
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

        I : ndarray of shape (N, time_steps)
            Input current to the neurons.

        r : ndarray of shape (N, time_steps)
            Rate trajectory to plot.

        r_op : ndarray of shape (N, time_steps)
            Solution to the optimization problem with x(t) as input.

        r_op_lim : ndarray of shape (N, time_steps)
            Solution to the optimization problem with x(t) as input, in the limit of small spikes.

        input_change: bool, default=False
            If True, the input has changed.

        current_change: bool, default=False
            If True, the input current has changed.

        spiking : ndarray(int), default=None
            Neurons spiking in this frame. Index starting at 1. -n if the neuron needs to be restored.
        """

        offset = 0
        if r_op is not None:
            offset += 1
        if r_op_lim is not None:
            offset += 1

        plot._animate_traj(ax, artists[-2 - offset], r)
        plot._animate_small_vector(artists[-1 - offset], r[:, -1], -r[:, -1])

        if input_change or current_change:
            x0 = x[:, -1]
            I0 = I[:, -1]
            (
                self._draw_rate_space_2D(x0, I0, ax, artists)
                if self.N == 2
                else self._draw_rate_space_3D(x0, I0, ax, artists)
            )
            if r_op is not None:
                plot._animate_scatter(artists[-offset], r_op[:, -1:])
            if r_op_lim is not None:
                plot._animate_scatter(artists[-1], r_op_lim[:, -1:])

        if spiking is not None and len(spiking) > 0:
            plot._animate_spiking(artists, spiking)

    def _animate_vol_space(
        self,
        ax: matplotlib.axes.Axes,
        artists: list,
        x: np.ndarray,
        I: np.ndarray,
        V: np.ndarray,
        input_change: bool = False,
        current_change: bool = False,
        spiking: np.ndarray | None = None,
        voltage_bias: str = "",
    ) -> None:
        """
        Animate the voltage space by modifying the artists.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot to.

        artists : list
            List of artists to modify.

        x : ndarray of shape (di, time_steps)
            Input trajectory to plot.

        I : ndarray of shape (N, time_steps)
            Input current to the neurons.

        V : ndarray of shape (N, time_steps)
            Voltage trajectory to plot.

        input_change: bool, default=False
            If True, the input has changed.

        current_change: bool, default=False
            If True, the input current has changed.

        spiking : ndarray(int), default=None
            Neurons spiking in this frame. Index starting at 1. -n if the neuron needs to be restored.

        voltage_bias : str, default=""
            If "", the thresholds change with Fx + I.
            If "F", the thresholds change with I only.
            If "I", the thresholds change with Fx only.
            If "FI", the thresholds do not change with Fx + I but the voltage is biased.
        """

        offset = 0

        plot._animate_traj(ax, artists[-2 - offset], V)
        plot._animate_small_vector(artists[-1 - offset], V[:, -1], -V[:, -1])

        if (input_change or current_change) and voltage_bias != "FI":
            x0 = x[:, -1]
            I0 = I[:, -1]
            match voltage_bias:
                case "":
                    Tv = self.T - self.F @ x0 - I0
                case "F":
                    Tv = self.T - I0
                case "I":
                    Tv = self.T - self.F @ x0
                case _:
                    raise ValueError('voltage_biased should be in {"", "F", "I", "FI"}')

            (
                self._draw_vol_space_2D(Tv, ax, artists)
                if self.N == 2
                else self._draw_vol_space_3D(Tv, ax, artists)
            )

        if spiking is not None and len(spiking) > 0:
            plot._animate_spiking(artists, spiking)

    def _draw_bbox_2D(
        self,
        centered: np.ndarray,
        Ty: np.ndarray,
        ax: matplotlib.axes.Axes,
        artists: list | None = None,
    ) -> list:
        """
        Draw the bounding box visualization of the network.

        Parameters
        ----------
        centered : ndarray of shape (2,)
            Center of the bounding box.

        Ty : ndarray of shape (N,)
            Effective thresholds of the network.

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
            c = -Ty[n]
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
            q = _line_closest_point(centered[0], centered[1], a, b, c)
            if first_frame:
                quiver = _plot_big_vector(ax, q, self.D[:, n], color=colors[n])
                artists.append([poly, line, quiver])
                # quiver = ax.quiver(
                #     q[0],
                #     q[1],
                #     self.D[0, n],
                #     self.D[1, n],
                #     color=colors[n],
                #     scale=5,
                #     scale_units="xy",
                #     angles="xy",
                #     zorder=n,
                # )
            else:
                _animate_big_vector(artists[n][2], q, self.D[:, n])
                # artists[n][2].set_offsets([q0, q1])
                # artists[n][2].set_UVC(self.D[0, n], self.D[1, n])
                # artists[n][2].set_zorder(n)

        ax.set_xlim(centered[0] - 1, centered[0] + 1)
        ax.set_ylim(centered[1] - 1, centered[1] + 1)
        ax.set_ylabel("y2")
        ax.set_xlabel("y1")
        ax.set_aspect("equal")

        return artists

    def _draw_bbox_3D(
        self,
        centered: np.ndarray,
        Ty: np.ndarray,
        ax: matplotlib.axes.Axes,
        artists: list | None = None,
    ) -> list:
        """
        Draw the bounding box visualization of the network for 3D cases.

        Parameters
        ----------
        centered : ndarray of shape (2,)
            Center of the bounding box.

        Ty : ndarray of shape (N,)
            Effective thresholds of the network.

        ax : matplotlib.axes.Axes
            Axes to plot the network.

        artists : list, default = None
            List of artists to update the plot. If None, new artists are created.

        Returns
        -------
        artists : list
            List of artists to update the plot.

        """

        assert isinstance(ax, Axes3D)

        first_frame = artists is None

        if first_frame:
            artists = []

        colors = _get_colors(self.N, self.W)

        def plane_func(
            y1: np.ndarray, y2: np.ndarray, a: float, b: float, c: float, d: float
        ) -> np.ndarray:
            return (-a * y1 - b * y2 - d) / c

        # TODO: Revisit where to center the plot
        y1x = np.linspace(centered[0] - 1, centered[0] + 1, 100)
        y2x = np.linspace(centered[1] - 1, centered[1] + 1, 100)
        y3x = np.linspace(centered[2] - 1, centered[2] + 1, 100)

        points = np.zeros((100, 100, self.N, 3))
        a = np.zeros(self.N)
        b = np.zeros(self.N)
        c = np.zeros(self.N)
        d = np.zeros(self.N)
        ver = np.zeros(self.N)
        for n in range(self.N):
            # TODO: This could be all that changes (a,b,c) so maybe this is where you need to separate
            a[n] = self.E[n, 0]
            b[n] = self.E[n, 1]
            c[n] = self.E[n, 2]
            d[n] = -Ty[n]
            ver[n] = (
                0
                if np.max(np.abs([b[n], c[n]])) < np.abs(a[n])
                else (1 if np.max(np.abs([a[n], c[n]])) < np.abs(b[n]) else 2)
            )

            match ver[n]:
                case 0:
                    Y, Z = np.meshgrid(y2x, y3x)
                    X = plane_func(Y, Z, b[n], c[n], a[n], d[n])
                case 1:
                    X, Z = np.meshgrid(y1x, y3x)
                    Y = plane_func(X, Z, a[n], c[n], b[n], d[n])
                case 2:
                    X, Y = np.meshgrid(y1x, y2x)
                    Z = plane_func(X, Y, a[n], b[n], c[n], d[n])
                case _:
                    raise ValueError("Invalid case")

            points[:, :, n, 0] = X
            points[:, :, n, 1] = Y
            points[:, :, n, 2] = Z

        for n in range(self.N):
            suprathresh = np.any(
                +self.E[:, 0][:, np.newaxis, np.newaxis] * points[:, :, n, 0]
                + self.E[:, 1][:, np.newaxis, np.newaxis] * points[:, :, n, 1]
                + self.E[:, 2][:, np.newaxis, np.newaxis] * points[:, :, n, 2]
                > Ty[:, np.newaxis, np.newaxis] + 1e-10,
                axis=0,
            )
            points[suprathresh, n, :] = np.nan

            # polygon (to optimize: no redraw)
            if not first_frame:
                artists[n][0].remove()
            poly = ax.plot_surface(
                points[:, :, n, 0],
                points[:, :, n, 1],
                points[:, :, n, 2],
                color=colors[n],
                alpha=0.2,
                zorder=n,
            )
            if not first_frame:
                artists[n][0] = poly

            # quiver
            quiver = None
            q = _plane_closest_point(
                centered[0], centered[1], centered[2], a[n], b[n], c[n], d[n]
            )
            if first_frame:
                quiver = _plot_big_vector(ax, q, self.D[:, n], color=colors[n])
                artists.append([poly, quiver])
            else:
                _animate_big_vector(artists[n][1], q)

        ax.set_xlim(centered[0] - 1, centered[0] + 1)
        ax.set_ylim(centered[1] - 1, centered[1] + 1)
        ax.set_zlim(centered[2] - 1, centered[2] + 1)
        ax.set_zlabel("y3")
        ax.set_ylabel("y2")
        ax.set_xlabel("y1")
        ax.set_aspect("equal")

        return artists

    def _draw_rate_space_2D(
        self,
        x0: np.ndarray,
        I0: np.ndarray,
        ax: matplotlib.axes.Axes,
        artists: list | None = None,
    ) -> list:
        """
        Draw the rate space visualization of the network. For N = 2 neurons.

        Parameters
        ----------

        x0 : ndarray of shape (di,)
            Input of the network.

        I0 : ndarray of shape (N,)
            Input currents of the neurons.

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
            c[n] = -self.T[n] + self.F[n, :] @ x0 + I0[n]
            compare = []
            if np.abs(a[n]) > 1e-3:
                compare.append(-c[n] / a[n])
            if np.abs(b[n]) > 1e-3:
                compare.append(-c[n] / b[n])
            maxinter = np.max(compare + [maxinter])

        y1x = np.linspace(0, maxinter + 1, 100)
        y2x = np.linspace(0, maxinter + 1, 100)
        for n in range(self.N):
            diag = np.abs(a[n]) < np.abs(b[n])
            yo_p = (
                line_func(y1x, a[n], b[n], c[n])
                if diag
                else line_func(y2x, b[n], a[n], c[n])
            )
            y1 = y1x if diag else yo_p
            y2 = yo_p if diag else y2x
            y1[y1 < 0] = 0
            y2[y2 < 0] = 0

            # polygon (to optimize: no redraw)
            if not first_frame:
                artists[n][0].remove()

            if diag:
                px = y1
                py = y2
            else:
                px = y2
                py = y1

            if diag != (n == 0):
                mx = self.W[n, n] > 0
            else:
                slope = np.sign(a[n] * -b[n])
                mx = slope * self.W[n, n] < 0
            if diag:
                if not mx:
                    poly = ax.fill_between(
                        px,
                        0,
                        py,
                        color=colors[n],
                        interpolate=True,
                        alpha=0.2,
                        zorder=n,
                    )
                else:
                    poly = ax.fill_between(
                        px,
                        py,
                        maxinter + 1,
                        color=colors[n],
                        interpolate=True,
                        alpha=0.2,
                        zorder=n,
                    )
            else:
                if not mx:
                    poly = ax.fill_betweenx(
                        px,
                        0,
                        py,
                        color=colors[n],
                        interpolate=True,
                        alpha=0.2,
                        zorder=n,
                    )
                else:
                    poly = ax.fill_betweenx(
                        px,
                        py,
                        maxinter + 1,
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
            q = _line_closest_point(0, 0, a[n], b[n], c[n])
            if first_frame:
                quiver = _plot_big_vector(
                    ax,
                    q,
                    np.array([not n, n]),
                    color=colors[n],
                    on=bool(np.all(q >= 0)),
                )
                artists.append([poly, line, quiver])
            else:
                _animate_big_vector(
                    artists[n][2], q, np.array([not n, n]), on=bool(np.all(q >= 0))
                )

        ax.hlines(0, 0, maxinter + 1, color="k")
        ax.vlines(0, 0, maxinter + 1, color="k")
        ax.set_xlim(-0.02 * maxinter, maxinter + 1)
        ax.set_ylim(-0.02 * maxinter, maxinter + 1)
        ax.set_ylabel("r2")
        ax.set_xlabel("r1")
        ax.set_aspect("equal")

        return artists

    def _draw_rate_space_3D(
        self,
        x0: np.ndarray,
        I0: np.ndarray,
        ax: matplotlib.axes.Axes,
        artists: list | None = None,
    ) -> list:
        """
        Draw the rate space visualization of the network. For N = 3 neurons.

        Parameters
        ----------

        x0 : ndarray of shape (di,)
            Input of the network.

        I0 : ndarray of shape (N,)
            Input currents of the neurons.

        ax : matplotlib.axes.Axes
            Axes to plot the network.

        artists : list, default = None
            List of artists to update the plot. If None, new artists are created.

        Returns
        -------
        artists : list
            List of artists to update the plot.

        """

        assert isinstance(ax, Axes3D)

        first_frame = artists is None
        if first_frame:
            artists = []

        colors = _get_colors(self.N, self.W)

        def plane_func(
            y1: np.ndarray, y2: np.ndarray, a: float, b: float, c: float, d: float
        ) -> np.ndarray:
            return (-a * y1 - b * y2 - d) / c

        maxinter = 1
        a = np.zeros(self.N)
        b = np.zeros(self.N)
        c = np.zeros(self.N)
        d = np.zeros(self.N)
        for n in range(self.N):
            a[n] = self.W[n, 0]
            b[n] = self.W[n, 1]
            c[n] = self.W[n, 2]
            d[n] = -self.T[n] + self.F[n, :] @ x0 + I0[n]
            compare = []
            if np.abs(a[n]) > 1e-3:
                compare.append(-d[n] / a[n])
            if np.abs(b[n]) > 1e-3:
                compare.append(-d[n] / b[n])
            if np.abs(c[n]) > 1e-3:
                compare.append(-d[n] / c[n])
            maxinter = np.max(compare + [maxinter])

        y1x = np.linspace(0, maxinter + 1, 100)
        y2x = np.linspace(0, maxinter + 1, 100)
        y3x = np.linspace(0, maxinter + 1, 100)
        points = np.zeros((100, 100, self.N, 3))
        ver = np.zeros(self.N)
        for n in range(self.N):
            # TODO: This could be all that changes (a,b,c) so maybe this is where you need to separate
            ver[n] = (
                0
                if np.max(np.abs([b[n], c[n]])) < np.abs(a[n])
                else (1 if np.max(np.abs([a[n], c[n]])) < np.abs(b[n]) else 2)
            )

            match ver[n]:
                case 0:
                    Y, Z = np.meshgrid(y2x, y3x)
                    X = plane_func(Y, Z, b[n], c[n], a[n], d[n])
                case 1:
                    X, Z = np.meshgrid(y1x, y3x)
                    Y = plane_func(X, Z, a[n], c[n], b[n], d[n])
                case 2:
                    X, Y = np.meshgrid(y1x, y2x)
                    Z = plane_func(X, Y, a[n], b[n], c[n], d[n])
                case _:
                    raise ValueError("Invalid case")

            X[X < 0] = np.nan
            Y[Y < 0] = np.nan
            Z[Z < 0] = np.nan
            points[:, :, n, 0] = X
            points[:, :, n, 1] = Y
            points[:, :, n, 2] = Z

        for n in range(self.N):
            suprathresh = np.any(
                (self.F @ x0)[:, np.newaxis, np.newaxis]
                + self.W[:, 0][:, np.newaxis, np.newaxis] * points[:, :, n, 0]
                + self.W[:, 1][:, np.newaxis, np.newaxis] * points[:, :, n, 1]
                + self.W[:, 2][:, np.newaxis, np.newaxis] * points[:, :, n, 2]
                + I0[:, np.newaxis, np.newaxis]
                > self.T[:, np.newaxis, np.newaxis] + 1e-10,
                axis=0,
            )
            points[suprathresh, n, :] = np.nan

            # polygon (to optimize: no redraw)
            if not first_frame:
                artists[n][0].remove()
            poly = ax.plot_surface(
                points[:, :, n, 0],
                points[:, :, n, 1],
                points[:, :, n, 2],
                color=colors[n],
                alpha=0.2,
                zorder=n,
            )
            if not first_frame:
                artists[n][0] = poly

            # quiver
            quiver = None
            q = _plane_closest_point(0, 0, 0, a[n], b[n], c[n], d[n])
            vector = np.zeros(3)
            vector[n] = 1
            if first_frame:
                quiver = _plot_big_vector(
                    ax, q, vector, color=colors[n], on=bool(np.all(q >= 0))
                )
                artists.append([poly, quiver])
            else:
                _animate_big_vector(artists[n][1], q, vector, on=bool(np.all(q >= 0)))

        ax.set_xlim(0, maxinter + 1)
        ax.set_ylim(maxinter + 1, 0)
        ax.set_zlim(0, maxinter + 1)
        ax.set_zlabel("r3")
        ax.set_ylabel("r2")
        ax.set_xlabel("r1")
        ax.set_aspect("equal")

        return artists

    def _draw_vol_space_2D(
        self,
        Tv: np.ndarray,
        ax: matplotlib.axes.Axes,
        artists: list | None = None,
    ) -> list:
        """
        Draw the voltage space visualization of the network. For N = 2 neurons.

        Parameters
        ----------

        Tv : ndarray of shape (2,)
            Effective thresholds of the neurons after biasing.

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

        corner = Tv
        rext = corner[0] + 0.25
        lext = corner[0] - 1.75
        uext = corner[1] + 0.25
        dext = corner[1] - 1.75
        for n in range(self.N):

            if n == 0:
                y2x = np.linspace(dext, uext, 100)
                y1x = np.ones(100) * corner[0]
            else:
                y1x = np.linspace(lext, rext, 100)
                y2x = np.ones(100) * corner[1]

            # polygon (to optimize: no redraw)
            if not first_frame:
                artists[n][0].remove()
            if n == 0:
                poly = ax.fill_betweenx(
                    y2x,
                    corner[0],
                    rext,
                    color=colors[n],
                    interpolate=True,
                    alpha=0.2,
                    zorder=n,
                )
            else:
                poly = ax.fill_between(
                    y1x,
                    corner[1],
                    uext,
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
                line = ax.plot(y1x, y2x, linewidth=3, c=colors[n], zorder=n)[0]
            else:
                artists[n][1].set_xdata(y1x)
                artists[n][1].set_ydata(y2x)

            # quiver
            quiver = None
            q = np.zeros(2)
            q[0] = corner[0] if n == 0 else lext + 1
            q[1] = corner[1] if n == 1 else dext + 1
            if first_frame:
                quiver = _plot_big_vector(ax, q, self.W[:, n], color=colors[n])
                artists.append([poly, line, quiver])
            else:
                _animate_big_vector(artists[n][2], q, self.W[:, n])

        ax.hlines(0, lext, rext, color="k")
        ax.vlines(0, dext, uext, color="k")
        ax.set_xlim(lext, rext)
        ax.set_ylim(dext, uext)
        ax.set_ylabel("V2")
        ax.set_xlabel("V1")
        ax.set_aspect("equal")

        return artists

    def _draw_vol_space_3D(
        self,
        Tv: np.ndarray,
        ax: matplotlib.axes.Axes,
        artists: list | None = None,
    ) -> list:
        """
        Draw the voltage space visualization of the network. For N = 3 neurons.

        Parameters
        ----------

        Tv : ndarray of shape (3,)
            Effective thresholds of the neurons after biasing.

        ax : matplotlib.axes.Axes
            Axes to plot the network.

        artists : list, default = None
            List of artists to update the plot. If None, new artists are created.

        Returns
        -------
        artists : list
            List of artists to update the plot.

        """

        assert isinstance(ax, Axes3D)

        first_frame = artists is None
        if first_frame:
            artists = []

        colors = _get_colors(self.N, self.W)

        corner = Tv
        rext = corner[0]
        lext = corner[0] - 1.75
        uext = corner[1]
        dext = corner[1] - 1.75
        iext = corner[2]
        oext = corner[2] - 1.75
        y1x = np.linspace(lext, rext, 100)
        y2x = np.linspace(dext, uext, 100)
        y3x = np.linspace(oext, iext, 100)

        Edir = np.cross(self.E[:, 0], self.E[:, 1])
        X, Y = np.meshgrid(y1x, y2x)
        Z = Edir[0] * X + Edir[1] * Y / Edir[2]
        ax.plot_surface(
            X,
            Y,
            Z,
            color="k",
            alpha=0.05,
            zorder=-1,
        )

        for n in range(self.N):
            q = np.zeros(3)
            match n:
                case 0:
                    Y, Z = np.meshgrid(y2x, y3x)
                    X = corner[0] * np.ones_like(Y)
                    q[0] = corner[0]
                    q[1] = dext + 1
                    q[2] = oext + 1
                case 1:
                    X, Z = np.meshgrid(y1x, y3x)
                    Y = corner[1] * np.ones_like(X)
                    q[0] = lext + 1
                    q[1] = corner[1]
                    q[2] = oext + 1
                case 2:
                    X, Y = np.meshgrid(y1x, y2x)
                    Z = corner[2] * np.ones_like(X)
                    q[0] = lext + 1
                    q[1] = dext + 1
                    q[2] = corner[2]
                case _:
                    raise ValueError("Invalid case")

            # polygon (to optimize: no redraw)
            if not first_frame:
                artists[n][0].remove()
            poly = ax.plot_surface(
                X,
                Y,
                Z,
                color=colors[n],
                alpha=0.2,
                zorder=n,
            )
            if not first_frame:
                artists[n][0] = poly

            # quiver
            quiver = None
            if first_frame:
                quiver = _plot_big_vector(ax, q, self.W[:, n], color=colors[n])
                artists.append([poly, quiver])
            else:
                _animate_big_vector(artists[n][1], q, self.W[:, n])

        ax.set_xlim(lext, rext)
        ax.set_ylim(dext, uext)
        ax.set_zlim(oext, iext)
        ax.set_zlabel("V3")
        ax.set_ylabel("V2")
        ax.set_xlabel("V1")
        ax.set_aspect("equal")

        return artists
