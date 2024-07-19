import time
from typing import Self

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from SCN import plot

from . import boundary
from .low_rank_LIF import Low_rank_LIF
from .utils_plots import _save_fig


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
    >>> net = Autoencoder.init_random(d=2, N=6)
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
        T: int | float | np.ndarray = 0.5,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
    ) -> None:
        r"""
        Constructor with specific parameters.

        Parameters
        ----------
        D : ndarray of shape (do, N)
            Weights of the network. The columns need to be normalized.

        T : int, float or ndarray of shape (N,), default = 0.5
            Threshold of the neurons.

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

        if isinstance(T, (int, float)):
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
        T: int | float | np.ndarray = 0.5,
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

        T : int, float or ndarray of shape (N,), default = 0.5
            Threshold of the neurons.

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
        D = -boundary._sphere_random(d=d, N=N, seed=seed).T

        return cls(D, T, lamb, spike_scale)

    @classmethod
    def init_cube(
        cls,
        d: int = 2,
        one_quadrant: bool = False,
        T: int | float | np.ndarray = 0.5,
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

        T : int, float or ndarray of shape (N,), default = 0.5
            Threshold of the neurons.

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
        D = -boundary._cube(d=d, one_quadrant=one_quadrant).T

        return cls(D, T, lamb, spike_scale)

    @classmethod
    def init_2D_spaced(
        cls,
        N: int = 10,
        angle_range: list | None = None,
        T: int | float | np.ndarray = 0.5,
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

        T : int, float or ndarray of shape (N,), default = 0.5
            Threshold of the neurons.

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
        D = -boundary._2D_circle_spaced(N=N, angle_range=angle_range).T

        return cls(D, T, lamb, spike_scale)

    @classmethod
    def init_2D_random(
        cls,
        N: int = 10,
        angle_range: list | None = None,
        seed: int | None = None,
        T: int | float | np.ndarray = 0.5,
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

        T : int, float or ndarray of shape (N,), default = 0.5
            Threshold of the neurons.

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
        D = -boundary._2D_circle_random(N=N, angle_range=angle_range, seed=seed).T

        return cls(D, T, lamb, spike_scale)

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
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

        # Bounding box
        centered = x[:, -1]
        x0 = centered

        artists = []

        # plot the network
        if self.di == 2:
            artists = self._draw_bbox_2D(centered, x0, ax)

            # X Trajectory
            artists_x = plot._plot_traj(ax, x, gradient=False)
            artists.append(artists_x)
            # Y Trajectory
            if y is not None:
                artists_y = plot._plot_traj(ax, y, gradient=True)
                artists.append(artists_y)
                artists_leak = plot._plot_vector(ax, y[:, -1], -y[:, -1])
                artists.append(artists_leak)
        else:
            raise NotImplementedError("Only 2D Autoencoder vis. is implemented for now")

        fig = ax.get_figure()
        assert fig is not None
        if save:
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            _save_fig(fig, time_stamp + "-autoencoder.png")

        return fig, ax, artists

    def _animate(
        self,
        ax: matplotlib.axes.Axes,
        artists: list,
        x: np.ndarray,
        y: np.ndarray,
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

        input_change : bool, default=False
            If True, the input is changing.

        spiking : ndarray(int), default=None
            Neurons spiking in this frame. Index starting at 1. -n if the neuron needs to be restored.
        """

        centered = x[:, 0] - x[:, -1]
        yinv = y + centered[:, np.newaxis]

        plot._animate_traj(ax, artists[-2], yinv)
        plot._animate_vector(artists[-1], yinv[:, -1], yinv[:, -1] - x[:, 0])
        if spiking is not None:
            plot._animate_spiking(artists, spiking)
        if input_change:
            plot._animate_axis(ax, x0=x[:, 0], xf=x[:, -1])

            xinv = x + centered[:, np.newaxis]
            plot._animate_traj(ax, artists[-3], traj=xinv, gradient=False)
