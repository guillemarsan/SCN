import time
from typing import Self

import cvxpy as cp
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from SCN import plot

from . import boundary
from .low_rank_LIF import Low_rank_LIF
from .utils_plots import _save_fig


class EI_Network(Low_rank_LIF):
    r"""
    Double population (E and I) model. Subclass of the Low-rank LIF model with
    :math:`\mathbf{W}_{j} \leq 0` (neuron `j` is inhibitory) or :math:`\mathbf{W}_{j} \geq 0` (neuron `j` is excitatory).

    :math:`N` neurons, :math:`d_i` input dimensions and :math:`d_o` output dimensions.

    See Also
    --------
    :class:`~SCN.low_rank_LIF.Low_rank_LIF` : Parent model Low_rank_LIF.

    References
    ----------
    Podlaski, William & Machens, Christian. (2024). Approximating Nonlinear Functions
    With Latent Boundaries in Low-Rank Excitatory-Inhibitory Spiking Networks.
    Neural Computation. 36. 803-857. 10.1162/neco_a_01658.

    Examples
    --------
    >>> from SCN import Single_Population
    >>> from SCN import Simulation
    >>> net.plot()
    ...
    >>> sim = Simulation()
    >>> x = np.tile([[0.5], [1]], (1, 10000))
    >>> sim.run(net, x)
    >>> sim.animate()
    """

    D: np.ndarray
    r"Weights of the network. :math:`d_o \times N`."

    F: np.ndarray
    r"Forward weights of the network. :math:`N \times d_i`."

    E: np.ndarray
    r"Encoding weights of the network. :math:`N \times d_o`."

    W: np.ndarray
    r"Recurrent weights of the network. :math:`N \times N`. :math:`\mathbf{W}_j \leq 0` (I) or :math:`\mathbf{W}_j \geq 0` (E)."

    lamb: float
    "Leak timescale of the network."

    T: np.ndarray
    r"Thresholds of the neurons. :math:`N \times 1`"

    def __init__(
        self,
        F: np.ndarray,
        E: np.ndarray,
        D: np.ndarray,
        T: int | float | np.ndarray = 0.5,
        lamb: float = 1,
    ) -> None:
        r"""
        Constructor with specific parameters.

        The :math:`N` rows of the matrix :math:`(\mathbf{F} \mathbf{E})` need to be normalized.

        :math:`\mathbf{W}_j = \mathbf{D}^\top_{i} \mathbf{E}^\top_{j=0,...,N}  \leq 0` (I neuron)
        or :math:`\mathbf{W}_j = \mathbf{D}^\top_{i} \mathbf{E}^\top_{j=0,...,N} \geq 0` (E neuron).

        Parameters
        ----------
        F : ndarray of shape (N, di)
            Forward weights of the network.

        E : ndarray of shape (N, do)
            Encoding weights of the network.

        D : ndarray of shape (do, N)
            Decoding weights of the network.

        T : int, float or ndarray of shape (N,), default = 0.5
            Threshold of the neurons.

        lamb : float, default=1
            Leak timescale of the network.

        """

        assert all(np.abs(np.linalg.norm(np.hstack([F, E]), axis=1) - 1) < 1e-10), (
            "The rows of (F|E) need to be normalized, if you want to change the boundary"
            + "use T, if you want to change the spike size use D"
        )

        # assert EI
        W = E @ D
        W[np.argwhere(np.abs(W) < 1e-10)] = 0
        excitatory = np.all(W >= 0, axis=0)
        inhibitory = np.all(W <= 0, axis=0)

        assert np.all(
            np.logical_or(excitatory, inhibitory)
        ), "Each row of W needs to be either excitatory or inhibitory"

        if isinstance(T, (int, float)):
            T = T * np.ones(D.shape[1])

        assert T is not None and not isinstance(T, (int, float))
        super().__init__(F=F, E=E, D=D, T=T, lamb=lamb)

    @classmethod
    def init_random(  # type: ignore[reportIncompatibleMethodOverride]
        cls,
        di: int = 2,
        do: int = 2,
        NE: int = 5,
        NI: int = 5,
        seed: int | None = None,
        T: int | float | np.ndarray = 0.5,
        lamb: float = 1,
    ) -> Self:
        r"""
        Random initialization of the EI Network.
        (see :func:`~SCN.boundary._sphere_random`)

        Characteristics:
        - :math:`(\mathbf{F}|\mathbf{E})` initialized randomly in a hemihemisphere of radius 1 in :math:`\mathbb{R}^{d_i+d_o}`
        - :math:`\mathbf{D}` as close to :math:`\pm E^\top` while respecting EI
        - :math:`\mathbf{T} = \mathbf{T} - \mathbf{E}_N` to center the boundary at :math:`(0, 0, ..., -1)`

        Parameters
        ----------
        di : float, default=2
            Input dimensions.

        do: float, default=2
            Output dimensions.

        NE : float, default=5
            Number of excitatory neurons.

        NI : float, default=5
            Number of inhibitory neurons.

        seed : int or None, default=None
            Seed for the random number generator.

        T : ndarray of shape (N,)
            Threshold of the neurons. If None, :math:`T_i = 1/2`.

        lamb : float, default=1
            Leak timescale of the network.

        Returns
        -------
        net: Autoencoder
            Autoencoder network with random boundary.
        """

        N = NE + NI

        # random boundary
        M = boundary._sphere_random(d=di + do, N=N, seed=seed)

        F = M[:, :di]
        E = M[:, di:]

        # standarize in semi-sphere
        E[:, -2:] = np.abs(E[:, -2:])

        # find D that respects E/I
        EIvec = np.ones(N)
        EIvec[:NI] = -1
        D = cp.Variable((do, N))
        penalty = cp.norm(D - EIvec * E.T)
        prob = cp.Problem(
            cp.Minimize(penalty),
            [D.T[:NI, :] @ E.T <= 0, D.T[NI:, :] @ E.T >= 0],
        )
        prob.solve(cp.SCS, eps_abs=1e-9, eps_rel=0)
        if prob.status != cp.OPTIMAL:
            raise ValueError("Problem not feasible")
        else:
            D = D.value

        # TODO: take this to I
        T = T - E[:, 1]
        return cls(F=F, E=E, D=D, T=T, lamb=lamb)

    @classmethod
    def init_2D_spaced(
        cls,
        di: int = 1,
        NE: int = 5,
        NI: int = 5,
        angle_range: list | None = None,
        T: int | float | np.ndarray = 0.5,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
    ) -> Self:
        r"""
        Regularly spaced 2D initialization of the Single_Population network.

        :math:`N` neurons spaced regularly between `angle_range[0]` and `angle_range[1]`.
        The encoders are :math:`\mathbf{E}_i = (\cos(\alpha_i), \sin(\alpha_i))`.

        Parameters
        ----------

        di: int, default=1
            Input dimensions.

        NE: int, default=5
            Number of excitatory neurons.

        NI: int, default=5
            Number of inhibitory neurons.

        angle_range : list, default=None
            Range of angles for the neurons. If None, the range is :math:`[\pi/4, 3\pi/4]`.

        T : int, float or ndarray of shape (N,), default = 0.5
            Threshold of the neurons.

        lamb : float, default=1
            Leak timescale of the network.

        spike_scale : int, float or ndarray, default=1
            Scale of the spikes.

        Returns
        -------
        net: Single_Population
            Single_Population network with regularly spaced latent boundary.
        """
        N = NE + NI

        if angle_range is None:
            angle_range = [np.pi / 4, 3 * np.pi / 4]

        assert np.abs(angle_range[1] - angle_range[0]) <= np.pi / 2
        # evenly spaced circular parameters
        E = boundary._2D_circle_spaced(N=N, angle_range=angle_range)

        EIvec = np.ones(N)
        EIvec[:NI] = -1
        D = EIvec * E.T
        D = spike_scale * D

        F_scale = 0.1
        E_scale = np.sqrt(1 - F_scale**2)
        F = F_scale * np.random.choice([-1, 1], (N, di))
        E = E_scale * E / np.linalg.norm(E, axis=1)[:, np.newaxis]

        T = T - E[:, 1]
        return cls(F, E, D, T, lamb)

    @classmethod
    def init_2D_random(
        cls,
        di: int = 1,
        NE: int = 5,
        NI: int = 5,
        angle_range: list | None = None,
        seed: int | None = None,
        T: int | float | np.ndarray = 0.5,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
    ) -> Self:
        # TODO consider sending this also to low_rank_LIF
        r"""
        Randomly spaced 2D initialization of the Single_Population network.

        :math:`N` neurons spaced randomly between `angle_range[0]` and `angle_range[1]`.
        The decoders are :math:`\mathbf{E}_i = (\cos(\alpha_i), \sin(\alpha_i))`.

        Parameters
        ----------

        di: int, default=1
            Input dimensions.

        N: int, default=10
            Number of neurons.

        dale : str, default="I"
            Dale's law of the network. "I" for inhibitory, "E" for excitatory.

        angle_range : list, default=None
            Range of angles for the neurons. If None, the range is :math:`[\pi/4, 3\pi/4]`.

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
        net: Single_Population
            Single_Population network with randomly spaced latent boundary.
        """
        N = NE + NI

        if angle_range is None:
            angle_range = [np.pi / 4, 3 * np.pi / 4]

        assert np.abs(angle_range[1] - angle_range[0]) <= np.pi / 2
        # evenly spaced circular parameters
        E = boundary._2D_circle_random(N=N, angle_range=angle_range, seed=seed)

        EIvec = np.ones(N)
        EIvec[:NI] = -1
        D = EIvec * E.T
        D = spike_scale * D

        F_scale = 0.1
        E_scale = np.sqrt(1 - F_scale**2)
        F = F_scale * np.random.choice([-1, 1], (N, di))
        E = E_scale * E / np.linalg.norm(E, axis=1)[:, np.newaxis]

        T = T - E[:, 1]
        return cls(F, E, D, T, lamb)

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        save: bool = True,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list]:
        # TODO: if this is the same as single_pop then maybe move it to low_rank_LIF, and let autoencoder override
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
        else:
            raise NotImplementedError("Only 2D Latents vis. is implemented for now")

        fig = ax.get_figure()
        assert fig is not None
        if save:
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            _save_fig(fig, time_stamp + "-single-population.png")

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

        input_change: bool, default=False
            If True, the input has changed.

        spiking : ndarray(int), default=None
            Neurons spiking in this frame. Index starting at 1. -n if the neuron needs to be restored.
        """

        plot._animate_traj(ax, artists[-1], y)
        if spiking is not None:
            plot._animate_spiking(artists, spiking)
        if input_change:
            centered = np.array([0, -1])
            x0 = x[:, -1]

            self._draw_bbox_2D(centered, x0, ax, artists)
