from typing import Self

import cvxpy as cp
import numpy as np

from . import boundary
from .low_rank_LIF import Low_rank_LIF


class Single_Population(Low_rank_LIF):
    r"""
    Single population (E or I) model. Subclass of the Low-rank LIF model with
    :math:`\mathbf{W}_{ij} \leq 0` (I) or :math:`\mathbf{W}_{ij} \geq 0` (E).

    :math:`N` neurons, :math:`d_i` input dimensions and :math:`d_o` output dimensions.

    See Also
    --------
    :class:`~SCN.low_rank_LIF.Low_rank_LIF` : Parent model Low_rank_LIF.

    References
    ----------
    Podlaski, W. F., & Machens, C. K. (2024). Approximating nonlinear functions with latent
    boundaries in low-rank excitatory-inhibitory spiking networks. Neural Computation, 36(5), 803-857.
    https://doi.org/10.1162/neco_a_01658

    Mancoo, A., Keemink, S., & Machens, C. K. (2020). Understanding spiking networks through convex optimization.
    Advances in neural information processing systems, 33, 8824-8835.

    Examples
    --------
    >>> from SCN import Single_Population, Simulation
    >>> net = Single_Population.init_2D_random(di=2, N=5, dale="I")
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
    r"Recurrent weights of the network. :math:`N \times N`. :math:`\mathbf{W} \leq 0` (I) or :math:`\mathbf{W} \geq 0` (E)."

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

        :math:`\mathbf{W} = \mathbf{E} \mathbf{D} \leq 0` (I) or :math:`\mathbf{W} = \mathbf{E} \mathbf{D} \geq 0` (E).

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
        W[np.abs(W) < 1e-10] = 0
        assert np.all(W <= 0) or np.all(
            W >= 0
        ), "W should be either positive (E) or negative (I)"

        if isinstance(T, (int, float)):
            T = T * np.ones(D.shape[1])

        assert T is not None and not isinstance(T, (int, float))
        super().__init__(F=F, E=E, D=D, T=T, lamb=lamb)

    @classmethod
    def init_random(  # type: ignore[reportIncompatibleMethodOverride]
        cls,
        di: int = 2,
        do: int = 2,
        N: int = 10,
        dale: str = "I",
        seed: int | None = None,
        T: int | float | np.ndarray = 0.5,
        lamb: float = 1,
    ) -> Self:
        r"""

        Random initialization of the Single Population network.
        (see :func:`~SCN.boundary._sphere_random`)

        Characteristics:
        - :math:`(\mathbf{F}|\mathbf{E})` initialized randomly in a hemisphere of radius 1 in :math:`\mathbb{R}^{d_i+d_o}`
        - :math:`\mathbf{D}` as close to :math:`\pm E^\top` while respecting EI
        - :math:`\mathbf{T} = \mathbf{T} - \mathbf{E}_N` to center the boundary at :math:`(0, 0, ..., -1)`

        Parameters
        ----------
        di : float, default=2
            Input dimensions.

        do: float, default=2
            Output dimensions.

        N : float, default=10
            Number of neurons.

        dale : str, default="I"
            Dale's law of the network. "I" for inhibitory, "E" for excitatory.

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

        # random boundary
        M = boundary._sphere_random(d=di + do, N=N, seed=seed)

        F = M[:, :di]
        E = M[:, di:]

        # standarize in semi-sphere
        E[:, -1] = np.abs(E[:, -1])

        # find D that respects E/I
        D = cp.Variable((do, N))
        penalty = cp.norm(D + E.T) if dale == "I" else cp.norm(D - E.T)
        prob = cp.Problem(
            cp.Minimize(penalty), [E @ D <= 0 if dale == "I" else E @ D >= 0]
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
        N: int = 10,
        dale: str = "I",
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

        N: int, default=10
            Number of neurons.

        dale : str, default="I"
            Dale's law of the network. "I" for inhibitory, "E" for excitatory.

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

        if angle_range is None:
            angle_range = [np.pi / 4, 3 * np.pi / 4]

        assert np.abs(angle_range[1] - angle_range[0]) <= np.pi / 2
        # evenly spaced circular parameters
        E = boundary._2D_circle_spaced(N=N, angle_range=angle_range)

        D = -E.T if dale == "I" else E.T
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
        N: int = 10,
        dale: str = "I",
        angle_range: list | None = None,
        seed: int | None = None,
        T: int | float | np.ndarray = 0.5,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
    ) -> Self:
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

        if angle_range is None:
            angle_range = [np.pi / 4, 3 * np.pi / 4]

        assert np.abs(angle_range[1] - angle_range[0]) <= np.pi / 2
        # evenly spaced circular parameters
        E = boundary._2D_circle_random(N=N, angle_range=angle_range, seed=seed)

        D = -E.T if dale == "I" else E.T
        D = spike_scale * D

        F_scale = 0.1
        E_scale = np.sqrt(1 - F_scale**2)
        F = F_scale * np.random.choice([-1, 1], (N, di))
        E = E_scale * E / np.linalg.norm(E, axis=1)[:, np.newaxis]

        T = T - E[:, 1]
        return cls(F, E, D, T, lamb)
