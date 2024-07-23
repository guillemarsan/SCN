from typing import Self

import numpy as np

from . import boundary
from .low_rank_LIF import Low_rank_LIF


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
    Podlaski, W. F., & Machens, C. K. (2024). Approximating nonlinear functions with latent
    boundaries in low-rank excitatory-inhibitory spiking networks. Neural Computation, 36(5), 803-857.
    https://doi.org/10.1162/neco_a_01658

    Examples
    --------
    >>> from SCN import EI_Network, Simulation
    >>> net = EI_Network.init_2D_spaced(di=2, NE=2, NI=3)
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
            + " use T, if you want to change the spike size use D"
        )

        # assert EI
        W = E @ D
        W[np.abs(W) < 1e-9] = 0
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
        NE: int = 0,
        NI: int = 0,
        seed: int | None = None,
        T: int | float | np.ndarray = 0.5,
        lamb: float = 1,
        latent_sep: np.ndarray | None = None,
    ) -> Self:
        r"""
        Random initialization of the EI Network.
        (see :func:`~SCN.boundary._sphere_random`)

        Characteristics:
        - :math:`(\mathbf{F}|\mathbf{E})` initialized randomly in a hemihemisphere of radius 1 in :math:`\mathbb{R}^{d_i+d_o}`
        - :math:`\mathbf{D}` as close to :math:`\pm E^\top` while respecting EI
        - :math:`\mathbf{T} = \mathbf{T} - \mathbf{E}_N` to center the boundary at :math:`(0, 0, ..., 0, 1, -1)`

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

        latent_sep : ndarray(bool) of shape (do, 2), default=None
            If not None, the latent dimensions are separated. The first (second) column is True if the latent dimension is influenced
            by excitatory (inhibitory) neurons.

        Returns
        -------
        net: Autoencoder
            Autoencoder network with random boundary.
        """

        N = NE + NI
        assert N > 0, "There needs to be at least one neuron"

        # random boundary
        M = boundary._sphere_random(d=di + do, N=N, seed=seed)

        F = M[:, :di]
        E = M[:, di:]

        # standarize in semi-sphere
        E[:, -1:] = np.abs(E[:, -1:])

        # find D that respects E/I and latent constraints
        D = boundary._dale_decoder_ortho(E, NE, NI, latent_sep=latent_sep)

        # TODO: take this to I
        T = T - E[:, -1]
        return cls(F=F, E=E, D=D, T=T, lamb=lamb)

    @classmethod
    def init_2D_spaced(
        cls,
        di: int = 1,
        NE: int = 0,
        NI: int = 0,
        angle_range: list | None = None,
        Fseed: int | None = None,
        T: int | float | np.ndarray = 0.5,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
        latent_sep: np.ndarray | None = None,
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

        Fseed : int or None, default=None
            Seed for the random number generator for determining the sign of :math:`\mathbf{F}`.

        T : int, float or ndarray of shape (N,), default = 0.5
            Threshold of the neurons.

        lamb : float, default=1
            Leak timescale of the network.

        spike_scale : int, float or ndarray, default=1
            Scale of the spikes.

        latent_sep : ndarray(bool) of shape (do, 2), default=None
            If not None, the latent dimensions are separated. The first (second) column is True if the latent dimension is influenced
            by excitatory (inhibitory) neurons.
        Returns
        -------
        net: Single_Population
            Single_Population network with regularly spaced latent boundary.
        """
        N = NE + NI
        assert N > 0, "There needs to be at least one neuron"

        if angle_range is None:
            angle_range = [np.pi / 4, 3 * np.pi / 4]

        assert (
            np.abs(angle_range[1] - angle_range[0]) <= np.pi / 2
        ), "Angle range too large to mantain E/I"
        assert (NE != 0 and NI != 0) or not np.any(
            latent_sep == 1
        ), "For there to be latent separation there needs to be an E and I population"

        # evenly spaced circular parameters
        E = boundary._2D_circle_spaced(N=N, angle_range=angle_range, edges=False)

        D = spike_scale * boundary._dale_decoder_ortho(
            E, NE, NI, optim=latent_sep is not None, latent_sep=latent_sep
        )

        F_scale = 0.5
        E_scale = np.sqrt(1 - F_scale**2)
        if Fseed is not None:
            np.random.seed(Fseed)
        F = F_scale / np.sqrt(di) * np.random.choice([-1, 1], (N, di))
        E = E_scale * E / np.linalg.norm(E, axis=1)[:, np.newaxis]

        T = T - E[:, -1]
        return cls(F, E, D, T, lamb)

    @classmethod
    def init_2D_random(
        cls,
        di: int = 1,
        NE: int = 0,
        NI: int = 0,
        angle_range: list | None = None,
        seed: int | None = None,
        Fseed: int | None = None,
        T: int | float | np.ndarray = 0.5,
        lamb: float = 1,
        spike_scale: int | float | np.ndarray = 1,
        latent_sep: np.ndarray | None = None,
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
            Range of angles for the neurons. If None, the range is :math:`[\pi/8, 3\pi/8]`.

        seed : int or None, default=None
            Seed for the random number generator.

        Fseed : int or None, default=None
            Seed for the random number generator for determining the sign of :math:`\mathbf{F}`.

        T : int, float or ndarray of shape (N,), default = 0.5
            Threshold of the neurons.

        lamb : float, default=1
            Leak timescale of the network.

        spike_scale : int, float or ndarray, default=1
            Scale of the spikes.

        latent_sep : ndarray(bool) of shape (do, 2), default=None
            If not None, the latent dimensions are separated. The first (second) column is True if the latent dimension is influenced
            by excitatory (inhibitory) neurons.

        Returns
        -------
        net: Single_Population
            Single_Population network with randomly spaced latent boundary.
        """
        N = NE + NI
        assert N > 0, "There needs to be at least one neuron"

        if angle_range is None:
            angle_range = [np.pi / 4, 3 * np.pi / 4]

        assert (
            np.abs(angle_range[1] - angle_range[0]) <= np.pi / 2
        ), "Angle range too large to mantain E/I"
        assert (NE != 0 and NI != 0) or not np.any(
            latent_sep == 1
        ), "For there to be latent separation there needs to be an E and I population"

        # evenly spaced circular parameters
        E = boundary._2D_circle_random(N=N, angle_range=angle_range, seed=seed)

        D = spike_scale * boundary._dale_decoder_ortho(
            E, NE, NI, optim=latent_sep is not None, latent_sep=latent_sep
        )

        F_scale = 0.5
        E_scale = np.sqrt(1 - F_scale**2)
        if Fseed is not None:
            np.random.seed(Fseed)
        F = F_scale / np.sqrt(di) * np.random.choice([-1, 1], (N, di))
        E = E_scale * E / np.linalg.norm(E, axis=1)[:, np.newaxis]

        # T = T + E[:, -2]
        T = T - E[:, -1]
        return cls(F, E, D, T, lamb)
