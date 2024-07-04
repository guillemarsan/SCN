import matplotlib.axes
import matplotlib.figure
import numpy as np


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

    Notes
    -----
    ...

    References
    ----------
    Podlaski, William & Machens, Christian. (2024). Approximating Nonlinear Functions
    With Latent Boundaries in Low-Rank Excitatory-Inhibitory Spiking Networks.
    Neural Computation. 36. 803-857. 10.1162/neco_a_01658.

    Examples
    --------
    >>> from sklearn.neural_network import MLPClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
    ...                                                     random_state=1)
    >>> clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    >>> clf.predict_proba(X_test[:1])
    array([[0.038..., 0.961...]])
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.8...
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
        inverse: bool = False,
        save: bool = True,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list]:
        """
        Plot the network.

        Parameters
        ----------

        """
        # TODO plot_LIF(self)

        ...

    def _animate(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        artists: list,
        spiking: np.ndarray,
    ) -> None:
        """
        Animate the network.

        Parameters
        ----------

        """

        # TODO animate_LIF(self)
        return
