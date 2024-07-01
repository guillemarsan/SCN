from typing import Self

import matplotlib.axes
import matplotlib.figure
import numpy as np


class Low_rank_LIF:
    F: np.ndarray
    "Forward weights of the network: Nxdi."

    E: np.ndarray
    "Encoding weights of the network: Nxdo."

    D: np.ndarray
    "Decoding weights of the network: doxN."

    W: np.ndarray
    "Recurrent weights of the network: NxN. W = ED"

    lamb: float
    "Leak timescale of the network."

    T: np.ndarray
    "Threshold of the neurons: Nx1."

    """
    Low-rank LIF model.

    Differential equation:

    N neurons, di input dimensions, do output dimensions.

    Attributes
    ----------
    F : ndarray of shape (N, di)
        Forward weights of the network.

    E : ndarray of shape (N, do)
        Encoding weights of the network.

    D : ndarray of shape (do, N)
        Decoding weights of the network.

    W : ndarray of shape (N, N)
        Recurrent weights of the network. W = ED.

    lamb : float, default=1
        Leak timescale of the network.

    T : ndarray of shape (N,)
        Threshold of the neurons.

    See Also
    --------
    MLPRegressor : Multi-layer Perceptron regressor.

    Notes
    -----
    Blabla.

    References
    ----------
    Hinton, Geoffrey E. "Connectionist learning procedures."
    Artificial intelligence 40.1 (1989): 185-234.

    :arxiv:`Kingma, Diederik, and Jimmy Ba (2014)
    "Adam: A method for stochastic optimization." <1412.6980>`

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
    def random_init(
        cls,
        di: int = 1,
        N: int = 10,
        do: int = 1,
    ) -> Self:
        """
        Constructor with specific dimensions, but random parameters.

        Parameters
        ----------
        di : float, default=1
            Input dimensions.

        N : float, default=10
            Number of neurons.

        do : float, default=1
            Output dimensions.

        """

        # random parameters
        F = np.random.randn(N, di)
        E = np.random.randn(N, do)
        D = np.random.randn(do, N)
        T = np.random.randn(N)

        return cls(F, E, D, T)

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
    ) -> None:

        # TODO animate_LIF(self)
        return
