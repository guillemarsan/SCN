from typing import Self

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from .low_rank_LIF import Low_rank_LIF
from .utils_plots import _gradient_line, _line_closest_point, _trick_axis


class Autoencoder(Low_rank_LIF):
    F: np.ndarray
    "Forward weights of the network: Nxdi. F= D^\top"

    E: np.ndarray
    "Encoding weights of the network: Nxdo. E = -D^\top"

    W: np.ndarray
    "Recurrent weights of the network: NxN. W = -D^\top D"

    """
    Autoencoder model.

    Differential equation:

    N neurons, do = di dimensions.

    Attributes
    ----------
    F : ndarray of shape (N, di)
        Forward weights of the network. F= D^\top

    E : ndarray of shape (N, do)
        Encoding weights of the network. E = -D^\top

    D : ndarray of shape (do, N)
        Decoding weights of the network.

    W : ndarray of shape (N, N)
        Recurrent weights of the network. W = -D^\top D

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
        D: np.ndarray,
        T: np.ndarray | None = None,
        lamb: float = 1,
    ) -> None:
        """
        Constructor with specific parameters.

        Parameters
        ----------
        D : ndarray of shape (do, N)
            Decoding weights of the network.

        T : ndarray of shape (N,)
            Threshold of the neurons. If None, Ti = 1/2 ||D_i||^2.

        lamb : float, default=1
            Leak timescale of the network.

        """
        if T is None:
            T = 0.5 * np.linalg.norm(D, axis=0) ** 2
            assert T is not None

        super().__init__(F=D.T, E=-D.T, D=D, T=T, lamb=lamb)

    @classmethod
    def random_init(  # type: ignore[reportIncompatibleMethodOverride]
        cls,
        di: int = 2,
        N: int = 10,
    ) -> Self:
        """
        Constructor with specific dimensions, but random parameters.

        Parameters
        ----------
        di : float, default=1
            Input dimensions.

        N : float, default=10
            Number of neurons.

        """

        # random parameters
        D = np.random.randn(di, N)
        T = 0.5 * np.linalg.norm(D, axis=0) ** 2

        return cls(D, T)

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
            fig.savefig("autoencoder.png")

        return fig, ax, artists

    def _animate(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        artists: list,
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

    def _draw_bbox(
        self,
        x0: np.ndarray,
        ax: matplotlib.axes.Axes,
        artists: list | None = None,
    ) -> list:
        """
        Plot an Autoencoder network.

        Parameters
        ----------
        net : Autoencoder
            Autoencoder to plot.

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
                a = self.D[0, n]
                b = self.D[1, n]
                c = self.T[n] - self.D[:, n].T @ x0
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
                        a,
                        b,
                        color=colors[n],
                        scale=5,
                        scale_units="xy",
                        angles="xy",
                    )
                else:
                    artists[n][2].set_offsets([q0, q1])
                    artists[n][2].set_UVC(a, b)

                if first_frame:
                    artists.append([poly, line, quiver])

            ax.set_xlim(x0[0] - 1, x0[0] + 1)
            ax.set_ylim(x0[1] - 1, x0[1] + 1)
            ax.set_aspect("equal")

        return artists
