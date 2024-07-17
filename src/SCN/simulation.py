import random
import string
import time
from functools import partial

import matplotlib.axes
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .autoencoder import Autoencoder
from .ei_network import EI_Network
from .low_rank_LIF import Low_rank_LIF
from .single_population import Single_Population
from .utils_neuro import (
    _deintegrate,
    _integrate,
    _neurons_spiked_between,
    _stimes_from_s,
)
from .utils_plots import _get_colors, _save_ani, _save_fig

available_plots = [[Autoencoder, 2, 2], [Single_Population, 1, 2], [EI_Network, 1, 2]]


class Simulation:
    r"""
    Simulation of a network.

    :math:`\dot{\mathbf{V}}(t) = -\lambda \mathbf{V}(t) + \mathbf{F} \mathbf{c}(t) + \mathbf{W} \mathbf{s}(t)`
    (see :class:`~SCN.low_rank_LIF.Low_rank_LIF`)

    Main functions:
        - run: run the network with specific input and integration parameters
        - plot: plot the results of the simulation
        - animate: animate the results of the simulation

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

    net: Low_rank_LIF
    "Network to run. Class :class:`~SCN.low_rank_LIF.Low_rank_LIF` or subclasses."

    x: np.ndarray
    r"Integrated input to the network. :math:`d_i \times time\_steps`."

    y0: np.ndarray
    r"Initial output of the network. :math:`d_o \times 1`."

    r0: np.ndarray
    r"Initial rate of the neurons. :math:`N \times 1`."

    V0: np.ndarray
    r"Initial voltage of the neurons. :math:`N \times 1`."

    I: float
    r"External input current. :math:`N \times 1`."

    draw_break: str
    "How to break a draw between spikes. Either 'no', 'slowmo' or 'one'."

    criterion: str
    "How to choose the neuron to spike in case draw_break='slowmo' or 'one'."

    dt: float
    "Time step of the simulation (s)."

    Tmax: float
    "Duration of the simulation (s)."

    c: np.ndarray
    r"Input to the network. :math:`d_i \times time\_steps`."

    y: np.ndarray
    r"Output of the network. :math:`d_o \times time\_steps`."

    r: np.ndarray
    r"Rate of the neurons. :math:`N \times time\_steps`."

    s: np.ndarray
    r"Spike trains of the neurons. bool :math:`N \times time\_steps`."

    stimes: np.ndarray
    r"Spike times of the neurons. :math:`\#spikes \times 2`.. First row is the neuron index and the second row the spike time."

    V: np.ndarray
    r"Voltage of the neurons. :math:`N \times time\_steps`."

    def run(
        self,
        net: Low_rank_LIF,
        x: np.ndarray | None = None,
        y0: np.ndarray | None = None,
        r0: np.ndarray | None = None,
        V0: np.ndarray | None = None,
        I: float = 0,
        draw_break: str = "no",
        criterion: str = "max",
        dt: float = 0.001,
        Tmax: float = 10,
        c: np.ndarray | None = None,
        tag: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the network.

        Parameters
        ----------

        net : Low_rank_LIF
            Network to run.

        x : ndarray of shape (di,time_steps), default=None
            Input to the network. Only x or c should be provided.

        y0 : ndarray of shape (do,), default=None
            Initial output of the network. Prioritized over r0 and V0.

        V0 : ndarray of shape (N,), default=None
            Initial voltage of the neurons. Prioritized over r0.

        r0 : ndarray of shape (N,), default=None
            Initial rate of the neurons.

        I : float, default=0
            External input current.

        draw_break : str, default='no'
            How to break a draw between spikes:
            - 'no': neurons spike at once in the same time
            - 'slowmo': neurons spike one after the other in the same time step
            - 'one': only one neuron spikes at each time step

        criterion : str, default='max'
            How to choose the neuron to spike in case draw_break='slowmo' or 'one':
            - 'max': neuron with the highest voltage spikes
            - 'rand': neuron is chosen randomly
            - 'inh_max': neuron with the highest voltage spikes (all inhibitory priority)
            - 'inh_rand': neuron is chosen randomly (all inhibitory priority)

        dt : float, default=0.001
            Time step of the simulation (s).

        Tmax : float, default=10
            Duration of the simulation (s)

        c : ndarray of shape (di,time_steps)
            Filtered input to the network. Only x or c should be provided.

        tag : str, default=None
            Tag of the simulation. If None, the tag is randomly generated.

        Returns
        -------
        y: ndarray of shape (do,time_steps)
            Output of the network.

        r: ndarray of shape (N,time_steps)
            Rate of the neurons.

        s: ndarray(bool) of shape (N,time_steps)
            Spike trains of the neurons.

        V: ndarray of shape (N,time_steps)
            Voltage of the neurons.
        """

        time_steps = int(Tmax / dt)

        if x is not None and c is not None:
            raise Warning("Both x and c provided, c will be used")
        elif x is not None:
            if x.ndim == 2:
                assert x.shape[0] == net.di, "x first dimension should be equal to di"
                assert (
                    x.shape[1] == time_steps
                ), "x second dim. should be equal to time_steps"
            elif x.ndim == 1:
                if x.shape[0] == net.di:
                    x = np.tile(x, (1, time_steps))
                elif x.shape[0] == time_steps:
                    x = np.tile(x, (net.di, 1))
                else:
                    raise ValueError("x should have either di or time_steps elements")

        if c is not None:
            assert c.shape[0] == net.di, "c first dimension should be equal to di"
            assert (
                c.shape[1] == time_steps
            ), "c second dim. should be equal to time_steps"

        self.net = net
        self.I = I
        self.draw_break = draw_break
        self.criterion = criterion
        self.dt = dt
        self.Tmax = Tmax

        if c is None:
            assert x is not None, "An input (either x or c) should be provided"
            c = _deintegrate(x, self.net.lamb, dt)
        else:
            x = _integrate(c, self.net.lamb, dt)
        self.x = x
        self.c = c

        if y0 is not None:
            if V0 is not None or r0 is not None:
                raise Warning("y0 was given and prioritized over r0 and V0")
            r0 = np.linalg.lstsq(self.net.D, y0, rcond=None)[0]
            V0 = self.net.F @ x[:, 0] + self.net.E @ y0 + I
        elif r0 is not None:
            if V0 is not None:
                raise Warning("r0 was given and prioritized over V0")
            y0 = self.net.D @ r0
            V0 = self.net.F @ x[:, 0] + self.net.E @ y0 + I
        elif V0 is not None:
            y0 = np.linalg.lstsq(
                self.net.E, (V0 - self.net.F @ x[:, 0] - I), rcond=None
            )[0]
            r0 = np.linalg.lstsq(self.net.D, y0, rcond=None)[0]
        else:
            if isinstance(self.net, Autoencoder):
                y0 = x[:, 0]
                r0 = np.linalg.lstsq(self.net.D, y0, rcond=None)[0]
                V0 = self.net.F @ x[:, 0] + self.net.E @ y0 + I
            else:
                # TODO Start within the subthreshold area
                y0 = np.zeros(self.net.do)
                r0 = np.zeros(self.net.N)
                V0 = self.net.F @ x[:, 0] + self.net.E @ y0 + I

        self.y0 = y0
        self.r0 = r0
        self.V0 = V0

        match draw_break:

            case "no":
                y, r, s, V = self._run_draw()
            case "slowmo":
                y, r, s, V = self._run_slowmo()
            case "one":
                y, r, s, V = self._run_one()
            case _:
                raise ValueError("draw_break should be 'no', 'slowmo' or 'one'")

        self.y = y
        self.r = r
        self.s = s
        self.stimes = _stimes_from_s(s, dt)
        self.V = V
        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")
        self.tag = (
            tag
            if tag is not None
            else "".join(random.choice(string.ascii_letters) for i in range(5))
        )

        return y, r, s, V

    def _run_draw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the network with no draw breaks, i.e. all neurons spike simultaneously in the same time step

        Returns
        -------
        y: ndarray of shape (do,time_steps)
            Output of the network.

        r: ndarray of shape (N,time_steps)
            Rate of the neurons.

        s: ndarray(bool) of shape (N,time_steps)
            Spike trains of the neurons.

        V: ndarray of shape (N,time_steps)
            Voltage of the neurons.
        """

        time_steps = int(self.Tmax / self.dt)

        r = np.zeros([self.net.N, time_steps])
        s = np.zeros([self.net.N, time_steps], dtype=bool)
        V = np.zeros([self.net.N, time_steps])

        V[:, 0] = self.V0
        r[:, 0] = self.r0

        for t in range(time_steps - 1):
            s[:, t][np.where(V[:, t] > self.net.T)] = 1
            V[:, t + 1] = (
                V[:, t]
                + self.dt
                * (-self.net.lamb * V[:, t] + self.net.F @ self.c[:, t] + self.I)
                + self.net.W @ s[:, t]
            )
            r[:, t + 1] = r[:, t] + self.dt * (-self.net.lamb * r[:, t]) + s[:, t]

        y = self.net.D @ r
        return y, r, s, V

    def _run_slowmo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the network with slow motion draw breaks, i.e. all neurons spike in order in the same time step.

        The order is given by the criterion parameter: max or rand.

        Returns
        -------
        y: ndarray of shape (do,time_steps)
            Output of the network.

        r: ndarray of shape (N,time_steps)
            Rate of the neurons.

        s: ndarray(bool) of shape (N,time_steps)
            Spike trains of the neurons.

        V: ndarray of shape (N,time_steps)
            Voltage of the neurons.
        """

        time_steps = int(self.Tmax / self.dt)

        r = np.zeros([self.net.N, time_steps])
        s = np.zeros([self.net.N, time_steps], dtype=bool)
        V = np.zeros([self.net.N, time_steps])

        V[:, 0] = self.V0
        r[:, 0] = self.r0

        for t in range(time_steps - 1):
            candidates = np.where(V[:, t] > self.net.T)[0]
            while len(candidates) > 0:
                idx = self._idx_choose(V[:, t], candidates)
                s[idx, t] = 1
                V[:, t] = V[:, t] + self.net.W[:, idx]
                candidates = np.where(V[:, t] > self.net.T)[0]

            V[:, t + 1] = V[:, t] + self.dt * (
                -self.net.lamb * V[:, t] + self.net.F @ self.c[:, t] + self.I
            )
            r[:, t + 1] = r[:, t] + self.dt * (-self.net.lamb * r[:, t]) + s[:, t]

        y = self.net.D @ r
        return y, r, s, V

    def _run_one(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the network with unique draw breaks, i.e. only one neuron spikes in each time step.

        The winner is given by the criterion parameter: max or rand.

        Returns
        -------
        y: ndarray of shape (do,time_steps)
            Output of the network.

        r: ndarray of shape (N,time_steps)
            Rate of the neurons.

        s: ndarray(bool) of shape (N,time_steps)
            Spike trains of the neurons.

        V: ndarray of shape (N,time_steps)
            Voltage of the neurons.
        """

        time_steps = int(self.Tmax / self.dt)

        r = np.zeros([self.net.N, time_steps])
        s = np.zeros([self.net.N, time_steps], dtype=bool)
        V = np.zeros([self.net.N, time_steps])

        V[:, 0] = self.V0
        r[:, 0] = self.r0

        for t in range(time_steps - 1):
            candidates = np.where(V[:, t] > self.net.T)[0]
            if len(candidates) > 0:
                idx = self._idx_choose(V[:, t], candidates)
                s[idx, t] = 1

            V[:, t + 1] = (
                V[:, t]
                + self.dt
                * (-self.net.lamb * V[:, t] + self.net.F @ self.c[:, t] + self.I)
                + self.net.W @ s[:, t]
            )
            r[:, t + 1] = r[:, t] + self.dt * (-self.net.lamb * r[:, t]) + s[:, t]

        y = self.net.D @ r
        return y, r, s, V

    def _idx_choose(self, V: np.ndarray, candidates: np.ndarray) -> int:
        """
        Choose the neuron to spike in case of draw.

        Parameters
        ----------
        V : np.ndarray
            Voltage of the neurons.

        candidates : np.ndarray
            Neurons that can spike.

        Returns
        -------
        idx : int
            Index of the neuron to spike.
        """

        match self.criterion:
            case "max":
                idx = int(np.argmax(V - self.net.T))
            case "rand":
                idx = np.random.choice(candidates)
            case "inh_max":
                inh = np.all(self.net.W < 0, axis=0)
                if np.any(inh[candidates]):
                    inhidx = np.where(inh)[0]
                    idx = int(np.argmax(V[inhidx] - self.net.T[inhidx]))
                else:
                    idx = int(np.argmax(V - self.net.T))
            case "inh_rand":
                inh = np.all(self.net.W < 0, axis=0)
                if np.any(inh[candidates]):
                    inhidx = np.where(inh)[0]
                    idx = np.random.choice(inhidx)
                else:
                    idx = np.random.choice(candidates)
            case _:
                raise ValueError(
                    "criterion should be 'max', 'rand', 'inh_max' or 'inh_rand'"
                )
        return idx

    # PLOTTING ####

    def plot(
        self,
        geometry: bool = True,
        save: bool = True,
    ) -> tuple[matplotlib.figure.Figure, list, list]:
        """
        Plot the results of the simulation.

        Parameters
        ----------
        geometry : bool, default=True
            If False, do not plot the geometry of the network.

        save : bool, default=True
            If True, save the figure.

        Returns
        -------
        fig: matplotlib.figure.Figure
            Figure of the plot.

        axes: list
            Axes of the plot.

        artists: list
            Artists of the plot.
        """

        fig = plt.figure(figsize=(20, 10))

        geometry = (
            geometry and [type(self.net), self.net.di, self.net.do] in available_plots
        )

        if geometry:
            gs = gridspec.GridSpec(3, 2)
            ax1 = plt.subplot(gs[0, 1])
            ax2 = plt.subplot(gs[1, 1])
            ax3 = plt.subplot(gs[2, 1])
            ax4 = plt.subplot(gs[:, 0])
            axes = [ax1, ax2, ax3, ax4]
        else:
            gs = gridspec.GridSpec(3, 1)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[1, 0])
            ax3 = plt.subplot(gs[2, 0])
            ax4 = None
            axes = [ax1, ax2, ax3]

        _, _, artists_io = self.plot_io(ax=ax1, t=self.Tmax, save=False)
        ax1.set_xlabel("")
        _, _, artists_spikes = self.plot_spikes(ax=ax2, t=self.Tmax, save=False)
        ax2.set_xlabel("")
        _, _, artists_rates = self.plot_rates(ax=ax3, t=self.Tmax, save=False)
        artists = [artists_io, artists_spikes, artists_rates]
        if geometry:
            _, _, artists_net = self.net.plot(ax=ax4, x=self.x, y=self.y, save=False)
            artists.append(artists_net)

        plt.tight_layout()
        if save:
            _save_fig(fig, self.time_stamp + "-" + self.tag + "-plot.png")

        return fig, axes, artists

    def plot_io(
        self, ax: matplotlib.axes.Axes | None = None, t: float = -1, save: bool = True
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list]:
        """
        Plot the input-output of the network as a function of time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes to plot the input-output. If None, a new figure is created.

        t : float, default=-1
            Time to crop the input-output. If -1 the whole simulation is plotted.

        save : bool, default=True
            If True, save the figure.

        Returns
        -------
        fig: matplotlib.figure.Figure
            Figure of the plot.

        ax: matplotlib.axes.Axes
            Axes of the plot.

        artists: list
            Artists of the plot.
        """

        alone = ax is None
        if alone:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.gca()

        x, y = self._crop(t, "io")

        artists = []
        xaxis = np.linspace(0, x.shape[1] * self.dt, x.shape[1])
        cmap = plt.get_cmap("rainbow")
        colorsio = [
            cmap(i) for i in np.linspace(0, 1, np.maximum(self.net.di, self.net.do))
        ]

        linex_arr = []
        for i in range(self.net.di):
            linex = ax.plot(
                xaxis, x[i, :], color=colorsio[i], label=f"x{i + 1}", alpha=0.5
            )[0]
            linex_arr.append(linex)
        artists.append(linex_arr)

        liney_arr = []
        for i in range(self.net.do):
            liney = ax.plot(xaxis, y[i, :], color=colorsio[i], label=f"y{i + 1}")[0]
            liney_arr.append(liney)
        artists.append(liney_arr)

        ax.set_ylabel("x(t)/y(t)")
        ax.set_xlabel("time (s)")
        ax.set_xlim(-0.5, self.Tmax + 0.5)
        ax.set_ylim(
            np.min([np.min(self.y), np.min(self.x)]) - 0.05,
            np.max([np.max(self.y), np.max(self.x)]) + 0.05,
        )
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig = ax.get_figure()
        assert fig is not None
        if save:
            _save_fig(fig, self.time_stamp + "-" + self.tag + "-io-plot.png")

        return fig, ax, artists

    def plot_spikes(
        self, ax: matplotlib.axes.Axes | None = None, t: float = -1, save: bool = True
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list]:
        """
        Plot the spike events of the network as a function of time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes to plot the spikes. If None, a new figure is created.

        t : float, default=-1
            Time to crop the spikes. If -1 the whole simulation is plotted.

        save : bool, default=True
            If True, save the figure.

        Returns
        -------
        fig: matplotlib.figure.Figure
            Figure of the plot.

        ax: matplotlib.axes.Axes
            Axes of the plot.

        artists: list
            Artists of the plot.
        """

        alone = ax is None
        if alone:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.gca()

        (stimes,) = self._crop(t, "stimes")
        artists = []

        colors = _get_colors(self.net.N, self.net.W)

        scatter = ax.scatter(
            stimes[:, 1],
            stimes[:, 0],
            facecolor=[colors[int(val)] for val in stimes[:, 0]],
        )
        artists.append(scatter)

        ax.set_ylabel("s(t)")
        ax.set_xlabel("time (s)")
        ax.set_ylim(-0.5, self.net.N - 0.5)
        ax.set_yticks(np.arange(self.net.N))
        ax.set_xlim(-0.5, self.Tmax + 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig = ax.get_figure()
        assert fig is not None
        if save:
            _save_fig(fig, self.time_stamp + "-" + self.tag + "-spikes-plot.png")

        return fig, ax, artists

    def plot_rates(
        self, ax: matplotlib.axes.Axes | None = None, t: float = -1, save: bool = True
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list]:
        """
        Plot the rates of the network as a function of time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes to plot the rates. If None, a new figure is created.

        t : float, default=-1
            Time to crop the rates. If -1 the whole simulation is plotted.

        save : bool, default=True
            If True, save the figure.

        Returns
        -------
        fig: matplotlib.figure.Figure
            Figure of the plot.

        ax: matplotlib.axes.Axes
            Axes of the plot.

        artists: list
            Artists of the plot.
        """

        alone = ax is None
        if alone:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.gca()

        (r,) = self._crop(t, "rates")

        artists = []
        xaxis = np.linspace(0, r.shape[1] * self.dt, r.shape[1])
        colors = _get_colors(self.net.N, self.net.W)

        for i in range(self.net.N):
            line = ax.plot(xaxis, r[i, :], color=colors[i], label=f"r{i + 1}")[0]
            artists.append(line)

        ax.set_ylabel("r(t)")
        ax.set_xlabel("time (s)")
        ax.set_xlim(-0.5, self.Tmax + 0.5)
        ax.set_ylim(np.min(self.r) - 0.05, np.max(self.r) + 0.05)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig = ax.get_figure()
        assert fig is not None
        if save:
            _save_fig(fig, self.time_stamp + "-" + self.tag + "-rates-plot.png")

        return fig, ax, artists

    # ANIMATION ####

    def animate(
        self,
        geometry: bool = True,
    ) -> None:
        """
        Animate the results of a simulation.

        Parameters
        ----------
        geometry : bool, default=True
            If False, do not plot the geometry of the network.

        """

        fig = plt.figure(figsize=(20, 10))

        geometry = (
            geometry and [type(self.net), self.net.di, self.net.do] in available_plots
        )

        if geometry:
            gs = gridspec.GridSpec(3, 2)
            ax1 = plt.subplot(gs[0, 1])
            ax2 = plt.subplot(gs[1, 1])
            ax3 = plt.subplot(gs[2, 1])
            ax4 = plt.subplot(gs[:, 0])
        else:
            gs = gridspec.GridSpec(3, 1)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[1, 0])
            ax3 = plt.subplot(gs[2, 0])
            ax4 = None

        artists = []
        _, _, artists_io = self.plot_io(ax=ax1, t=0, save=False)
        ax1.set_xlabel("")
        _, _, artists_spikes = self.plot_spikes(ax=ax2, t=0, save=False)
        ax2.set_xlabel("")
        _, _, artists_rates = self.plot_rates(ax=ax3, t=0, save=False)
        artists = [artists_io, artists_spikes, artists_rates]

        if geometry:
            x, y = self._crop(t=0, type="io")
            _, _, artists_net = self.net.plot(ax=ax4, x=x, y=y, save=False)
            artists.append(artists_net)

        def flatten(l: list) -> list:
            return (
                [l]
                if not isinstance(l, list)
                else [a for sub in l for a in flatten(sub)]
            )

        artists_flatten = flatten(artists)

        def init():
            return artists_flatten

        def update(frame, artists, artists_flatten):
            t = frame / anim_freq
            tpast = (frame - 1) / anim_freq
            tpastpast = (frame - 2) / anim_freq

            self._animate_io(artists=artists[0], t=t)
            self._animate_spikes(artists=artists[1], t=t)
            self._animate_rates(artists=artists[2], t=t)

            if geometry:
                x, y = self._crop(t, "io")

                newspiked = _neurons_spiked_between(self.stimes, tpast, t)
                oldspiked = _neurons_spiked_between(self.stimes, tpastpast, tpast)
                spiking = np.concatenate(
                    [
                        -(np.array(oldspiked, dtype=int) + 1),
                        np.array(newspiked, dtype=int) + 1,
                    ]
                )

                input_change = (
                    not np.array_equal(
                        x[:, int(t / self.dt)], x[:, int(tpast / self.dt)]
                    )
                    if tpast >= 0
                    else False
                )

                assert ax4 is not None
                self.net._animate(
                    ax=ax4,
                    artists=artists[3],
                    x=x,
                    y=y,
                    input_change=input_change,
                    spiking=spiking,
                )

            return artists_flatten

        anim_freq = 10
        frames = self.Tmax * anim_freq

        ani = FuncAnimation(
            fig,
            func=partial(update, artists=artists, artists_flatten=artists_flatten),
            frames=np.arange(0, frames),
            init_func=init,
            blit=True,
        )

        _save_ani(ani, self.time_stamp + "-" + self.tag + "-animation.gif", anim_freq)

    def _animate_io(self, artists: list, t: float) -> None:
        """
        Animate the input-output of the network as a function of time.
        Modifies the artists for the frame at time t

        Parameters
        ----------
        artists : list
            Artists of the plot.

        t : float
            Time to crop the input-output. For the frames of the animation.
        """

        x, y = self._crop(t, "io")
        xaxis = np.linspace(0, x.shape[1] * self.dt, x.shape[1])
        for i in range(self.net.di):
            artists[0][i].set_xdata(xaxis)
            artists[0][i].set_ydata(x[i, :])

        for i in range(self.net.do):
            artists[1][i].set_xdata(xaxis)
            artists[1][i].set_ydata(y[i, :])

    def _animate_spikes(self, artists: list, t: float) -> None:
        """
        Animate the spikes of the network as a function of time.
        Modifies the artists for the frame at time t

        Parameters
        ----------
        artists : list
            Artists of the plot.

        t : float
            Time to crop the spikes. For the frames of the animation.
        """

        colors = _get_colors(self.net.N, self.net.W)
        (stimes,) = self._crop(t, "stimes")
        artists[0].set_facecolor([colors[int(val)] for val in stimes[:, 0]])
        artists[0].set_offsets(stimes[:, ::-1])

    def _animate_rates(
        self,
        artists: list,
        t: float,
    ) -> None:
        """
        Animate the rates of the network as a function of time.
        Modifies the artists for the frame at time t

        Parameters
        ----------
        artists : list
            Artists of the plot.

        t : float
            Time to crop the rates. For the frames of the animation.

        """

        (r,) = self._crop(t, "rates")
        xaxis = np.linspace(0, r.shape[1] * self.dt, r.shape[1])
        for i in range(self.net.N):
            artists[i].set_xdata(xaxis)
            artists[i].set_ydata(r[i, :])

    def _crop(self, t: float = -1, type: str = "io") -> tuple[np.ndarray, ...]:
        """
        Crop the results of the simulation to time t. For animation purposes.

        Parameters
        ----------
        t : float, default=-1
            Time to crop the results. If -1 the whole simulation is returned.

        type : str, default='io'
            Type of results to crop: 'io' (x,y), 'stimes' (stimes) or 'rates' (r).

        Returns
        -------
        x, y : np.ndarray of (di, t/dt), np.ndarray (do, t/dt)
        stimes: np.ndarray of (#spikes at time < t, 2)
        rates: np.ndarray (N, t/dt)
            Cropped results.
        """

        if t == -1:
            t = self.Tmax
        time_step = int(t / self.dt)

        match type:
            case "io":
                x = self.x[:, : time_step + 1]
                y = self.y[:, : time_step + 1]
                return x, y
            case "stimes":
                stimes = self.stimes[np.where(self.stimes[:, 1] <= t)]
                return (stimes,)
            case "rates":
                r = self.r[:, : time_step + 1]
                return (r,)
            case _:
                raise ValueError("type should be 'io', 'stimes' or 'rates'")
