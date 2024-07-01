from functools import partial

import matplotlib.axes
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .Autoencoder import Autoencoder
from .Low_rank_LIF import Low_rank_LIF
from .utils_neuro import deintegrate, integrate, stimes_from_s

available_plots = [[Autoencoder, 2, 2]]


class Simulation:

    net: Low_rank_LIF
    "Network to run."

    I: float
    "External input current."

    draw_break: str
    "How to break a draw between spikes: 'no', 'slowmo' or 'one'."

    criterion: str
    "How to choose the neuron to spike in case draw_break='slowmo' or 'one'."

    dt: float
    "Time step of the simulation (s)."

    Tmax: float
    "Duration of the simulation (s)."

    x: np.ndarray
    "Integrated input to the network: di x time_steps."

    c: np.ndarray
    "Input to the network. di x time_steps."

    y: np.ndarray
    "Output of the network. do x time_steps."

    y0: np.ndarray
    "Initial output of the network. do x 1."

    r: np.ndarray
    "Rate of the neurons. N x time_steps."

    r0: np.ndarray
    "Initial rate of the neurons. N x 1."

    s: np.ndarray
    "Spike trains of the neurons. bool N x time_steps."

    stimes: np.ndarray
    "Spike times of the neurons. #spikes x 2. First row is the neuron index and the second row the spike time."

    V: np.ndarray
    "Voltage of the neurons. N x time_steps."

    V0: np.ndarray
    "Initial voltage of the neurons. N x 1."

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

        dt : float, default=0.001
            Time step of the simulation (s).

        Tmax : float, default=10
            Duration of the simulation (s)

        c : ndarray of shape (di,time_steps)
            Filtered input to the network. Only x or c should be provided.

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
            assert x.shape[0] == net.di, "x first dimension should be equal to di"
            assert (
                x.shape[1] == time_steps
            ), "x second dim. should be equal to time_steps"

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
            c = deintegrate(x, self.net.lamb, dt)
        else:
            x = integrate(c, self.net.lamb, dt)
        self.x = x
        self.c = c

        if y0 is not None:
            if V0 is not None or r0 is not None:
                raise Warning("y0 was given and prioritized over r0 and V0")
            r0 = self.net.D @ y0
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
        self.stimes = stimes_from_s(s, dt)
        self.V = V

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
        idx = (
            np.argmax(V - self.net.T)
            if self.criterion == "max"
            else np.random.choice(candidates)
        )
        assert isinstance(idx, int)
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
        _, _, artists_spikes = self.plot_spikes(ax=ax2, t=self.Tmax, save=False)
        _, _, artists_rates = self.plot_rates(ax=ax3, t=self.Tmax, save=False)
        artists = [artists_io, artists_spikes, artists_rates]
        if geometry:
            _, _, artists_net = self.net.plot(ax=ax4, x=self.x, y=self.y, save=False)
            artists.append(artists_net)

        plt.tight_layout()
        if save:
            fig.savefig("test-plot.png")

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
        colorsio = [cmap(i) for i in np.linspace(0, 1, self.net.di)]

        for i in range(self.net.di):
            linex = ax.plot(
                xaxis, x[i, :], color=colorsio[i], label=f"x{i + 1}", alpha=0.5
            )[0]
            liney = ax.plot(xaxis, y[i, :], color=colorsio[i], label=f"y{i + 1}")[0]
            artists.append([linex, liney])

        ax.set_ylabel("x(t)/y(t)")
        ax.set_xlabel("Time (s)")
        ax.set_xlim(-0.5, self.Tmax + 0.5)
        ax.set_ylim(np.min(self.y) - 0.05, np.max(self.y) + 0.05)
        ax.legend()

        fig = ax.get_figure()
        assert fig is not None
        if save:
            fig.savefig("test-io.png")

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
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        scatter = ax.scatter(
            stimes[:, 1],
            stimes[:, 0],
            facecolor=[colors[int(val)] for val in stimes[:, 0]],
        )
        artists.append(scatter)

        ax.set_ylabel("s(t)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(-0.5, self.net.N + 0.5)
        ax.set_xlim(-0.5, self.Tmax + 0.5)

        fig = ax.get_figure()
        assert fig is not None
        if save:
            fig.savefig("test-spikes.png")

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
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i in range(self.net.N):
            line = ax.plot(xaxis, r[i, :], color=colors[i], label=f"r{i + 1}")[0]
            artists.append(line)

        ax.set_ylabel("r(t)")
        ax.set_xlabel("Time (s)")
        ax.set_xlim(-0.5, self.Tmax + 0.5)
        ax.set_ylim(np.min(self.r) - 0.05, np.max(self.r) + 0.05)
        ax.legend()

        fig = ax.get_figure()
        assert fig is not None
        if save:
            fig.savefig("test-rates.png")

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
        _, _, artists_spikes = self.plot_spikes(ax=ax2, t=0, save=False)
        _, _, artists_rates = self.plot_rates(ax=ax3, t=0, save=False)
        artists = [artists_io, artists_spikes, artists_rates]

        if geometry:
            x, y = self._crop(t=0, type="io")
            _, _, artists_net = self.net.plot(
                ax=ax4, x=x, y=y, inverse=True, save=False
            )
            artists_net = artists_net[-2:]
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

            self._animate_io(artists=artists[0], t=t)
            self._animate_spikes(artists=artists[1], t=t)
            self._animate_rates(artists=artists[2], t=t)

            if geometry:
                x, y = self._crop(t, "io")

                assert ax4 is not None
                self.net._animate(ax=ax4, artists=artists[3], x=x, y=y)
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

        ani.save("test-inv_ticks.gif", writer="ffmpeg", fps=anim_freq)
        plt.tight_layout()
        plt.show(block=False)

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
            artists[i][0].set_xdata(xaxis)
            artists[i][0].set_ydata(x[i, :])
            artists[i][1].set_xdata(xaxis)
            artists[i][1].set_ydata(y[i, :])

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

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
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
        frame = int(t / self.dt)

        match type:
            case "io":
                x = self.x[:, : frame + 1]
                y = self.y[:, : frame + 1]
                return x, y
            case "stimes":
                stimes = self.stimes[np.where(self.stimes[:, 1] <= t)]
                return (stimes,)
            case "rates":
                r = self.r[:, : frame + 1]
                return (r,)
            case _:
                raise ValueError("type should be 'io', 'stimes' or 'rates'")
