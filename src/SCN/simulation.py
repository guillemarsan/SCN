import os
import pickle
import random
import string
import time
from functools import partial

import cvxpy as cp
import matplotlib.axes
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from gekko import GEKKO
from matplotlib.animation import FuncAnimation
from scipy.optimize import nnls

from .autoencoder import Autoencoder
from .low_rank_LIF import Low_rank_LIF
from .utils_neuro import (
    _canon_symmetric,
    _deintegrate,
    _integrate,
    _neurons_spiked_between,
    _stimes_from_s,
)
from .utils_plots import _get_colors, _save_ani, _save_fig


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

    I: np.ndarray
    r"External input current. :math: `N \times time\_steps`."

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

    def __init__(
        self,
        net: Low_rank_LIF,
        x: np.ndarray | None = None,
        y0: np.ndarray | None = None,
        r0: np.ndarray | None = None,
        V0: np.ndarray | None = None,
        I: float | np.ndarray = 0.0,
        c: np.ndarray | None = None,
        dt: float = 0.001,
        Tmax: float = 10,
        voltage_bias: str = "",
        latent_bias: bool = False,
        tag: str | None = None,
    ):
        """
        Initialize the simulation.

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

        I : float or ndrray of shape (N, time_steps), default=0
            External input current.

        c : ndarray of shape (di,time_steps), default=None
            Filtered input to the network. Only x or c should be provided.

        dt : float, default=0.001
            Time step of the simulation (s).

        Tmax : float, default=10
            Duration of the simulation (s)

        voltage_bias : str, default=""
            If "", the voltage is not biased.
            If "F", the voltage is biased by V(t) = Ey(t) + Fx(t).
            If "I", the voltage is biased by V(t) = Ey(t) + I.
            If "FI", the voltage is biased by V(t) = Ey(t) + Fx(t) + I.

        latent_bias : bool, default=False
            If False, the latent space is not biased.
            If True, the latent space is biased by y(t) = D r(t) + b with b such that E b = I.

        tag : str, default=None
            Tag of the simulation. If None, the tag is randomly generated.
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
                    x = np.tile(x[:, np.newaxis], (1, time_steps))
                elif x.shape[0] == time_steps:
                    x = np.tile(x, (net.di, 1))
                else:
                    raise ValueError("x should have either di or time_steps elements")

        assert (
            type(I) is float or type(I) is np.ndarray
        ), "I should be a float or a ndarray"
        if type(I) is float:
            I = I * np.ones((net.N, time_steps))
        elif type(I) is np.ndarray:
            if I.ndim == 1:
                if I.shape[0] == net.N:
                    I = np.tile(I[:, np.newaxis], (1, time_steps))
                elif I.shape[0] == time_steps:
                    I = np.tile(I, (net.N, 1))
                else:
                    raise ValueError("I should have either N or time_steps elements")
            else:
                assert I.shape[0] == net.N, "I first dimension should be equal to N"
                assert (
                    I.shape[1] == time_steps
                ), "I second dim. should be equal to time_steps"
            assert not (
                voltage_bias in {"I", "FI"} and not np.all(I == I[:, [0]])
            ), "To bias the voltage with I, I should be constant in time"
        assert type(I) is np.ndarray

        bias = np.zeros((net.do,))
        if latent_bias:
            assert np.all(
                I == I[:, [0]]
            ), "To bias the latent space, I should be constant in time"
            bias, res, *_ = np.linalg.lstsq(net.E, I[:, 0], rcond=None)
            assert (
                len(res) == 0 or res[0] < 1e-6
            ) and bias is not None, (
                "To bias the latent space, I should be in the column space of E"
            )
        assert type(I) is np.ndarray
        assert type(bias) is np.ndarray or not latent_bias

        if c is not None:
            assert c.shape[0] == net.di, "c first dimension should be equal to di"
            assert (
                c.shape[1] == time_steps
            ), "c second dim. should be equal to time_steps"

        self.net = net
        self.I = I
        self.voltage_bias = voltage_bias
        self.latent_bias = latent_bias
        self.bias = bias
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
            if latent_bias:
                y0 = y0 - bias
            if V0 is not None or r0 is not None:
                raise Warning("y0 was given and prioritized over r0 and V0")
            r0, res = nnls(self.net.D, y0)
            assert (
                r0 is not None and res < 1e-6
            ), "Not possible to find r0 s.t. D r0 = y0"
            V0 = self.net.E @ y0
        elif r0 is not None:
            if V0 is not None:
                raise Warning("r0 was given and prioritized over V0")
            y0 = self.net.D @ r0
            V0 = self.net.E @ y0
        elif V0 is not None:
            match voltage_bias:
                case "F":
                    V0 = V0 - self.net.F @ x[:, 0]
                case "I":
                    V0 = V0 - I[:, 0]
                case "FI":
                    V0 = V0 - self.net.F @ x[:, 0] - I[:, 0]
            y0, res, *_ = np.linalg.lstsq(
                self.net.E,
                V0,
                rcond=None,
            )
            assert (
                len(res) == 0 or res[0] < 1e-6
            ), "Not possible to find y0 s.t. V0 = E y0"
            r0, res = nnls(self.net.D, y0)
            assert (
                r0 is not None and res < 1e-6
            ), "Not possible to find r0 s.t. D r0 = y0"
        else:
            if isinstance(self.net, Autoencoder):
                y0 = x[:, 0]
                r0, res = nnls(self.net.D, y0)
                assert (
                    r0 is not None and res < 1e-6
                ), "Not possible to find r0 s.t. D r0 = y0"
                V0 = self.net.E @ y0
            else:
                # TODO Start within the subthreshold area
                y0 = np.zeros(self.net.do)
                r0 = np.zeros(self.net.N)
                V0 = self.net.E @ y0
        self.y0 = y0
        self.r0 = r0
        self.V0 = V0

        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")
        self.stag = (
            "s" + tag
            if tag is not None
            else "s" + "".join(random.choice(string.ascii_letters) for i in range(5))
        )
        self.rtag = ""
        self.otag = ""

    @classmethod
    def init_optim(
        cls,
        Q: np.ndarray,
        b: np.ndarray,
        E: np.ndarray,
        Tp: np.ndarray,
        spike_scale: float = 1,
        y0: np.ndarray | None = None,
        r0: np.ndarray | None = None,
        V0: np.ndarray | None = None,
        dt: float = 0.001,
        Tmax: float = 10,
        tag: str | None = None,
    ):

        net = Low_rank_LIF.init_optim(
            Q=Q, E=E, T=np.ones(E.shape[0]), spike_scale=spike_scale, Fseed=0
        )

        net.F = net.T[:, np.newaxis] - Tp[:, np.newaxis]
        x = np.ones((1, int(Tmax / dt)))

        I = -net.E @ np.linalg.inv(Q) @ b
        I = I[:, np.newaxis] * np.ones((1, x.shape[1]))

        return cls(
            net=net,
            x=x,
            y0=y0,
            r0=r0,
            V0=V0,
            I=I,
            dt=dt,
            Tmax=Tmax,
            voltage_bias="FI",
            latent_bias=True,
            tag=tag,
        )

    def run(
        self,
        draw_break: str = "no",
        criterion: str = "max",
        tag: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the network.

        Parameters
        ----------

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

        tag : str, default=None
            Tag of the run. If None, the tag is randomly generated.

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

        self.draw_break = draw_break
        self.criterion = criterion

        match draw_break:

            case "no":
                y, r, s, V = self._run_draw()
            case "slowmo":
                y, r, s, V = self._run_slowmo()
            case "one":
                y, r, s, V = self._run_one()
            case _:
                raise ValueError("draw_break should be 'no', 'slowmo' or 'one'")

        self.r = r
        self.s = s
        self.stimes = _stimes_from_s(s, self.dt)
        match self.voltage_bias:
            case "":
                self.Tv = self.net.T[:, np.newaxis] - self.net.F @ self.x - self.I
            case "F":
                V = V + self.net.F @ self.x
                self.Tv = self.net.T[:, np.newaxis] - self.I
            case "I":
                V = V + self.I
                self.Tv = self.net.T[:, np.newaxis] - self.net.F @ self.x
            case "FI":
                V = V + self.net.F @ self.x + self.I
                self.Tv = self.net.T[:, np.newaxis] * np.ones_like(self.I)

        if not self.latent_bias:
            self.Ty = self.net.T[:, np.newaxis] - self.net.F @ self.x - self.I
        else:
            y = y + self.bias[:, np.newaxis]
            self.Ty = self.net.T[:, np.newaxis] - self.net.F @ self.x

        self.V = V
        self.y = y

        self.rtag = (
            "r" + tag
            if tag is not None
            else "r" + "".join(random.choice(string.ascii_letters) for i in range(2))
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
            s[:, t][
                np.where(
                    V[:, t] > self.net.T - self.net.F @ self.x[:, t] - self.I[:, t]
                )
            ] = 1
            V[:, t + 1] = (
                V[:, t] + self.dt * (-self.net.lamb * V[:, t]) + self.net.W @ s[:, t]
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
            effthresh = self.net.T - self.net.F @ self.x[:, t] - self.I[:, t]
            candidates = np.where(V[:, t] > effthresh)[0]
            while len(candidates) > 0:
                idx = self._idx_choose(V[:, t], effthresh, candidates)
                s[idx, t] = 1
                V[:, t] = V[:, t] + self.net.W[:, idx]
                candidates = np.where(V[:, t] > effthresh)[0]

            V[:, t + 1] = V[:, t] + self.dt * (-self.net.lamb * V[:, t])
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
            effthresh = self.net.T - self.net.F @ self.x[:, t] - self.I[:, t]
            candidates = np.where(V[:, t] > effthresh)[0]
            if len(candidates) > 0:
                idx = self._idx_choose(V[:, t], effthresh, candidates)
                s[idx, t] = 1

            V[:, t + 1] = (
                V[:, t] + self.dt * (-self.net.lamb * V[:, t]) + self.net.W @ s[:, t]
            )
            r[:, t + 1] = r[:, t] + self.dt * (-self.net.lamb * r[:, t]) + s[:, t]

        y = self.net.D @ r
        return y, r, s, V

    def _idx_choose(
        self, V: np.ndarray, effthresh: np.ndarray, candidates: np.ndarray
    ) -> int:
        """
        Choose the neuron to spike in case of draw.

        Parameters
        ----------
        V : np.ndarray
            Voltage of the neurons.

        effthresh : np.ndarray
            Effective threshold of the neurons.

        candidates : np.ndarray
            Neurons that can spike.

        Returns
        -------
        idx : int
            Index of the neuron to spike.
        """

        match self.criterion:
            case "max":
                idx = int(np.argmax(V - effthresh))
            case "rand":
                idx = np.random.choice(candidates)
            case "inh_max":
                inh = np.argwhere(np.all(self.net.W < 0, axis=0)).flatten()
                inh_cand = np.intersect1d(candidates, inh)
                if len(inh_cand) > 0:
                    idx = inh_cand[np.argmax(V[inh_cand] - effthresh[inh_cand])]
                else:
                    idx = int(np.argmax(V - effthresh))
            case "inh_rand":
                inh = np.argwhere(np.all(self.net.W < 0, axis=0)).flatten()
                inh_cand = np.intersect1d(candidates, inh)
                if len(inh_cand) > 0:
                    idx = np.random.choice(inh_cand)
                else:
                    idx = np.random.choice(candidates)
            case _:
                raise ValueError(
                    "criterion should be 'max', 'rand', 'inh_max' or 'inh_rand'"
                )
        return idx

    # OPTIMIZE ####

    def optimize(
        self,
        Q: np.ndarray | None = None,
        options: list | None = None,
        tag: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimize the network.

        Parameters
        ----------

        Q : ndarray of shape (do,do), default=None
            Matrix of the optimization. If None, it is inferred from the decoders and encoders.


        options : ndarray of str, default=None
            Options of the optimization. Subset of ['y_op', 'y_op_lim', 'r_op', 'r_op_lim']. If None, all are computed.

        tag : str, default=None
            Tag of the optimization. If None, the tag is randomly generated.

        Returns
        -------
        y_op: ndarray of shape (do,time_steps)
            Latent optimum of the network.

        y_op_lim: ndarray of shape (do,time_steps)
            Latent optimum of the network with infinite rates / infinitesimal spikes.

        r_op: ndarray of shape (N,time_steps)
            Rate optimum of the neurons.

        r_op_lim: ndarray of shape (N,time_steps)
            Rate optimum of the neurons with infinite rates / infinitesimal spikes.
        """

        if Q is None:
            Q, residuals, _, _ = np.linalg.lstsq(self.net.D.T, -self.net.E, rcond=None)
            assert (
                np.allclose(residuals, 0, atol=1e-10)
                and np.allclose(Q, Q.T, atol=1e-10)
                and np.all(np.linalg.eigvals(Q) >= 0)
            ), "Q inference only possibe for convex case with N>=do: There must be a unique matrix Q such that Q sym, Q>0 and QD = -E^T"
        else:
            assert Q.shape[0] == self.net.do, "Q first dimension should be equal to do"
            assert Q.shape[1] == self.net.do, "Q second dimension should be equal to do"
            assert np.allclose(Q, Q.T, atol=1e-10), "Q should be symmetric"

        if options is None:
            options = ["y_op", "y_op_lim", "r_op", "r_op_lim"]

        if np.all(
            np.linalg.eigvals(Q) >= 0
        ):  # positive semidefinite -> convex optimization
            y_op, y_op_lim, r_op, r_op_lim = self._optimize_cvx(Q, options)
        elif np.all(
            np.linalg.eigvals(Q) < 0
        ):  # negative definite -> concave optimization
            y_op, y_op_lim, r_op, r_op_lim = self._optimize_cvx(-Q, options)
        else:
            # not positive semidefinite -> non-convex optimization
            # TODO
            y_op, y_op_lim, r_op, r_op_lim = self._optimize_cvx_ccv(Q, options)
            # raise ValueError("Non convex optimization not implemented yet")

        if self.latent_bias:
            if "y_op" in options:
                y_op += self.bias[:, np.newaxis]
            if "y_op_lim" in options:
                y_op_lim += self.bias[:, np.newaxis]

        if "y_op" in options:
            self.y_op = y_op
        if "y_op_lim" in options:
            self.y_op_lim = y_op_lim
        if "r_op" in options:
            self.r_op = r_op
        if "r_op_lim" in options:
            self.r_op_lim = r_op_lim

        self.otag = (
            "o" + tag
            if tag is not None
            else "o" + "".join(random.choice(string.ascii_letters) for i in range(2))
        )

        return y_op, y_op_lim, r_op, r_op_lim

    def _optimize_cvx(self, Q, options):

        inps = np.vstack([self.x, self.I])
        inps_values = np.unique(inps, axis=1)

        xp = cp.Parameter(self.net.di)
        Ip = cp.Parameter(self.net.N)

        probs = []
        y_opv = cp.Variable(self.net.do)
        y_opv_lim = cp.Variable(self.net.do)
        r_opv = cp.Variable(self.net.N)
        r_opv_lim = cp.Variable(self.net.N)
        if "y_op" in options:
            obj = cp.Minimize(y_opv.T @ Q @ y_opv)
            constraints = [
                self.net.F @ xp
                + self.net.E @ y_opv
                + Ip
                - self.net.T
                - np.diag(self.net.W) / 2
                <= 0
            ]
            prob = cp.Problem(obj, constraints)
            probs.append(prob)
        if "y_op_lim" in options:
            obj = cp.Minimize(y_opv_lim.T @ Q @ y_opv_lim)
            constraints = [
                self.net.F @ xp + self.net.E @ y_opv_lim + Ip - self.net.T <= 0
            ]
            prob = cp.Problem(obj, constraints)
            probs.append(prob)
        if "r_op" in options:
            obj = cp.Minimize(
                -cp.quad_form(r_opv, self.net.W)
                + 2
                * r_opv.T
                @ (self.net.T - self.net.F @ xp - Ip + np.diag(self.net.W) / 2)
            )
            constraints = [r_opv >= 0]
            prob = cp.Problem(obj, list(constraints))
            probs.append(prob)
        if "r_op_lim" in options:
            obj = cp.Minimize(
                -cp.quad_form(r_opv_lim, self.net.W)
                + 2 * r_opv_lim.T @ (self.net.T - self.net.F @ xp - Ip)
            )
            constraints = [r_opv_lim >= 0]
            prob = cp.Problem(obj, list(constraints))
            probs.append(prob)

        y_op = np.zeros((self.net.do, self.x.shape[1]))
        y_op_lim = np.zeros((self.net.do, self.x.shape[1]))
        r_op = np.zeros((self.net.N, self.x.shape[1]))
        r_op_lim = np.zeros((self.net.N, self.x.shape[1]))
        for j in range(inps_values.shape[1]):
            x_value = inps_values[: self.net.di, j]
            I_value = inps_values[self.net.di :, j]
            xp.value = x_value
            Ip.value = I_value
            cols = np.where(
                np.all(self.x == x_value[:, np.newaxis], axis=0)
                * np.all(self.I == I_value[:, np.newaxis], axis=0)
            )[0]
            for prob in probs:
                prob.solve()

            if "y_op" in options:
                y_op[:, cols] = (
                    y_opv.value[:, np.newaxis] if y_opv.value is not None else np.nan
                )
            if "y_op_lim" in options:
                y_op_lim[:, cols] = (
                    y_opv_lim.value[:, np.newaxis]
                    if y_opv_lim.value is not None
                    else np.nan
                )
            if "r_op" in options:
                r_op[:, cols] = (
                    r_opv.value[:, np.newaxis] if r_opv.value is not None else np.nan
                )
            if "r_op_lim" in options:
                r_op_lim[:, cols] = (
                    r_opv_lim.value[:, np.newaxis]
                    if r_opv_lim.value is not None
                    else np.nan
                )

        return y_op, y_op_lim, r_op, r_op_lim

    def _optimize_cvx_ccv(self, Q, options):

        x_values = np.unique(self.x, axis=1)

        Q_norm = Q  # / Q[0, 0]
        A, S = _canon_symmetric(Q_norm)
        EAL = self.net.E @ np.linalg.pinv(A)

        signs = np.diag(S)
        min_idx = np.where(signs == 1)[0]
        max_idx = np.where(signs == -1)[0]
        mins = min_idx.shape[0]

        rncoup_idx = np.where(
            np.all(np.isclose(EAL[:, max_idx], 0, atol=1e-7), axis=1)
        )[0]
        rcoup_idx = np.where(
            ~np.all(np.isclose(EAL[:, max_idx], 0, atol=1e-7, rtol=0), axis=1)
        )[0]
        rncoups = rncoup_idx.shape[0]
        rcoups = rcoup_idx.shape[0]

        jump = np.diag(self.net.W).copy()
        C_op = self.net.T - self.I[:, 0] + jump / 2
        C_op_lim = self.net.T - self.I[:, 0]

        # Gekko limited memory
        # EAL = np.round(EAL, decimals=6)
        # C_op = np.round(C_op, decimals=6)
        # C_op_lim = np.round(C_op_lim, decimals=6)
        # F = np.round(self.net.F, decimals=6)
        # W = np.round(self.net.W, decimals=6)
        F = self.net.F
        W = self.net.W
        E = self.net.E

        probs = []
        if "y_op" in options or "y_op_lim" in options:

            def _prob_y_gen(C):
                prob = GEKKO(remote=False)

                y_min = prob.Array(prob.Var, mins)
                lamb = prob.Array(prob.Var, rcoups, lb=0.0)
                xp = prob.Array(prob.Param, self.net.di)

                for i in range(rncoups):
                    prob.Equation(
                        F[rncoup_idx[i], :] @ xp
                        + E[rncoup_idx[i], min_idx] @ y_min
                        - C[rncoup_idx[i]]
                        <= 0
                    )

                # quad_min = sum(
                #     [
                #         y_min[i] * Q[i][j] * y_min[j]
                #         for i in range(mins)
                #         for j in range(mins)
                #     ]
                # )

                quad_min = 0
                for i in range(mins):
                    for j in range(mins):
                        if Q[i][j] != 0:
                            quad_min += y_min[i] * Q[i][j] * y_min[j]

                EQET = (
                    E[rcoup_idx][:, max_idx]
                    @ np.linalg.inv(Q[max_idx][:, max_idx])
                    @ E[rcoup_idx][:, max_idx].T
                )
                EQET = np.round(EQET, decimals=6)
                # quad_max = sum(
                #     [
                #         lamb[i] * EQET[i][j] * lamb[j]
                #         for i in range(rcoups)
                #         for j in range(rcoups)
                #     ]
                # )
                for i in range(rcoups):
                    quad_max_subi = lamb[i] * EQET[i][i] * lamb[i]
                    quad_max_sub = (
                        2
                        * lamb[i]
                        * sum([EQET[i][j] * lamb[j] for j in range(i + 1, rcoups)])
                    )
                    prob.Minimize(-quad_max_subi)
                    prob.Minimize(-quad_max_sub)

                # lag_coup = sum(
                #     [
                #         lamb[i]
                #         * (
                #             F[rcoup_idx[i], :] @ xp
                #             + E[rcoup_idx[i], min_idx] @ y_min
                #             - C[rcoup_idx[i]]
                #         )
                #         for i in range(rcoups)
                #     ]
                # )
                for i in range(rcoups):
                    lag_coup_sub = lamb[i] * (
                        F[rcoup_idx[i], :] @ xp
                        + E[rcoup_idx[i], min_idx] @ y_min
                        - C[rcoup_idx[i]]
                    )
                    prob.Minimize(-2 * lag_coup_sub)

                prob.Minimize(quad_min)
                # prob.Minimize(-quad_max)
                # prob.Minimize(-2 * lag_coup)
                return {
                    "input": xp,
                    "y_min": y_min,
                    "lamb": lamb,
                    "prob": prob,
                }

            if "y_op" in options:
                # TODO Probably just have to change the definition of normD in C_op
                prob_dict = _prob_y_gen(C_op)
                prob_dict["name"] = "prob_y_op"
                probs.append(prob_dict)
            if "y_op_lim" in options:
                prob_dict = _prob_y_gen(C_op_lim)
                prob_dict["name"] = "prob_y_op_lim"
                probs.append(prob_dict)

        if "r_op" in options or "r_op_lim" in options:
            # TODO
            # raise ValueError(
            #     "Non convex optimization not implemented yet for r and r_lim"
            # )

            def _prob_r_gen(C):
                prob = GEKKO(remote=False)

                if (
                    rncoups == 0 or rcoups == 0
                ):  # all (non) coupled constraints -> min r
                    (
                        print("all coupled constraints")
                        if rncoups == 0
                        else print("all non-coupled constraints")
                    )
                    r = prob.Array(prob.Var, self.net.N)
                    xp = prob.Array(prob.FV, self.net.di)

                    for i in range(self.net.N):
                        prob.Equation(-r[i] <= 0)

                    quad = sum(
                        [
                            r[i] * W[i][j] * r[j]
                            for i in range(self.net.N)
                            for j in range(self.net.N)
                        ]
                    )
                    cost = 2 * sum(
                        [r[i] * (C[i] - F[i, :] @ xp) for i in range(self.net.N)]
                    )
                    prob.Obj(-quad + cost)
                    return {"input": xp, "r": r, "prob": prob}
                else:  # some non-coupled constraints -> min r_min (max r_max)
                    r_ncoups = prob.Array(prob.Var, rncoups, lb=0.0)
                    lamb = prob.Array(prob.Var, rncoups, lb=0.0)
                    r_coups = prob.Array(prob.Var, rcoups, lb=0.0)
                    xp = prob.Array(prob.FV, self.net.di)

                    for i in range(rncoups):
                        prob.Equation(-lamb[i] * r_ncoups[i] == 0)

                        imax = rncoup_idx[i]
                        prob.Equation(
                            -2
                            * sum(
                                [
                                    r_coups[j] * W[rcoup_idx[j], imax]
                                    for j in range(rcoups)
                                ]
                            )
                            + 2
                            * sum(
                                [
                                    r_ncoups[j] * W[rncoup_idx[j], imax]
                                    for j in range(rncoups)
                                ]
                            )
                            - 2 * (C[imax] - F[imax, :] @ xp)
                            + lamb[i]
                            == 0
                        )

                    quad_minmin = sum(
                        [
                            r_coups[i] * W[rcoup_idx[i], rcoup_idx[j]] * r_coups[j]
                            for i in range(rcoups)
                            for j in range(rcoups)
                        ]
                    )
                    quad_minmax = sum(
                        [
                            r_coups[i] * W[rcoup_idx[i], rncoup_idx[j]] * r_ncoups[j]
                            for i in range(rcoups)
                            for j in range(rncoups)
                        ]
                    )
                    quad_maxmax = sum(
                        [
                            r_ncoups[i] * W[rncoup_idx[i], rncoup_idx[j]] * r_ncoups[j]
                            for i in range(rncoups)
                            for j in range(rncoups)
                        ]
                    )
                    cost_min = 2 * sum(
                        [
                            r_coups[i] * (C[rcoup_idx[i]] - F[rcoup_idx[i], :] @ xp)
                            for i in range(rcoups)
                        ]
                    )
                    cost_max = 2 * sum(
                        [
                            r_ncoups[i] * (C[rncoup_idx[i]] - F[rncoup_idx[i], :] @ xp)
                            for i in range(rncoups)
                        ]
                    )
                    prob.Obj(
                        -quad_minmin
                        - 2 * quad_minmax
                        + quad_maxmax
                        + cost_min
                        - cost_max
                    )
                    return {
                        "input": xp,
                        "r_coup": r_coups,
                        "r_ncoup": r_ncoups,
                        "lamb": lamb,
                        "prob": prob,
                    }

            if "r_op" in options:
                # TODO Probably just have to change the definition of normD in C_op
                prob_dict = _prob_r_gen(C_op)
                prob_dict["name"] = "prob_r_op"
                probs.append(prob_dict)
            if "r_op_lim" in options:
                prob_dict = _prob_r_gen(C_op_lim)
                prob_dict["name"] = "prob_r_op_lim"
                probs.append(prob_dict)

        y_op = np.zeros((self.net.do, self.x.shape[1]))
        y_op_lim = np.zeros((self.net.do, self.x.shape[1]))
        r_op = np.zeros((self.net.N, self.x.shape[1]))
        r_op_lim = np.zeros((self.net.N, self.x.shape[1]))
        for j in range(x_values.shape[1]):
            cols = np.where(np.all(self.x == x_values[:, j : j + 1], axis=0))[0]

            for prob_dict in probs:
                for i in range(self.net.di):
                    prob_dict["input"][i].value = x_values[i, j]

                # initial guess
                if prob_dict["name"] in {"prob_y_op", "prob_y_op_lim"}:
                    if hasattr(
                        self, "y"
                    ):  # initialize guess near last y with this input
                        y_init = (
                            self.y[:, cols[-1]]
                            if not self.latent_bias
                            else self.y[:, cols[-1]] - self.bias
                        )
                    else:  # initialize at least in feasible region
                        y_init = np.linalg.lstsq(
                            self.net.E,
                            C_op + self.net.F @ x_values[:, j] - 1e-2,
                            rcond=None,
                        )[0]
                    y_min_init = y_init[min_idx]
                    y_max_init = y_init[max_idx]
                    for i in range(mins):
                        prob_dict["y_min"][i].value = y_min_init[i]
                    lamb_init, _ = nnls(
                        E[rcoup_idx][:, max_idx].T, Q[max_idx][:, max_idx] @ y_max_init
                    )
                    for i, val in enumerate(lamb_init):
                        prob_dict["lamb"][i].value = float(val)

                if prob_dict["name"] in {"prob_r_op", "prob_r_op_lim"}:
                    if hasattr(
                        self, "r"
                    ):  # initialize guess near last r with this input
                        r_init = self.r[:, cols[-1]]
                    else:  # initialize at least in feasible region
                        r_init = 1e-2 * np.ones(self.net.N)
                    if (
                        rncoups == 0 or rcoups == 0
                    ):  # all (non) coupled constraints -> min r
                        for i in range(self.net.N):
                            prob_dict["r"][i].value = r_init[i]
                    else:
                        r_coup_init = r_init[rcoup_idx]
                        r_ncoup_init = r_init[rncoup_idx]
                        for i in range(rcoups):
                            prob_dict["r_coup"][i].value = r_coup_init[i]
                        for i in range(rncoups):
                            prob_dict["r_ncoup"][i].value = r_ncoup_init[i]
                        C_temp = C_op if prob_dict["name"] == "prob_r_op" else C_op_lim
                        lamb_init = 2 * (
                            (r_coup_init @ self.net.W[rcoup_idx][:, rncoup_idx])
                            - (r_ncoup_init @ self.net.W[rncoup_idx][:, rncoup_idx])
                            + (
                                C_temp[rncoup_idx]
                                - self.net.F[rncoup_idx, :] @ x_values[:, j]
                            )
                        )
                        for i, val in enumerate(lamb_init):
                            prob_dict["lamb"][i].value = float(val)

                prob = prob_dict["prob"]
                fail = False
                prob.options.SOLVER = 1
                prob.options.MAX_ITER = 1000
                try:
                    prob.solve(disp=True)
                except BaseException:
                    fail = True

                if prob_dict["name"] == "prob_y_op":
                    if not fail:
                        y_min = np.array(
                            [prob_dict["y_min"][i].value for i in range(mins)]
                        )
                        lamb = np.array(
                            [prob_dict["lamb"][i].value for i in range(rcoups)]
                        )
                        y_max = np.linalg.inv(Q[max_idx][:, max_idx]) @ (
                            E[rcoup_idx][:, max_idx].T @ lamb
                        )
                        y_op_val = np.vstack((y_min, y_max))
                        y_op[:, cols] = y_op_val
                    else:
                        y_op[:, cols] = np.nan
                if prob_dict["name"] == "prob_y_op_lim":
                    if not fail:
                        y_min = np.array(
                            [prob_dict["y_min"][i].value for i in range(mins)]
                        )
                        lamb = np.array(
                            [prob_dict["lamb"][i].value for i in range(rcoups)]
                        )
                        y_max = np.linalg.inv(Q[max_idx][:, max_idx]) @ (
                            E[rcoup_idx][:, max_idx].T @ lamb
                        )
                        y_op_lim_val = np.vstack((y_min, y_max))
                        y_op_lim[:, cols] = y_op_lim_val
                    else:
                        y_op_lim[:, cols] = np.nan
                if prob_dict["name"] == "prob_r_op":
                    if not fail:
                        if rncoups == 0 or rcoups == 0:
                            r_op[:, cols] = np.array(
                                [prob_dict["r"][i].value for i in range(self.net.N)]
                            )
                        else:
                            r_coup = np.array(
                                [prob_dict["r_coup"][i].value for i in range(rcoups)]
                            )
                            r_ncoup = np.array(
                                [prob_dict["r_ncoup"][i].value for i in range(rncoups)]
                            )

                            r_op[np.ix_(rcoup_idx, cols)] = r_coup
                            r_op[np.ix_(rncoup_idx, cols)] = r_ncoup
                    else:
                        r_op[:, cols] = np.nan

                if prob_dict["name"] == "prob_r_op_lim":
                    if not fail:
                        if rncoups == 0 or rcoups == 0:
                            r_op_lim[:, cols] = np.array(
                                [prob_dict["r"][i].value for i in range(self.net.N)]
                            )
                        else:
                            r_coup = np.array(
                                [prob_dict["r_coup"][i].value for i in range(rcoups)]
                            )
                            r_ncoup = np.array(
                                [prob_dict["r_ncoup"][i].value for i in range(rncoups)]
                            )

                            r_op_lim[np.ix_(rcoup_idx, cols)] = r_coup
                            r_op_lim[np.ix_(rncoup_idx, cols)] = r_ncoup
                    else:
                        r_op_lim[:, cols] = np.nan

        return y_op, y_op_lim, r_op, r_op_lim

    # PLOTTING ####

    def plot(
        self,
        geometry: bool = True,
        centergeom: np.ndarray | None = None,
        rate_space: bool = True,
        vol_space: bool = False,
        save: bool = True,
    ) -> tuple[matplotlib.figure.Figure, list, list]:
        """
        Plot the results of the simulation.

        Parameters
        ----------
        geometry : bool, default=True
            If False, do not plot the geometry of the network.

        centergeom : np.ndarray | None, default=None
            Center of the geometry for the plot. If None, it is estimated automatically.

        rate_space : bool, default=True
            If False, do not plot the rate space of the network.

        vol_space : bool, default=False
            If False, do not plot the voltage space of the network.

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

        geometry = geometry and self.net.do in {2, 3}
        rate_space = rate_space and self.net.N in {2, 3}
        vol_space = vol_space and self.net.N in {2, 3}
        assert centergeom is None or centergeom.shape == (
            self.net.do,
        ), "centergeom should be of shape (do,) or None"

        geom_plots = geometry + rate_space + vol_space
        if geom_plots == 3:
            gs = gridspec.GridSpec(4, 4)
            ax1 = plt.subplot(gs[0, 3])
            ax2 = plt.subplot(gs[1, 3])
            ax3 = plt.subplot(gs[2, 3])
            ax4 = plt.subplot(gs[3, 3])
            ax5 = (
                plt.subplot(gs[:, 0])
                if self.net.do == 2
                else plt.subplot(gs[:, 0], projection="3d")
            )
            ax6 = (
                plt.subplot(gs[:, 1])
                if self.net.N == 2
                else plt.subplot(gs[:, 1], projection="3d")
            )
            ax7 = (
                plt.subplot(gs[:, 2])
                if self.net.N == 2
                else plt.subplot(gs[:, 2], projection="3d")
            )
            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        elif geom_plots == 2:
            gs = gridspec.GridSpec(4, 3)
            ax1 = plt.subplot(gs[0, 2])
            ax2 = plt.subplot(gs[1, 2])
            ax3 = plt.subplot(gs[2, 2])
            ax4 = plt.subplot(gs[3, 2])
            if geometry:
                ax5 = (
                    plt.subplot(gs[:, 0])
                    if self.net.do == 2
                    else plt.subplot(gs[:, 0], projection="3d")
                )
            else:
                ax5 = (
                    plt.subplot(gs[:, 0])
                    if self.net.N == 2
                    else plt.subplot(gs[:, 0], projection="3d")
                )
            ax6 = (
                plt.subplot(gs[:, 1])
                if self.net.N == 2
                else plt.subplot(gs[:, 1], projection="3d")
            )
            axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        elif geom_plots == 1:
            gs = gridspec.GridSpec(4, 2)
            ax1 = plt.subplot(gs[0, 1])
            ax2 = plt.subplot(gs[1, 1])
            ax3 = plt.subplot(gs[2, 1])
            ax4 = plt.subplot(gs[3, 1])
            if geometry:
                ax5 = (
                    plt.subplot(gs[:, 0])
                    if self.net.do == 2
                    else plt.subplot(gs[:, 0], projection="3d")
                )
            else:
                ax5 = (
                    plt.subplot(gs[:, 0])
                    if self.net.N == 2
                    else plt.subplot(gs[:, 0], projection="3d")
                )
            axes = [ax1, ax2, ax3, ax4, ax5]
        else:
            gs = gridspec.GridSpec(4, 1)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[1, 0])
            ax3 = plt.subplot(gs[2, 0])
            ax4 = plt.subplot(gs[3, 0])
            ax5 = None
            axes = [ax1, ax2, ax3, ax4]

        _, _, artists_io = self.plot_io(ax=ax1, t=self.Tmax, save=False)
        ax1.set_xlabel("")
        _, _, artists_spikes = self.plot_spikes(ax=ax2, t=self.Tmax, save=False)
        ax2.set_xlabel("")
        _, _, artists_rates = self.plot_rates(ax=ax3, t=self.Tmax, save=False)
        ax3.set_xlabel("")
        _, _, artists_voltages = self.plot_voltages(ax=ax4, t=self.Tmax, save=False)
        artists = [artists_io, artists_spikes, artists_rates, artists_voltages]
        if geometry:
            y_op = self.y_op[:, -1:] if hasattr(self, "y_op") else None
            y_op_lim = self.y_op_lim[:, -1:] if hasattr(self, "y_op_lim") else None
            _, _, artists_net = self.net.plot(
                ax=ax5,
                x=self.x,
                y=self.y,
                I=self.I,
                y_op=y_op,
                y_op_lim=y_op_lim,
                latent_bias=self.latent_bias,
                centered=centergeom,
                save=False,
            )
            artists.append(artists_net)
        if vol_space:
            _, _, artists_net = self.net.plot_vol_space(
                x=self.x,
                I=self.I,
                ax=axes[-1] if not rate_space else axes[-2],
                V=self.V,
                voltage_bias=self.voltage_bias,
                save=False,
            )
            artists.append(artists_net)
        if rate_space:
            r_op = self.r_op[:, -1:] if hasattr(self, "r_op") else None
            r_op_lim = self.r_op_lim[:, -1:] if hasattr(self, "r_op_lim") else None
            _, _, artists_net = self.net.plot_rate_space(
                x=self.x,
                I=self.I,
                ax=axes[-1],
                r=self.r,
                r_op=r_op,
                r_op_lim=r_op_lim,
                save=False,
            )
            artists.append(artists_net)

        plt.tight_layout()
        if save:
            _save_fig(
                fig,
                self.time_stamp
                + "-"
                + self.stag
                + "-"
                + self.rtag
                + "-"
                + self.otag
                + "-plot.svg",
            )

        return fig, axes, artists

    def plot_io(
        self, ax: matplotlib.axes.Axes | None = None, t: float = -1, save: bool = True
    ) -> tuple[
        matplotlib.figure.Figure | matplotlib.figure.SubFigure,
        matplotlib.axes.Axes,
        list,
    ]:
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
        fig: matplotlib.figure.Figure or matplotlib.figure.SubFigure
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

        y_op, y_op_lim, _, _ = self._crop(t, "op")
        if hasattr(self, "y_op"):
            liney_op_arr = []
            for i in range(self.net.do):
                liney_op = ax.plot(
                    xaxis,
                    y_op[i, :],
                    color=colorsio[i],
                    linestyle=":",
                    label="y_op" if i == 0 else "",
                    alpha=0.5,
                )[0]
                liney_op_arr.append(liney_op)
            artists.append(liney_op_arr)

        if hasattr(self, "y_op_lim"):
            liney_op_lim_arr = []
            for i in range(self.net.do):
                liney_op_lim = ax.plot(
                    xaxis,
                    y_op_lim[i, :],
                    color=colorsio[i],
                    linestyle="--",
                    label="y_op_lim" if i == 0 else "",
                    alpha=0.5,
                )[0]
                liney_op_lim_arr.append(liney_op_lim)
            artists.append(liney_op_lim_arr)

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
            assert type(fig) is matplotlib.figure.Figure
            _save_fig(
                fig,
                self.time_stamp
                + "-"
                + self.stag
                + "-"
                + self.rtag
                + "-"
                + self.otag
                + "-ioplot.svg",
            )

        return fig, ax, artists

    def plot_spikes(
        self, ax: matplotlib.axes.Axes | None = None, t: float = -1, save: bool = True
    ) -> tuple[
        matplotlib.figure.Figure | matplotlib.figure.SubFigure,
        matplotlib.axes.Axes,
        list,
    ]:
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
        fig: matplotlib.figure.Figure or matplotlib.figure.SubFigure
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
            assert type(fig) is matplotlib.figure.Figure
            _save_fig(
                fig,
                self.time_stamp
                + "-"
                + self.stag
                + "-"
                + self.rtag
                + "-"
                + self.otag
                + "-spikesplot.svg",
            )

        return fig, ax, artists

    def plot_rates(
        self, ax: matplotlib.axes.Axes | None = None, t: float = -1, save: bool = True
    ) -> tuple[
        matplotlib.figure.Figure | matplotlib.figure.SubFigure,
        matplotlib.axes.Axes,
        list,
    ]:
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
        fig: matplotlib.figure.Figure or matplotlib.figure.SubFigure
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

        liner = []
        for i in range(self.net.N):
            label = ""
            if i < 10:
                if self.net.N < 50:
                    for j in np.arange(i, self.net.N, 10):
                        label += f"r{j + 1},"
                else:
                    label += f"r_{i + 1}"
            line = ax.plot(xaxis, r[i, :], color=colors[i], label=label)[0]
            liner.append(line)
        artists.append(liner)

        _, _, r_op, r_op_lim = self._crop(t, "op")
        if hasattr(self, "r_op"):
            liner_op_arr = []
            for i in range(self.net.N):
                liner_op = ax.plot(
                    xaxis,
                    r_op[i, :],
                    color=colors[i],
                    linestyle=":",
                    label="r_op" if i == 0 else "",
                    alpha=0.5,
                )[0]
                liner_op_arr.append(liner_op)
            artists.append(liner_op_arr)

        if hasattr(self, "r_op_lim"):
            liner_op_lim_arr = []
            for i in range(self.net.N):
                liner_op_lim = ax.plot(
                    xaxis,
                    r_op_lim[i, :],
                    color=colors[i],
                    linestyle="--",
                    label="r_op_lim" if i == 0 else "",
                    alpha=0.5,
                )[0]
                liner_op_lim_arr.append(liner_op_lim)
            artists.append(liner_op_lim_arr)

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
            assert type(fig) is matplotlib.figure.Figure
            _save_fig(
                fig,
                self.time_stamp
                + "-"
                + self.stag
                + "-"
                + self.rtag
                + "-"
                + self.otag
                + "-ratesplot.svg",
            )

        return fig, ax, artists

    def plot_voltages(
        self, ax: matplotlib.axes.Axes | None = None, t: float = -1, save: bool = True
    ) -> tuple[
        matplotlib.figure.Figure | matplotlib.figure.SubFigure,
        matplotlib.axes.Axes,
        list,
    ]:
        """
        Plot the voltages of the network as a function of time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes to plot the voltages. If None, a new figure is created.

        t : float, default=-1
            Time to crop the voltages. If -1 the whole simulation is plotted.

        save : bool, default=True
            If True, save the figure.

        Returns
        -------
        fig: matplotlib.figure.Figure or matplotlib.figure.SubFigure
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

        (V,) = self._crop(t, "voltages")

        (Tv,) = self._crop(t, "voltage_thresh")

        artists = []
        xaxis = np.linspace(0, V.shape[1] * self.dt, V.shape[1])
        colors = _get_colors(self.net.N, self.net.W)

        lineV = []
        for i in range(self.net.N):
            label = ""
            if i < 10:
                if self.net.N < 50:
                    for j in np.arange(i, self.net.N, 10):
                        label += f"V{j + 1},"
                else:
                    label += f"V_{i + 1}"
            line = ax.plot(xaxis, V[i, :], color=colors[i], label=label)[0]
            thresh = ax.plot(
                xaxis,
                Tv[i, :],
                color=colors[i],
                linestyle="--",
                alpha=0.5,
            )[0]
            lineV.append([line, thresh])
        artists.append(lineV)

        ax.set_ylabel("V(t)")
        ax.set_xlabel("time (s)")
        ax.set_xlim(-0.5, self.Tmax + 0.5)
        ax.set_ylim(np.min(self.V) - 0.05, np.max(self.V) + 0.05)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig = ax.get_figure()
        assert fig is not None
        if save:
            assert type(fig) is matplotlib.figure.Figure
            _save_fig(
                fig,
                self.time_stamp
                + "-"
                + self.stag
                + "-"
                + self.rtag
                + "-"
                + self.otag
                + "-voltagesplot.svg",
            )

        return fig, ax, artists

    # ANIMATION ####

    def animate(
        self,
        geometry: bool = True,
        rate_space: bool = True,
        vol_space: bool = False,
    ) -> None:
        """
        Animate the results of a simulation.

        Parameters
        ----------
        geometry : bool, default=True
            If False, do not plot the geometry of the network.

        rate_space : bool, default=True
            If False, do not plot the rate space of the network.

        vol_space : bool, default=False
            If False, do not plot the voltage space of the network.

        """

        fig = plt.figure(figsize=(20, 10))

        geometry = geometry and self.net.do in {2, 3}
        rate_space = rate_space and self.net.N in {2, 3}
        vol_space = vol_space and self.net.N in {2, 3}

        geom_plots = geometry + rate_space + vol_space
        if geom_plots == 3:
            gs = gridspec.GridSpec(4, 4)
            ax1 = plt.subplot(gs[0, 3])
            ax2 = plt.subplot(gs[1, 3])
            ax3 = plt.subplot(gs[2, 3])
            ax4 = plt.subplot(gs[3, 3])
            ax5 = (
                plt.subplot(gs[:, 0])
                if self.net.do == 2
                else plt.subplot(gs[:, 0], projection="3d")
            )
            ax6 = (
                plt.subplot(gs[:, 1])
                if self.net.N == 2
                else plt.subplot(gs[:, 1], projection="3d")
            )
            ax7 = (
                plt.subplot(gs[:, 2])
                if self.net.N == 2
                else plt.subplot(gs[:, 2], projection="3d")
            )
            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        elif geom_plots == 2:
            gs = gridspec.GridSpec(4, 3)
            ax1 = plt.subplot(gs[0, 2])
            ax2 = plt.subplot(gs[1, 2])
            ax3 = plt.subplot(gs[2, 2])
            ax4 = plt.subplot(gs[3, 2])
            if geometry:
                ax5 = (
                    plt.subplot(gs[:, 0])
                    if self.net.do == 2
                    else plt.subplot(gs[:, 0], projection="3d")
                )
            else:
                ax5 = (
                    plt.subplot(gs[:, 0])
                    if self.net.N == 2
                    else plt.subplot(gs[:, 0], projection="3d")
                )
            ax6 = (
                plt.subplot(gs[:, 1])
                if self.net.N == 2
                else plt.subplot(gs[:, 1], projection="3d")
            )
            axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        elif geom_plots == 1:
            gs = gridspec.GridSpec(4, 2)
            ax1 = plt.subplot(gs[0, 1])
            ax2 = plt.subplot(gs[1, 1])
            ax3 = plt.subplot(gs[2, 1])
            ax4 = plt.subplot(gs[3, 1])
            if geometry:
                ax5 = (
                    plt.subplot(gs[:, 0])
                    if self.net.do == 2
                    else plt.subplot(gs[:, 0], projection="3d")
                )
            else:
                ax5 = (
                    plt.subplot(gs[:, 0])
                    if self.net.N == 2
                    else plt.subplot(gs[:, 0], projection="3d")
                )
            axes = [ax1, ax2, ax3, ax4, ax5]
        else:
            gs = gridspec.GridSpec(4, 1)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[1, 0])
            ax3 = plt.subplot(gs[2, 0])
            ax4 = plt.subplot(gs[3, 0])
            ax5 = None
            axes = [ax1, ax2, ax3, ax4]

        artists = []
        _, _, artists_io = self.plot_io(ax=ax1, t=0, save=False)
        ax1.set_xlabel("")
        _, _, artists_spikes = self.plot_spikes(ax=ax2, t=0, save=False)
        ax2.set_xlabel("")
        _, _, artists_rates = self.plot_rates(ax=ax3, t=0, save=False)
        ax3.set_xlabel("")
        _, _, artists_voltages = self.plot_voltages(ax=ax4, t=0, save=False)
        artists = [artists_io, artists_spikes, artists_rates, artists_voltages]

        x, y, I, r, V, y_op, y_op_lim, r_op, r_op_lim = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if geometry or rate_space or vol_space:
            x, y = self._crop(t=0, type="io")
            (I,) = self._crop(t=0, type="inp_curr")
            (r,) = self._crop(t=0, type="rates")
            (V,) = self._crop(t=0, type="voltages")
            y_op, y_op_lim, r_op, r_op_lim = self._crop(t=0, type="op")
        if geometry:
            y_op = self.y_op[:, :1] if hasattr(self, "y_op") else None
            y_op_lim = self.y_op_lim[:, :1] if hasattr(self, "y_op_lim") else None
            _, _, artists_net = self.net.plot(
                ax=ax5,
                x=x,
                y=y,
                I=I,
                y_op=y_op,
                y_op_lim=y_op_lim,
                latent_bias=self.latent_bias,
                save=False,
            )
            artists.append(artists_net)
        if vol_space:
            _, _, artists_net = self.net.plot_vol_space(
                x=x,
                I=I,
                ax=axes[-1] if not rate_space else axes[-2],
                V=V,
                voltage_bias=self.voltage_bias,
                save=False,
            )
            artists.append(artists_net)
        if rate_space:
            r_op = self.r_op[:, :1] if hasattr(self, "r_op") else None
            r_op_lim = self.r_op_lim[:, :1] if hasattr(self, "r_op_lim") else None
            _, _, artists_net = self.net.plot_rate_space(
                x=x,
                I=I,
                ax=axes[-1],
                r=r,
                r_op=r_op,
                r_op_lim=r_op_lim,
                save=False,
            )
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
            self._animate_voltages(artists=artists[3], t=t)

            if geometry or rate_space:

                (r,) = self._crop(t, "rates")
                (V,) = self._crop(t, "voltages")

                newspiked = _neurons_spiked_between(self.stimes, tpast, t)
                oldspiked = _neurons_spiked_between(self.stimes, tpastpast, tpast)
                spiking = np.concatenate(
                    [
                        -(np.array(oldspiked, dtype=int) + 1),
                        np.array(newspiked, dtype=int) + 1,
                    ]
                )

                x, y = self._crop(t, "io")
                input_change = (
                    not np.array_equal(
                        x[:, int(t / self.dt)], x[:, int(tpast / self.dt)]
                    )
                    if tpast >= 0
                    else False
                )

                (I,) = self._crop(t, "inp_curr")
                current_change = (
                    not np.array_equal(
                        I[:, int(t / self.dt)], I[:, int(tpast / self.dt)]
                    )
                    if tpast >= 0
                    else False
                )

                assert ax5 is not None
                if geometry:
                    y_op, y_op_lim, _, _ = self._crop(t, "op")
                    self.net._animate(
                        ax=ax5,
                        artists=artists[4],
                        x=x,
                        I=I,
                        y=y,
                        y_op=y_op,
                        y_op_lim=y_op_lim,
                        latent_bias=self.latent_bias,
                        input_change=input_change,
                        current_change=current_change,
                        spiking=spiking,
                    )
                if vol_space:
                    self.net._animate_vol_space(
                        ax=axes[-1] if not rate_space else axes[-2],
                        artists=artists[-1] if not rate_space else artists[-2],
                        x=x,
                        I=I,
                        V=V,
                        input_change=input_change,
                        current_change=current_change,
                        spiking=spiking,
                        voltage_bias=self.voltage_bias,
                    )
                if rate_space:
                    _, _, r_op, r_op_lim = self._crop(t, "op")
                    self.net._animate_rate_space(
                        ax=axes[-1],
                        artists=artists[-1],
                        x=x,
                        I=I,
                        r=r,
                        r_op=r_op,
                        r_op_lim=r_op_lim,
                        input_change=input_change,
                        current_change=current_change,
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

        _save_ani(
            ani,
            self.time_stamp
            + "-"
            + self.stag
            + "-"
            + self.rtag
            + "-"
            + self.otag
            + "-animation.gif",
            anim_freq,
        )

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

        y_op, y_op_lim, _, _ = self._crop(t, "op")
        k = 2
        if hasattr(self, "y_op"):
            for i in range(self.net.do):
                artists[k][i].set_xdata(xaxis)
                artists[k][i].set_ydata(y_op[i, :])
            k += 1

        if hasattr(self, "y_op_lim"):
            for i in range(self.net.do):
                artists[k][i].set_xdata(xaxis)
                artists[k][i].set_ydata(y_op_lim[i, :])

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
            artists[0][i].set_xdata(xaxis)
            artists[0][i].set_ydata(r[i, :])

        _, _, r_op, r_op_lim = self._crop(t, "op")
        k = 1
        if hasattr(self, "r_op"):
            for i in range(self.net.N):
                artists[k][i].set_xdata(xaxis)
                artists[k][i].set_ydata(r_op[i, :])
            k += 1

        if hasattr(self, "r_op_lim"):
            for i in range(self.net.N):
                artists[k][i].set_xdata(xaxis)
                artists[k][i].set_ydata(r_op_lim[i, :])

    def _animate_voltages(
        self,
        artists: list,
        t: float,
    ) -> None:
        """
        Animate the voltages of the network as a function of time.
        Modifies the artists for the frame at time t

        Parameters
        ----------
        artists : list
            Artists of the plot.

        t : float
            Time to crop the voltages. For the frames of the animation.

        """

        (V,) = self._crop(t, "voltages")
        (Tv,) = self._crop(t, "voltage_thresh")

        xaxis = np.linspace(0, V.shape[1] * self.dt, V.shape[1])
        for i in range(self.net.N):
            artists[0][i][0].set_xdata(xaxis)
            artists[0][i][0].set_ydata(V[i, :])
            artists[0][i][1].set_xdata(xaxis)
            artists[0][i][1].set_ydata(Tv[i, :])

    def save(self, dir: str = "./data/", name: str = "") -> tuple[str, str]:
        """
        Save the results of the simulation to pickle file.

        Parameters
        ----------
        dir : str, default='./data/'
            Directory to save the results.

        name : str, default=''
            Name of the saved file. If empty, a name is generated based on the
            simulation tags.

        Returns
        -------
        path : str
            Path to the saved file.

        name : str
            Name of the saved file.
        """

        os.makedirs(dir, exist_ok=True)
        name = (
            self.time_stamp + "-" + self.stag + "-" + self.rtag + "-" + self.otag
            if name == ""
            else name
        )
        path = dir + name + "-results.pkl"
        with open(
            path,
            "wb",
        ) as f:
            pickle.dump(self.__dict__, f)

        return path, name

    def load(self, dir: str = "./data/", name: str = "") -> None:
        """
        Load the results of the simulation from pickle file.

        Parameters
        ----------
        dir : str, default='./data/'
            Directory to load the results from.

        name : str, default=''
            Name of the file to load. If empty, the name used is based on the
            simulation tags.
        """
        name = (
            self.time_stamp + "-" + self.stag + "-" + self.rtag + "-" + self.otag
            if name == ""
            else name
        )
        path = dir + name + "-results.pkl"
        with open(
            path,
            "rb",
        ) as f:
            tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)

    def _crop(self, t: float = -1, type: str = "io") -> tuple[np.ndarray, ...]:
        """
        Crop the results of the simulation to time t. For animation purposes.

        Parameters
        ----------
        t : float, default=-1
            Time to crop the results. If -1 the whole simulation is returned.

        type : str, default='io'
            Type of results to crop: 'io' (x,y), 'inp_curr' (I), 'stimes' (stimes), 'voltage_thresh' (Tv) or 'rates' (r).

        Returns
        -------
        x, y : np.ndarray of (di, t/dt), np.ndarray (do, t/dt)
        I: np.ndarray of (N, t/dt)
        stimes: np.ndarray of (#spikes at time < t, 2)
        Tv: np.ndarray of (N, t/dt)
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
            case "inp_curr":
                I = self.I[:, : time_step + 1]
                return (I,)
            case "stimes":
                stimes = self.stimes[np.where(self.stimes[:, 1] <= t)]
                return (stimes,)
            case "rates":
                r = self.r[:, : time_step + 1]
                return (r,)
            case "voltages":
                V = self.V[:, : time_step + 1]
                return (V,)
            case "voltage_thresh":
                Tv = self.Tv[:, : time_step + 1]
                return (Tv,)
            case "op":
                y_op = self.y_op[:, : time_step + 1] if hasattr(self, "y_op") else None
                y_op_lim = (
                    self.y_op_lim[:, : time_step + 1]
                    if hasattr(self, "y_op_lim")
                    else None
                )
                r_op = self.r_op[:, : time_step + 1] if hasattr(self, "r_op") else None
                r_op_lim = (
                    self.r_op_lim[:, : time_step + 1]
                    if hasattr(self, "r_op_lim")
                    else None
                )
                return y_op, y_op_lim, r_op, r_op_lim  # type: ignore
            case _:
                raise ValueError(
                    "type should be 'io', 'inp_curr', 'stimes', 'op', 'voltages' or 'rates'"
                )
