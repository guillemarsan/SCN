import numpy as np


def _stimes_from_s(s: np.ndarray, dt: float | None = None) -> np.ndarray:
    """
    Transforms spike raster to spike times

    s = bool matrix (Nxtime_steps) with 1 when the neuron spiked
    stimes = float /int matrix (#spikesx2) with the first column being the
    neuron index and the second column the spike time /time step

    Parameters
    ----------
    s: np.ndarray of bool(N, time_steps)
        Spike raster or spike times

    dt: float or None, default=None
        Time step of the simulation. If None, return the spike times in time steps

    Returns
    -------
    stimes: np.ndarray of float/int(#spikes, 2)
        Transformed spike times
    """
    # asserts
    assert s.dtype == bool, "s should be a boolean ndarray"

    # s -> stimes
    stimes = np.array(np.where(s), dtype=float).T
    if dt is not None:
        stimes[:, 1] = stimes[:, 1] * dt
    return stimes


def _s_from_stimes(
    stimes: np.ndarray, N: int = 10, time_steps: int = 10000, dt: float | None = None
) -> np.ndarray:
    """
    Transforms spike times to spike raster

    s = bool matrix (Nxtime_steps) with 1 when the neuron spiked
    stimes = float /int matrix (#spikesx2) with the first row being the
    neuron index and the second row the spike time /time step

    Parameters
    ----------
    stimes: np.ndarray of float/int(#spikes,2)
        Spike spike times

    N: int, default=10
        Number of neurons / first dimension of the returned spike raster

    time_steps: int, default=10000
        Number of time steps of the simulation / second dimension of the returned spike raster

    dt: float or None, default=None
        Time step of the simulation. If None, stimes are assumed to be in time steps units

    Returns
    -------
    s: np.ndarray of bool(N, time_steps)
        Transformed spike raster
    """

    s = np.zeros((N, time_steps), dtype=bool)

    # stimes -> s
    idxc = stimes[:, 1] if dt is not None else int(stimes[:, 1] / dt)
    s[stimes[:, 0], idxc] = 1
    return s


def _deintegrate(x: np.ndarray, lamb: float = 1, dt: float = 0.001) -> np.ndarray:
    """
    Reverse the integration of a signal with a given time step: from x to c

    Equation: c = dx + lambda x

    Parameters
    ----------
    x: np.ndarray of float(N, time_steps)
        Signal to deintegrate

    lamb: float, default=1
        Leak timescale of the integration

    dt: float
        Time step of the integration

    Returns
    -------
    c: np.ndarray of float(N, time_steps)
        Deintegrated signal
    """
    diff_x = np.diff(x)
    diff_x = np.hstack([diff_x, diff_x[:, -1:]])
    return lamb * x + diff_x / dt


def _integrate(c: np.ndarray, lamb: float = 1, dt: float = 0.001) -> np.ndarray:
    """
    Integrate a signal with a given time step: from c to x

    Equation: dx = lambda x + c

    Parameters
    ----------
    c: np.ndarray of float(N, time_steps)
        Signal to integrate

    lamb: float, default=1
        Leak timescale of the integration

    dt: float
        Time step of the simulation

    Returns
    -------
    x: np.ndarray of float(N, time_steps)
        Integrated signal
    """
    x = np.zeros_like(c)
    for i in range(1, c.shape[1]):
        x[:, i] = x[:, i - 1] + dt * (-lamb * x[:, i - 1] + c[:, i - 1])

    return x
