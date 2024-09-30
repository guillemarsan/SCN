import numpy as np


def angle_encode(x, freq=1, offset=0):
    """
    Encode a signal into a circular space.

    Parameters
    ----------
    x: np.ndarray of float(di, time_steps)
        Signal to encode.

    freq: float, default=1
        Frequency of the encoding.

    offset: float, default=0
        Offset of the encoding.

    Returns
    -------
    angx: np.ndarray of float(2*di, time_steps)
        Encoded signal.
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]

    angx = np.zeros((2 * x.shape[0], x.shape[1]))

    for d in range(x.shape[0]):
        angx[2 * d, :] = np.cos(freq * x[d, :] + offset)
        angx[2 * d + 1, :] = np.sin(freq * x[d, :] + offset)

    return angx
