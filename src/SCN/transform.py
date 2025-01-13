import numpy as np


def angle_encode(x: np.ndarray, freq: float = 1, offset: float = 0, dim: int = 2):
    """
    Encode a signal into a circular space.

    Parameters
    ----------
    x: np.ndarray of float(di, time_steps)
        Signal to encode.

    freq: float, default=1
        Frequency of the encoding. freq radians / 1 x unit.

    offset: float, default=0
        Offset of the encoding.

    dim: int, default=2
        Dimension of the encoding >= 2. 2 = circle.

    Returns
    -------
    angx: np.ndarray of float(2*di, time_steps)
        Encoded signal.
    """

    assert dim >= 2, "Dimension of the encoding must be >= 2."

    if x.ndim == 1:
        x = x[np.newaxis, :]

    angx = np.zeros((dim * x.shape[0], x.shape[1]))

    for d in range(x.shape[0]):
        for j in range(dim):
            angx[dim * d + j, :] = (
                np.cos(freq * x[d, :] + offset)
                if j % 2 == 0
                else np.sin(freq * x[d, :] + offset)
            )

    return angx


def angle_decode(angx: np.ndarray, freq: float = 1, offset: float = 0):
    """
    Decode a signal from a circular space.

    Parameters
    ----------
    angx: np.ndarray of float(2*di, time_steps)
        Signal to decode.

    freq: float, default=1
        Frequency of the encoding.

    offset: float, default=0
        Offset of the encoding.

    Returns
    -------
    x: np.ndarray of float(di, time_steps)
        Decoded signal.
    """
    if angx.ndim == 1:
        angx = angx[np.newaxis, :]

    x = np.zeros((angx.shape[0] // 2, angx.shape[1]))

    for d in range(x.shape[0]):
        x[d, :] = np.arctan2(angx[2 * d + 1, :], angx[2 * d, :]) / freq - offset

    return x
