import numpy as np


def _sphere_random(d: int = 2, N: int = 10, seed: int | None = None) -> np.ndarray:
    r"""
    Generate a matrix :math:`d \times N` with `N` unitary vectors in `d`-dim. space.

    Parameters
    ----------
    d : int, default=2
        Dimension of the space.

    N : int, default=10
        Number of vectors.

    seed : int or None, default=None
        Seed for the random number generator.

    Returns
    -------
    M: np.ndarray of float(d, N)
        Matrix with :math:`N` unitary vectors in :math:`d`-dim. space.
    """

    if seed is not None:
        np.random.seed(seed)

    # random parameters
    M = np.random.randn(d, N)
    M = M / np.linalg.norm(M, axis=0)

    return M


def _cube(d: int = 2, one_quadrant: bool = False) -> np.ndarray:
    r"""
    Generate a matrix :math:`d \times N` with `N` unitary vectors forming a hyper-cube in `d`-dim. space.
    If `one_quadrant` is True, the vectors are in the first quadrant (the axis).
    `N = 2d` for `one_quadrant = False` and `N = d` for `one_quadrant = True`.

    Parameters
    ----------
    d : int, default=2
        Dimension of the space.

    one_quadrant : bool, default=False
            If True, the weights are in the first quadrant.

    Returns
    -------
    M: np.ndarray of float(d, N)
        Matrix with :math:`N` unitary vectors in :math:`d`-dim. space.
    """

    # hypercube parameters
    M = np.eye(d)
    if not one_quadrant:
        M = np.hstack([M, -M])

    return M


def _2D_circle_random(
    N: int = 10, angle_range: list | None = None, seed: float | None = None
) -> np.ndarray:
    r"""
    Generate a matrix :math:`2 \times N` with `N` unitary vectors randomly spaced forming a circle in 2D.

    :math:`N` vectors spaced randomly between `angle_range[0]` and `angle_range[1]`.
    The vectors are :math:`\mathbf{D}_i = (-\cos(\alpha_i), -\sin(\alpha_i))`.

    Parameters
    ----------
    N: int, default=10
            Number of vectors.

    angle_range : list, default=None
        Range of angles for the vectors. If None, the range is :math:`[0, 2 \pi]`.

    seed : int or None, default=None
            Seed for the random number generator.

    Returns
    -------
    M: np.ndarray of float(2, N)
        Matrix with :math:`N` unitary vectors in 2D space.
    """

    if seed is not None:
        np.random.seed(seed)

    if angle_range is None:
        angle_range = [0, 2 * np.pi]

    # randomly spaced circular parameters
    alphas = np.random.uniform(angle_range[0], angle_range[1], N)
    M = np.array([-np.cos(alphas), -np.sin(alphas)])

    return M


def _2D_circle_spaced(N: int = 10, angle_range: list | None = None) -> np.ndarray:
    r"""
    Generate a matrix :math:`2 \times N` with `N` unitary vectors evenly spaced forming a circle in 2D.

    :math:`N` vectors spaced evenly between `angle_range[0]` and `angle_range[1]`.
    The vectors are :math:`\mathbf{D}_i = (-\cos(\alpha_i), -\sin(\alpha_i))`.

    Parameters
    ----------
    N: int, default=10
            Number of vectors.

    angle_range : list, default=None
        Range of angles for the vectors. If None, the range is :math:`[0, 2 \pi]`.

    Returns
    -------
    M: np.ndarray of float(2, N)
        Matrix with :math:`N` unitary vectors in 2D space.
    """

    if angle_range is None:
        angle_range = [0, 2 * np.pi]

    # evenly spaced circular parameters
    alphas = np.linspace(
        angle_range[0],
        angle_range[1],
        N,
        endpoint=(angle_range[1] - angle_range[0]) % (2 * np.pi) != 0,
    )
    M = np.array([-np.cos(alphas), -np.sin(alphas)])

    return M
