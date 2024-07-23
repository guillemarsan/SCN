import cvxpy as cp
import numpy as np


def _sphere_random(d: int = 2, N: int = 10, seed: int | None = None) -> np.ndarray:
    r"""
    Generate a matrix :math:`N \times d` with `N` unitary vectors in `d`-dim. space.

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
    M: np.ndarray of float(N, d)
        Matrix with :math:`N` unitary vectors in :math:`d`-dim. space.
    """

    if seed is not None:
        np.random.seed(seed)

    # random parameters
    M = np.random.randn(N, d)
    M = M / np.linalg.norm(M, axis=1)[:, np.newaxis]

    return M


def _cube(d: int = 2, one_quadrant: bool = False) -> np.ndarray:
    r"""
    Generate a matrix :math:`N \times d` with `N` unitary vectors forming a hyper-cube in `d`-dim. space.
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
    M: np.ndarray of float(N, d)
        Matrix with :math:`N` unitary vectors in :math:`d`-dim. space.
    """

    # hypercube parameters
    M = -np.eye(d)
    if not one_quadrant:
        M = np.vstack([M, -M])

    return M


def _2D_circle_random(
    N: int = 10, angle_range: list | None = None, seed: float | None = None
) -> np.ndarray:
    r"""
    Generate a matrix :math:`N \times 2` with `N` unitary vectors randomly spaced forming a circle in 2D.

    :math:`N` vectors spaced randomly between `angle_range[0]` and `angle_range[1]`.
    The vectors are :math:`\mathbf{M}_i = (\cos(\alpha_i), \sin(\alpha_i))`.

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
    M: np.ndarray of float(N, 2)
        Matrix with :math:`N` unitary vectors in 2D space.
    """

    if seed is not None:
        np.random.seed(seed)

    if angle_range is None:
        angle_range = [0, 2 * np.pi]

    # randomly spaced circular parameters
    alphas = np.random.uniform(angle_range[0], angle_range[1], N)
    M = np.array([np.cos(alphas), np.sin(alphas)]).T

    return M


def _2D_circle_spaced(
    N: int = 10, angle_range: list | None = None, edges=True
) -> np.ndarray:
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

    edges: bool, default=True
        If True, the edges are included in the range.

    Returns
    -------
    M: np.ndarray of float(2, N)
        Matrix with :math:`N` unitary vectors in 2D space.
    """

    if angle_range is None:
        angle_range = [0, 2 * np.pi]

    # evenly spaced circular parameters
    if edges:
        alphas = np.linspace(
            angle_range[0],
            angle_range[1],
            N,
            endpoint=(angle_range[1] - angle_range[0]) % (2 * np.pi) != 0,
        )
    else:
        alphas = np.linspace(angle_range[0], angle_range[1], N + 1, endpoint=False)[1:]
    M = np.array([np.cos(alphas), np.sin(alphas)]).T

    return M


def _dale_decoder_ortho(
    E: np.ndarray, NE: int, NI: int, optim=True, latent_sep=None
) -> np.ndarray:
    r"""
    Compute the decoder matrix for a EI network.

    Parameters
    ----------
    E : np.ndarray of float(do, N)
        Encoding matrix.

    NE : int
        Number of excitatory neurons.

    NI : int
        Number of inhibitory neurons.

    optim : bool, default=True
        If True, the optimization is performed. If False, :math:`\mathbf{D} = \pm \mathbf{E}^\top` (you need to make sure
        this does not violate daliean constraints)

    latent_sep : np.ndarray of bool(do, 2), default=None
        If not None, the latent space is separated in excitatory and inhibitory dimensions, according to the matric.
    """

    if not optim and latent_sep is not None:
        raise Warning("The latent_sep parameter is ignored when optim=False")

    N = NE + NI
    do = E.shape[1]
    EIvec = np.ones(N)
    EIvec[:NI] = -1
    mask = np.zeros((do, N), dtype=bool)

    if optim:
        if latent_sep is not None:
            for d_id in range(do):
                if not latent_sep[d_id, 0]:
                    mask[d_id, NI:] = True
                if not latent_sep[d_id, 1]:
                    mask[d_id, :NI] = True

        D = cp.Variable((do, N))
        penalty = cp.norm(D - EIvec * E.T)
        if NI != 0 and NE != 0:
            constraints = [D.T[:NI, :] @ E.T <= 0, D.T[NI:, :] @ E.T >= 0]
        elif NI == 0:
            constraints = [D.T @ E.T >= 0]  # fully E network
        else:
            constraints = [D.T @ E.T <= 0]  # fully I network
        if np.any(mask):
            constraints += [D[mask] == 0]
        prob = cp.Problem(cp.Minimize(penalty), constraints)  # type: ignore
        prob.solve(cp.SCS, eps_abs=1e-9, eps_rel=0)
        if prob.status != cp.OPTIMAL:
            raise ValueError("Problem not feasible")
        else:
            D = D.value
    else:
        D = EIvec * E.T

    return D
