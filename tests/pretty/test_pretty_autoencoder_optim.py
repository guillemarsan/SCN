import numpy as np

from SCN import Autoencoder, Simulation, transform


def test_pretty_autoencoder_optim():

    # Example network
    net = Autoencoder.init_2D_random(N=2, angle_range=[-np.pi / 2, -np.pi], seed=0)

    # Construct input
    x = transform.angle_encode(np.linspace(0, 1, 10000))

    # Run simulation
    sim = Simulation()
    sim.run(net, x)

    # Solve constrained optimization
    sim.optimize(net, x)

    # Animate the simulation
    sim.animate()
