import numpy as np

from SCN import Autoencoder, Simulation, transform


def test_pretty_autoencoder_3D():

    # Example network
    net = Autoencoder.init_cube(d=3, one_quadrant=True, T=np.array([0.4, 0.5, 0.6]))

    # Construct input
    x = transform.angle_encode(np.linspace(-np.pi / 2, np.pi / 2, 10000), dim=3)

    # Run simulation
    sim = Simulation()
    sim.run(net, x)

    # Animate the simulation
    sim.animate()
