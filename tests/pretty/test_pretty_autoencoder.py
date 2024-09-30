import numpy as np

from SCN import Autoencoder, Simulation, transform


def test_pretty_autoencoder():

    # Example network
    net = Autoencoder.init_2D_spaced(N=10, spike_scale=0.4)

    # Construct input
    x = transform.angle_encode(np.linspace(0, 1, 10000))

    # Run simulation
    sim = Simulation()
    sim.run(net, x)

    # Animate the simulation
    sim.animate()
