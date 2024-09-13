import numpy as np

from SCN import EI_Network, Simulation


def test_pretty_single_population():

    # Example network
    net = EI_Network.init_2D_random(di=1, NI=5, seed=1)

    # Construct input
    x1 = np.zeros(2000)
    x2 = np.linspace(0, 0.2, 8000)
    x = np.hstack([x1, x2])

    # Run simulation
    sim = Simulation()
    sim.run(net, x, y0=np.array([0, -1]))

    # Animate the simulation
    sim.animate()
