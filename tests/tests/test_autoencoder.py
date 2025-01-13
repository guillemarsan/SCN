import numpy as np

from SCN import Autoencoder, Simulation


def test_autoencoder_init():
    net1 = Autoencoder.init_2D_random(
        N=2, angle_range=[0, -np.pi / 2], spike_scale=0.8, T=0.2, seed=3
    )
    net2 = Autoencoder.init_2D_spaced(N=2, spike_scale=0.5)
    net3 = Autoencoder.init_cube(
        d=2,
        one_quadrant=True,
        spike_scale=np.array([0.5, 0.4]),
        T=np.array([0.1, 0.3]),
    )
    net4 = Autoencoder.init_random(d=2, N=6, spike_scale=0.05, T=0.2, seed=2)
    net5 = Autoencoder(np.array([[1, 0, -1, 0], [0, 1, 0, -1]]))
    sim = Simulation()
    x = np.array(
        [
            -np.cos(np.linspace(0, np.pi / 2, 10000)),
            np.sin(np.linspace(0, np.pi / 2, 10000)),
        ]
    )
    for net in [net1, net2, net3, net4, net5]:
        sim.run(net, x, draw_break="no")
        net.plot(save=False)
        sim.plot(save=False)


def test_autoencoder_3D():

    net1 = Autoencoder.init_random(d=3, N=5)
    net2 = Autoencoder.init_cube(d=3, one_quadrant=True)
    sim = Simulation()
    x = np.array(
        [
            -np.cos(np.linspace(0, np.pi / 2, 1000)),
            np.sin(np.linspace(0, np.pi / 2, 1000)),
            np.sin(np.linspace(0, np.pi / 2, 1000)),
        ]
    )
    for net in [net1, net2]:
        sim.run(net, x, draw_break="no", Tmax=1)
        net.plot(save=False)
        sim.plot(save=False)
        sim.animate()


def test_autoencoder_optimization():

    net = Autoencoder.init_random(d=2, N=6, spike_scale=0.05, T=0.2, seed=2)
    sim = Simulation()
    x = np.ones((2, 10000))
    x[:, 5000:] = -1

    sim.run(net, x, draw_break="no")
    sim.optimize(net, x)
    sim.plot(save=False)
    sim.animate()
    sim.plot_io(save=False)
    sim.plot_rates(save=False)
    sim.plot_spikes(save=False)


def test_autoencoder_rate_space():
    net = Autoencoder.init_2D_random(
        N=2, angle_range=[0, -np.pi / 2], spike_scale=0.8, T=0.2, seed=3
    )
    sim = Simulation()
    x = np.array(
        [
            -np.cos(np.linspace(0, np.pi / 2, 10000)),
            np.sin(np.linspace(0, np.pi / 2, 10000)),
        ]
    )
    sim.run(net, x, draw_break="no")
    net.plot_rate_space(x[:, 0])
    sim.plot(save=False)
    sim.animate()


test_autoencoder_3D()
