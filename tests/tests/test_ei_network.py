import numpy as np

from SCN import EI_Network, Simulation


def test_ei_network_init():
    net1 = EI_Network.init_random(di=1, do=2, NE=2, NI=2, seed=7, latent_sep=np.eye(2))
    net2 = EI_Network.init_2D_random(
        di=1, NE=3, NI=3, seed=1, spike_scale=0.5, latent_sep=np.eye(2)
    )
    net3 = EI_Network.init_2D_spaced(di=1, NE=2, NI=2, Fseed=0, spike_scale=0.5)
    nets = [net1, net2, net3]

    sim = Simulation()
    x1 = np.zeros((2000))
    x2 = np.linspace(0, 0.5, 8000)
    x = np.hstack([x1, x2])
    for net in nets:
        sim.run(net, x, draw_break="one", criterion="inh_max", y0=np.array([0, -1]))
        net.plot()
        sim.plot()
        sim.plot_io()
        sim.plot_rates()
        sim.plot_spikes()


def test_ei_network_optimization():

    net = EI_Network.init_2D_random(di=1, NE=3, NI=3, seed=5, Fseed=0)

    # Construct input
    x1 = np.zeros((1, 2000))
    x2 = np.linspace(0, 0.5, 8000)
    x = np.hstack([x1, x2[np.newaxis, :]])
    sim = Simulation()

    sim.run(net, x, draw_break="one", criterion="inh_max")
    sim.optimize(net, x)
    sim.plot(save=False)
    sim.animate()
    sim.plot_io(save=False)
    sim.plot_rates(save=False)
    sim.plot_spikes(save=False)


def test_ei_network_rate_space():
    net = EI_Network.init_2D_spaced(
        di=1,
        NE=1,
        NI=1,
        Fseed=0,
        angle_range=[np.pi / 8, 3 * np.pi / 8],
        latent_sep=np.eye(2),
        spike_scale=0.5,
    )

    x1 = np.zeros((1, 2000))
    x2 = np.linspace(0, 0.5, 8000)
    x = np.hstack([x1, x2[np.newaxis, :]])
    sim = Simulation()
    sim.run(net, x, draw_break="no")
    net.plot_rate_space(x[:, 0])
    sim.plot(save=False)
    sim.animate()
