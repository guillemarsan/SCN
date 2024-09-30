***********
Autoencoder
***********

.. role:: python(code)
   :language: python

.. Blabla

.. code-block:: python
    :caption: Autoencoder example
    :name: Autoencoder example

    import numpy as np

    from SCN import Autoencoder, Simulation, transform

    # Example network
    net = Autoencoder.init_2D_spaced(N=10, spike_scale=0.4)

    # Construct input in a circle with transform
    x = transform.angle_encode(np.linspace(0, 1, 10000))

    # Run simulation
    sim = Simulation()
    sim.run(net, x)

    # Animate the simulation
    sim.animate()

.. image:: ../_static/gifs/Autoencoder_long.gif
    :alt: Example of Autoencoder
    :align: center
