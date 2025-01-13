***********
3D Plots
***********

.. role:: python(code)
   :language: python

.. Blabla

.. code-block:: python
    :caption: Example with 3D plots
    :name: Example with 3D plots

    import numpy as np

    from SCN import Autoencoder, Simulation, transform

    # Example network
    net = Autoencoder.init_cube(d=3, one_quadrant=True, T=np.array([0.4, 0.5, 0.6]))

    # Construct input
    x = transform.angle_encode(np.linspace(-np.pi / 2, np.pi / 2, 10000), dim=3)

    # Run simulation
    sim = Simulation()
    sim.run(net, x)

    # Animate the simulation
    sim.animate()

.. image:: ../_static/gifs/Autoencoder_3D_long.gif
    :alt: Example of 3D plots
    :align: center
