**************************
Single Population (E or I)
**************************

.. role:: python(code)
   :language: python

.. Blabla

.. code-block:: python
    :caption: Single Population example
    :name: Single Population example

    from SCN import Single_Population, Simulation

    # Example network
    net = Single_Population.init_2D_random(di=1, N=5, seed=1, dale="I")

    # Construct input
    x1 = np.zeros(2000)
    x2 = np.linspace(0, 0.5, 8000)
    x = np.hstack([x1, x2])

    # Run simulation
    sim = Simulation()
    sim.run(net, x, y0=np.array([0, -1]))

    # Animate the simulation
    sim.animate()

.. image:: ../_static/gifs/SinglePopulation_long.gif
    :alt: Example of Single Population
    :align: center

You can also do the same with an all excitatory population by changing the ``dale`` parameter to "E". However, since
there is no inhibition, the network will simply explode!

.. code-block:: python
    :caption: Single Population E example
    :name: Single Population E example

    net = Single_Population.init_2D_random(di=1, N=5, seed=1, dale="E")

.. image:: ../_static/gifs/SinglePopulationE_long.gif
    :alt: Example of Single Population E
    :align: center
