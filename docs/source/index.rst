:sd_hide_title:
:html_theme.sidebar_secondary.remove: true

************
SCN
************

.. This is for the website preview
.. .. raw:: html

..     <img src="_static/favicon/wide_favicon_base.png" hidden>

.. div:: sd-text-center sd-text-primary sd-fs-4

    .. raw:: html

        SCN allows you to run, visualize and interpret low-rank LIF networks.

    .. grid:: 2 4 4 4
        :gutter: 1 2 2 2
        :padding: 4 0 2 2

        .. grid-item-card:: Autoencoder
            :link: play/Autoencoder.html

            .. image:: _static/gifs/Autoencoder_short.gif
                :alt: Example of Autoencoder

        .. grid-item-card:: I Population
            :link: play/SinglePopulation.html

            .. image:: _static/gifs/SinglePopulation_short.gif
                :alt: Example of Single_Population

        .. grid-item-card:: 2 Neurons
            :link: play/EINetwork2.html

            .. image:: _static/gifs/EINetwork2_short.gif
                :alt: Example of EI Network with 2 neurons

        .. grid-item-card:: E/I Network
            :link: play/EINetwork.html

            .. image:: _static/gifs/EINetwork_short.gif
                :alt: Example of EI Network

.. grid:: 1 2 3 3
    :margin: 4 4 0 0
    :padding: 1
    :gutter: 3

    .. grid-item-card:: :fa:`download` Plug
        :link: plug/plug
        :link-type: doc
        :text-align: center

        How to install SCN.

    .. grid-item-card:: :fa:`play` Play
        :link: play/play
        :link-type: doc
        :text-align: center

        Basic examples to get started.

    .. grid-item-card:: :fa:`book` API Reference
        :link: references/index
        :link-type: doc
        :text-align: center

        Detailed documentation.


=========

For further reading on:

- Autoencoder:

    .. centered:: `Calaim, N., Dehmelt, F. A., Gonçalves, P. J., & Machens, C. K. (2022). The geometry of robustness in spiking neural networks. Elife, 11, e73276. <https://doi.org/10.7554/eLife.73276>`_

    .. centered:: `Denève, S., & Machens, C. K. (2016). Efficient codes and balanced networks. Nature neuroscience, 19(3), 375-382.  <https://doi.org/10.1038/nn.4243>`_

    .. centered:: `Boerlin, M., Machens, C. K., & Denève, S. (2013). Predictive coding of dynamical variables in balanced spiking networks. PLoS computational biology, 9(11), e1003258. <https://doi.org/10.1371/journal.pcbi.1003258>`_

- E/I Network:

    .. centered:: `Mancoo, A., Keemink, S., & Machens, C. K. (2020). Understanding spiking networks through convex optimization. Advances in neural information processing systems, 33, 8824-8835. <https://proceedings.neurips.cc/paper_files/paper/2020/hash/64714a86909d401f8feb83e8c2d94b23-Abstract.html>`_

    .. centered:: `Podlaski, W. F., & Machens, C. K. (2024). Approximating nonlinear functions with latent boundaries in low-rank excitatory-inhibitory spiking networks. Neural Computation, 36(5), 803-857. <https://doi.org/10.1162/neco_a_01658>`_


=======

.. grid:: 1 2 2 2
    :margin: 4 4 0 0
    :gutter: 5

    .. grid-item-card:: :fa:`code` Source Code
        :link: https://github.com/guillemarsan/SCN
        :text-align: center

        You can read the code and contribute here.

    .. grid-item-card:: :fa:`people-group` Machens Lab
        :link: https://machenslab.org/
        :text-align: center

        Check the Theoretical Neuroscience (Machens) lab, Champalimaud Foundation, Lisbon, Portugal.

.. toctree::
    :hidden:

    plug/plug
    play/play
    references/index
