.. _topics-index:

Welcome to Open-ComBind's documentation!
===============================
Open-Combind is an open-source fork of `ComBind <https://github.com/drorlab/combind>`_. Open-ComBind removes all calls to proprietary software and replaces the docking software GLIDE with `GNINA <https://github.com/gnina/gnina>`_.

Open-ComBind is a molecular docking pipeline that harnesses multiple ligand predictions of a molecular docking software. Open-ComBind allows use of ligands without known docked poses to determine the docked pose of a query ligand. This is done through pairwise featurization of the poses and then minimization of the ComBind likelihood.

.. _first-steps:

First Steps
===========

.. toctree::
   :caption: First Steps
   :hidden:

   installation
   getting_started

:doc:`intro/installation`
        How to install Open-ComBind
:doc:`intro/getting_started`
        How to run Open-ComBind

.. _user-guide:

User Guide
==========
.. toctree::
   :caption: User Guide
   :hidden:
   :maxdepth: 1

   api/index
   api/dock
   api/features
   api/score
   api/pymol
   api/utils

:doc:`api/index`
        Basics of Open-ComBind: command line interface
:doc:`api/dock`
        Protein-ligand preprocessing and docking
:doc:`api/features`
        Featurization of docked ligand poses
:doc:`api/score`
        Scoring and prediction of top ligand pose
:doc:`api/pymol`
        Visualization of ligand pose features
:doc:`api/utils`
        Utilities for Open-ComBind

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
