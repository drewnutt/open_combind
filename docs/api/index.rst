.. _api/index:

=================
Command Line Interface
=================
Open-Combind is primarily used through its command line interface. It provides a series of commands that step through the molecular docking workflow or one command to run everything in serial.

.. toctree::
   :maxdepth: 3
    
Main Combind Commands
---------------------

structprep
**********
.. autofunction:: open_combind.structprep

ligprep
*******
.. autofunction:: open_combind.ligprep

dock_ligands
************
.. autofunction:: open_combind.dock_ligands

featurize
*********
.. autofunction:: open_combind.featurize

pose_prediction
***************
.. autofunction:: open_combind.pose_prediction

..
    .. automethod:: open_combind.extract_top_poses
