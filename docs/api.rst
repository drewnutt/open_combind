API Documentation
=================

.. toctree::
   :maxdepth: 3
    
Main Combind Commands
---------------------
.. autofunction:: open_combind.structprep

.. autofunction:: open_combind.ligprep

.. autofunction:: open_combind.dock_ligands

.. autofunction:: open_combind.featurize

.. autofunction:: open_combind.pose_prediction

..
.. automethod:: open_combind.extract_top_poses

Docking Preparation
-------------------

struct_process
*********************
.. automodule:: open_combind.dock.struct_process
   :members: struct_process, load_complex, get_ligands_frompdb
 
struct_align
*******************************
.. automodule:: open_combind.dock.struct_align
   :members: struct_align, align_separate_ligand
 
struct_sort
**********************
.. automodule:: open_combind.dock.struct_sort
   :members:
 
ligprep
*******
.. automodule:: open_combind.dock.ligprep
   :members:
 
dock
****
.. automodule:: open_combind.dock.dock
   :members:


Featurization
-------------

Features
********
.. autoclass:: open_combind.features.features.Features
   :members:
   :undoc-members:

Interaction Fingerprint (IFP)
*****************************
.. autoclass:: open_combind.features.ifp.Molecule
   :members:
   :undoc-members:
