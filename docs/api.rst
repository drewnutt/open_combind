API Documentation
=================

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

Docking Preparation
-------------------

struct_process
**************
.. automodule:: open_combind.dock.struct_process
   :members: struct_process, load_complex, create_correct_ligand_sdf
 
ligand_handling
***************
.. automodule:: open_combind.dock.ligand_handling
   :members: get_ligand_info_RCSB, get_ligands_from_RCSB, get_ligand_from_SMILES, ligand_selection_to_mol, RDKitParseException

struct_align
************
.. automodule:: open_combind.dock.struct_align
   :members: struct_align, align_separate_ligand
 
struct_sort
***********
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

Maximum Common Substructure (MCSS)
**********************************
.. automodule:: open_combind.features.mcss
   :members:

Interaction Fingerprint (IFP)
*****************************
.. autoclass:: open_combind.features.ifp.Molecule
   :members:
   :undoc-members:
