Getting Started
===============

1. Download the PDB files you would like to dock to into ``structures/raw``

2. Create the necessary ``.info`` files in ``structures/raw`` to describe the ligands in the downloaded PDB files

3. Create a smiles CSV, ``ligands.csv``, of any additional ligands that do not have a known docked structure. This should be of the form::

        ID,SMILES
        <ligand_name>,<smiles_string>

4. Run ``open_combind prep-dock-and-predict ligands.csv`` to run the whole docking procedure.
