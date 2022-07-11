Getting Started
===============

1. Install `GNINA <https://github.com/gnina/gnina>`_. It's best to install GNINA from source, but you can use the `pre-compiled binary <https://github.com/gnina/gnina/releases/>`_ for docking if you do not care to have GPU acceleration.

2. Clone this repository

3. `cd` into the cloned repository and run `pip install .`. This will install all remaining dependencies.

4. Download the PDB files you would like to dock to into `structures/raw`

5. Create the necessary `.info` files in `structures/raw` to describe the ligands in the downloaded PDB files

6. Create a smiles CSV, `ligands.csv`, of any additional ligands that do not have a known docked structure. This should be of the form::
        ID,SMILES
        <ligand_name>,<smiles_string>

7. Run `open_combind prep-dock-and-predict ligands.csv` to run the whole docking procedure.
