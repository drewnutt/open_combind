# Open-ComBind
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/drewnutt/open_combind/workflows/CI/badge.svg)](https://github.com/drewnutt/open_combind/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/drewnutt/open_combind/branch/main/graph/badge.svg)](https://codecov.io/gh/drewnutt/open_combind/branch/main)
[![docs](https://readthedocs.org/projects/open-combind/badge/?version=latest)](https://open-combind.readthedocs.io/en/latest/?badge=latest)

Open-source docking pipeline leveraging pairwise statistics

Open-ComBind is published [here](https://doi.org/10.1007/s10822-023-00544-y)

This is a fork of [ComBind](https://github.com/drorlab/combind) that removes all uses of proprietary API calls or proprietary software. This will produce similar (if not the exact same) results
as the original ComBind and it will be completely free to use.

This fork will focus on the use of [GNINA](https://github.com/gnina/gnina) as the
docking software rather than Glide. Mostly because Glide is proprietary, but also because
GNINA is a deep-learning based docking pipeline.

Example usage of Open-ComBind can be seen in this [Colab Notebook](https://colab.research.google.com/drive/1YhLydzEOW3g38SubIw1JOxOEivU7U1kg?usp=sharing)

### Open-ComBind

Open-ComBind integrates data-driven modeling and physics-based docking for
improved binding pose prediction and binding affinity prediction.

Given the chemical structures of several ligands that can bind
a given target protein, Open-ComBind solves for a set of poses, one per ligand, that
are both highly scored by physics-based docking and display similar interactions
with the target protein. Open-ComBind quantifies this vague notion of "similar" by
considering a diverse training set of protein complexes and computing the
overlap between protein–ligand interactions formed by distinct ligands when
they are in their correct poses, as compared to when they are in randomly
selected poses. To predict binding affinities, poses are predicted for
the known binders using Open-ComBind, and then the candidate molecule is scored
according to the Open-ComBind score w.r.t. the selected poses.

## Predicting poses for known binders

First, see instructions for software installation at the bottom of this page.

Running Open-ComBind can be broken into several components: data curation,
data preparation (including docking), featurization of docked poses,
and the Open-ComBind scoring itself.

Note that if you already have docked poses for your molecules of interest, you
can proceed to the featurization step. If you are knowledgeable about your target
protein, you may well be able to get better docking results by manually
preparing the data than would be obtained using the automated procedure
implemented here.

### Curation of raw data

To produce poses for a particular protein, you'll need to provide a 3D structure
of the target protein and chemical structures of ligands to dock.

These raw inputs need to be properly stored so that the rest of the pipeline
can recognize them.

The structure(s) should be stored in a directory `structures/raw`.
Each structure should be split into two files `NAME_prot.pdb` and `NAME_lig.pdb`
containing only the protein and only the ligand, respectively.

If you'd prefer to prepare your structures yourself, save your
prepared files to `structures/proteins` and `structures/ligands`. Moreover,
you could even just begin with a GNINA docking template file (i.e. `gnina -r <path_to_receptor_file> --autobox_ligand <path_to_crystal_ligand> `)
placed on the first line of a file.

Ligands can be specified in a csv file with a header line containing at
least the entries "ID" and "SMILES", specifying the ligand name and the ligand
chemical structure. Alternatively you can specify your ligands with a sdf
containing an entry for each ligand.

### Data preparation and docking

Use the following command to prepare the structural data using [ProDy](https://github.com/prody/ProDy), 
align the structures to each other, and produce a docking template line.

```
open_combind structprep
```

In parallel, you can prepare the ligand data using the following command.
By default, the ligands will be written to separate files (one ligand per file).
You can specify the `--multiplex` flag to write all of the ligands to the same
file.

```
open_combind ligprep ligands.csv
```

Once the GNINA template file and ligand data have been prepared, you can run the
docking. The arguments to the dock command are a list of ligand files to be
docked. By default, the GNINA template file is the alphabetically first template present
in `structures/template`; use the `--template` option to specify a different template. Additionally,
you can utilize `--slurm` to create a tarball of all of the necessary files and update the docking `.txt`
file to use paths in the tarball. You can run the GNINA commands right after they are generated by using `--now` (this will likely be slow and consume all of the cpus available on the workstation).

```
open_combind dock-ligands ligands/*/*.sdf
```

### Featurization

```
open_combind featurize features docking/*/*.sdf.gz
```

### Pose prediction with Open-ComBind

```
open_combind pose-prediction features poses.csv
```

Optionally, you can extract the poses selected by ComBind to a single file.
The resulting file will contain the protein structure followed by one pose (the
one selected by ComBind) for each ligand.

```
open_combind extract-top-poses poses.csv docking/*/*.sdf.gz
```

## Benchmarking data

See `stats_data/pdbs_for_benchmark.csv` for a list of PDBs used for benchmarking
Open-ComBind and ComBind. The "query" column gives the PDB for the ligand being docked, the
"grid" column gives the structure the query is docked to, and the "mcss<0.5"
column indicates whether the query ligand shares a common substructure with 
the co-crystal ligand in the structure being docked to.

See `stats_data/structures.tar.gz` for the raw structural data used for
benchmarking ComBind.

See `stats_data/helper_best_affinity_diverse.csv` and `stats_data/helper_best_mcss.csv`
for a list of the "helper ligands" used when benchmarking Open-ComBind and ComBind. Each row
lists a docking ligand and one helper ligand; all the entries for each docking ligand
should be aggregrated. (Most docking ligands have 20 associated helper ligands.)

## Installation

    1. Install [GNINA](https://github.com/gnina/gnina). It's best to install GNINA from source, but you can use the pre-compiled binary for docking if you do not care to have GPU acceleration.
    2. Install [OpenMM](https://github.com/openmm/openmm). 
    3. Install [PDBFixer](https://github.com/openmm/pdbfixer)
    4. Install [PyMol](https:/github.com/schrodinger/pymol-open-source). Easiest with conda/mamba: `conda install -c conda-forge pymol-open-source`
    3. Clone this repository
    4. `cd` into the cloned repository and run `pip install .`. This will install any remaining dependencies.

### Copyright

Copyright (c) 2022, Andrew McNutt


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
