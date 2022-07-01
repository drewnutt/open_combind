import os
import tempfile
import numpy as np
from rdkit.Chem import ShapeTanimotoDist
from open_combind.utils import mp

# CMD = '$SCHRODINGER/shape_screen -shape {poses1} -screen {poses2} {typing} {norm} -distinct -inplace -NOJOBID'

# This shape screening is based on the Schrodinger paper: https://doi.org/10.1021/ci2002704
# Used a method which is quite similar to tanimoto similarity of the shapes but easier to calculate
# Different atom/feature types of the overlaps:
# Name    | Description
# -----   |-------
# None    |   all atoms equivalent (shape-only scoring)
# QSAR    |   Phase QSAR atom types 
# Element |   elemental atom types
# MMod	|   MacroModel atom types (over 150 unique atom types)
# Pharm	|   Phase pharmacophore feature types

# QSAR atom types = hydrophobic, electron withdrawing, H-bond donor, negative ionic, positive ionic, and other

# Pharmacophore feature types= aromatic, hydrophobic, H-bond acceptor, H-bond donor, negative ionic, and positive ionic.
# Each site was represented by a hard sphere of radius 2 Ǻ, and as with the atom typing schemes, volume overlap scores were only computed between sites of the same type


def shape(conformers1, conformers2, version=None):
    shape_sims = []
    for i, conf1 in enumerate(conformers1):
        shape_sims += [np.zeros(len(conformers2)]
        for j, conf2 in enumerate(conformers2):
            if j >= i:
                continue
            shape_sims[-1][j] = ShapeTanimotoDist(conf1,conf2)
    shape_sims_bottom = np.vstack(shape_sims)
    
    filled_matrix = shape_sims_bottom + shape_sims_bottom.T - np.diag(np.diag(shape_sims_bottom))
    np.fill_diagonal(filled_matrix,1)

    return filled_matrix

def shape_mp(conformers1, conformers2, version=None, processes=1):
    def compute_shape_mp(conformation,idx):
        shape_sims = np.zeros(len(conformers2)
        for j, conf2 in enumerate(conformers2): 
            if j>= i: continue
            shape_sims[j] = ShapeTanimotoDist(conformation,conf2)

        return idx, shape_sims

    unfinished = []
    for i, conf1 in enumerate(conformers1):
        unfinished += [(conf1,i,conformers2)]
    results = mp(compute_shape_mp,unfinished,processes)
    idx, shape_sims_lists = zip(*results)
    shape_sims_bottom = np.zeros((len(conformers1),len(conformers2)))
    shape_sims_bottom[idx] = shape_sims_lists
    
    filled_matrix = shape_sims_bottom + shape_sims_bottom.T - np.diag(np.diag(shape_sims_bottom))
    np.fill_diagonal(filled_matrix,1)

    return filled_matrix

