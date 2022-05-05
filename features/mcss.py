import tempfile
import numpy as np
import subprocess
import os
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdFMCS
# from plumbum.cmd import obrms
from utils import mp

# To compute the substructure similarity for a pair of candidate poses, the maximum common
# substructure of the two ligands is identified using Canvas (Schrodinger LLC) and then mapped
# onto each candidate pose. Finally, the RMSD between these two sets of atoms is computed and
# used as the measure of substructure similarity. We defined custom atom and bond types for
# computation of the common scaffold (Supplementary Table 6). Substructure similarity is not
# considered for pairs of ligands with a maximum common substructure of less than half the size
# of the smaller ligand. Hydrogen atoms were not included in the substructure nor when
# determining the total number of atoms in each ligand.
def mcss(sts1, sts2, mcss_types_file):
    """
    Computes rmsd between mcss for atoms in two poseviewer files.

    Returns a (# poses in pv1) x (# poses in pv2) np.array of rmsds.
    """
    memo = {}
    # sts1 = [merge_halogens(st.copy()) for st in sts1]
    # sts2 = [merge_halogens(st.copy()) for st in sts2]

    bad_apples = []
    rmsds = []
    for i, st1 in enumerate(sts1):
        n_st1_atoms = st1.GetNumHeavyAtoms()
        smi1 = Chem.MolToSmiles(st1)
        rmsds += [np.zeros(len(sts2))]
        for j, st2 in enumerate(sts2):
            if j > i:  # only calculate lower triangle
                break
            n_st2_atoms = st2.GetNumHeavyAtoms()
            smi2 = Chem.MolToSmiles(st2)
            if (smi1, smi2) in memo:
                mcss, n_mcss_atoms, remove_idxs = memo[(smi1, smi2)]
            else:
                mcss, n_mcss_atoms, remove_idxs = compute_mcss(st1, st2, mcss_types_file)
                memo[(smi1, smi2)] = (mcss, n_mcss_atoms, remove_idxs)
                memo[(smi2,smi1)] = (mcss, n_mcss_atoms, {'st1':remove_idxs['st2'],'st2':remove_idxs['st1']})

            if (2*n_mcss_atoms <= min(n_st1_atoms, n_st2_atoms)
                or n_mcss_atoms <= 10):
                bad_apples += [(i,j)]
                continue

            rmsds[-1][j] = compute_mcss_rmsd(st1, st2, remove_idxs)

    rmsds_bottom = np.vstack(rmsds)
    print(rmsds_bottom)
    if len(bad_apples):
        xind,yind = np.array(bad_apples).T
        rmsds_bottom[(xind,yind)] = float('inf')
    return rmsds_bottom + rmsds_bottom.T - np.diag(np.diag(rmsds_bottom))

def mcss_mp(sts1, sts2, mcss_types_file,processes=1):
    """
    Computes rmsd between mcss for atoms in two poseviewer files.

    Returns a (# poses in pv1) x (# poses in pv2) np.array of rmsds.
    """
    memo = {}
    # sts1 = [merge_halogens(st.copy()) for st in sts1]
    # sts2 = [merge_halogens(st.copy()) for st in sts2]

    unfinished = []
    for j, st1 in enumerate(sts1):
        n_st1_atoms = st1.GetNumHeavyAtoms()
        smi1 = Chem.MolToSmiles(st1)
        # rmsds += [np.zeros(len(sts2))]
        for i, st2 in enumerate(sts2):
            if i > j:  # only calculate lower triangle
                break
            n_st2_atoms = st2.GetNumHeavyAtoms()
            smi2 = Chem.MolToSmiles(st2)
            if (smi1, smi2) in memo:
                mcss, n_mcss_atoms, remove_idxs = memo[(smi1, smi2)]
            else:
                mcss, n_mcss_atoms, remove_idxs = compute_mcss(st1, st2, mcss_types_file)
                memo[(smi1, smi2)] = (mcss, n_mcss_atoms, remove_idxs)
                memo[(smi2,smi1)] = (mcss, n_mcss_atoms, {'st1':remove_idxs['st2'],'st2':remove_idxs['st1']})

            retain_inf = False
            if (2*n_mcss_atoms <= min(n_st1_atoms, n_st2_atoms)
                or n_mcss_atoms <= 10):
                retain_inf = True
            unfinished += [(st1,i,st2,j,remove_idxs,retain_inf)]

    results = mp(compute_mcss_rmsd_mp,unfinished,processes)

    rows, cols, values = zip(*results)
    rmsds_bottom = np.zeros((len(sts1),len(sts2)))
    rmsds_bottom[rows,cols] = values
    return rmsds_bottom + rmsds_bottom.T - np.diag(np.diag(rmsds_bottom))

def compute_mcss_rmsd(st1, st2, remove_idxs):
    """
    Compute minimum rmsd between mcss(s).

    Takes into account that the mcss smarts pattern could
    map to multiple atom indices (e.g. symetric groups).

    remove_idxs: dictionary with keys 'st1' and 'st2' that have lists
    of atom indices lists to remove to create the MCSS for each pose
    """
    rmsd = float('inf')
    for rmatom_idx1 in remove_idxs['st1']:
        ss1 = get_substructure(st1,rmatom_idx1)
        for rmatom_idx2 in remove_idxs['st2']:
            ss2 = get_substructure(st2,rmatom_idx2)
            _rmsd = calculate_rmsd(ss1, ss2)
            rmsd = min(_rmsd, rmsd)
    return rmsd

def compute_mcss_rmsd_mp(st1, i, st2, j, remove_idxs, retain_inf):
    """
    Compute minimum rmsd between mcss(s).

    Takes into account that the mcss smarts pattern could
    map to multiple atom indices (e.g. symetric groups).

    remove_idxs: dictionary with keys 'st1' and 'st2' that have lists
    of atom indices lists to remove to create the MCSS for each pose
    """
    if retain_inf:
        rmsd = float('inf')
    else:
        rmsd = compute_mcss_rmsd(st1,st2,remove_idxs)
    return (i,j, rmsd)

def compute_mcss(st1, st2, mcss_types_file):
    """
    Compute smarts patterns for mcss(s) between two structures.
    """
    res = rdFMCS.FindMCS([st1,st2], ringMatchesRingOnly=True,
            completeRingsOnly=True, bondCompare=rdFMCS.BondCompare.CompareOrderExact)
    if not res.canceled:
        mcss = res.smartsString
        num_atoms = res.numAtoms
    else:
        mcss = '[#6]'
        num_atoms = 1
    mcss_mol = Chem.MolFromSmarts(mcss)
    rmv_idx = {'st1': mcss_to_rmv_idx(st1, mcss_mol),
            'st2': mcss_to_rmv_idx(st2, mcss_mol)}

    return mcss, num_atoms, rmv_idx

def calculate_rmsd(pose1, pose2, eval_rmsd=False):
    """
    Calculates the RMSD between pose1 and pose2.

    pose1, pose2: rdkit.Mol
    eval_rmsd: verify that RMSD calculation is the same as obrms
    """
    assert pose1.HasSubstructMatch(pose2) or pose2.HasSubstructMatch(pose1), f"{pose1.GetProp('_Name')}&{pose2.GetProp('_Name')}"
    try:
        rmsd = Chem.GetBestRMS(pose1,pose2)
    except:
        try:
            rmsd = Chem.GetBestRMS(pose2,pose1)
        except:
            print(f"{pose1.GetProp('_Name')} and {pose2.GetProp('_Name')}, GetBestRMS doesn't work either way")
    # if eval_rmsd:
    #     obrmsd = calculate_rmsd_slow(pose1,pose2)
    #     assert np.isclose(obrmsd,rmsd,atol=1E-4), print(f"obrms:{obrmsd}\nrdkit:{rmsd}")
    return rmsd

# def calculate_rmsd_slow(pose1,pose2):
#     with tempfile.TemporaryDirectory() as tempd:
#         tmp1 = f'{tempd}/tmp1.sdf'
#         tmp2 = f'{tempd}/tmp2.sdf'

#         writer = Chem.SDWriter(tmp1)
#         writer.write(pose1)
#         writer.close()

#         writer = Chem.SDWriter(tmp2)
#         writer.write(pose2)
#         writer.close()

#         raw_rmsd = obrms[tmp1,tmp2]()
#         rmsd = float(raw_rmsd.strip().split()[-1])

#     return rmsd

def merge_halogens(structure):
    """
    Sets atomic number for all halogens to be that for flourine.
    This enable use of ConformerRmsd for atom typing schemes that
    merge halogens.
    """
    for atom in structure.atom:
        if atom.atomic_number in [9, 17, 35, 53]:
            atom.atomic_number = 9
    return structure

def mcss_to_rmv_idx(mol,mcss_mol):
    """
    Finds the atom indices that need to be removed from mol
    to be left with the mcss
    """
    mol_mcss = mol.GetSubstructMatches(mcss_mol)
    mol_fidx = set(range(mol.GetNumAtoms()))
    remove_idxs = []
    for keep_idx in mol_mcss:
        remove_idxs.append(sorted(list(mol_fidx - set(keep_idx)),reverse=True))

    return remove_idxs

def get_substructure(mol, remove_idxs):
    """
    Gets the substructure of mol by removing the atoms with indices
    in remove_idxs
    """
    rw_mol = Chem.RWMol(mol)
    for idx in sorted(remove_idxs,reverse=True):
        rw_mol.RemoveAtom(idx)

    assert rw_mol.GetNumAtoms() == (mol.GetNumAtoms() - len(remove_idxs))

    substruct = Chem.Mol(rw_mol)
    Chem.SanitizeMol(substruct)
    return substruct
# def n_atoms(st):
#     return sum(atom.element != 'H' for atom in st.atom)
