import tempfile
import numpy as np
import subprocess
import os
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdFMCS
# from plumbum.cmd import obrms
from open_combind.utils import mp

# To compute the substructure similarity for a pair of candidate poses, the maximum common
# substructure of the two ligands is identified using Canvas (Schrodinger LLC) and then mapped
# onto each candidate pose. Finally, the RMSD between these two sets of atoms is computed and
# used as the measure of substructure similarity. We defined custom atom and bond types for
# computation of the common scaffold (Supplementary Table 6). Substructure similarity is not
# considered for pairs of ligands with a maximum common substructure of less than half the size
# of the smaller ligand. Hydrogen atoms were not included in the substructure nor when
# determining the total number of atoms in each ligand.

def mcss(sts1, sts2):
    """
    Computes root mean square deviation (RMSD) between the maximum common substructure (MCSS) for atoms in two poseviewer files.

    Parameters
    ----------
    sts1 : :class:`list[Mol]<list>`
        Set of ligand poses as :class:`~rdkit.Chem.rdchem.Mol` s to compute the MCSS RMSD with `sts2`
    sts2 : :class:`list[Mol]<list>`
        Set of ligand poses as :class:`~rdkit.Chem.rdchem.Mol` s to compute the MCSS RMSD with `sts1`

    Returns
    -------
    :class:`~numpy.ndarray`
        (# poses in sts1) x (# poses in sts2) of the computed MCSS RMSDs. -1 indicates no RMSD was calculated.

    See Also
    --------
    mcss_mp : parallelized

    Notes
    -----

    If a pair of ligand poses, :math:`pose_1` and :math:`pose_2`, has a MCSS, :math:`mcss`, then we compute the RMSD if 

    .. math:: \\textrm{numHeavyAtoms}(mcss)\\geq \\frac{1}{2}\\textrm{min}(\\textrm{numHeavyAtoms}(pose_1),\\textrm{numHeavyAtoms}(pose_2))

    """

    memo = {}
    # sts1 = [merge_halogens(st.copy()) for st in sts1]
    # sts2 = [merge_halogens(st.copy()) for st in sts2]

    bad_apples = []
    rmsds = []
    for i, st1 in enumerate(sts1):
        n_st1_atoms = st1.GetNumHeavyAtoms()
        sma1 = Chem.MolToSmarts(st1)
        rmsds += [np.zeros(len(sts2))]
        for j, st2 in enumerate(sts2):
            if j > i:  # only calculate lower triangle
                break
            n_st2_atoms = st2.GetNumHeavyAtoms()
            sma2 = Chem.MolToSmarts(st2)
            if (sma1, sma2) in memo:
                mcss, n_mcss_atoms, keep_idxs = memo[(sma1, sma2)]
            else:
                mcss, n_mcss_atoms, keep_idxs = compute_mcss(st1, st2)
                memo[(sma1, sma2)] = (mcss, n_mcss_atoms, keep_idxs)
                memo[(sma2, sma1)] = (mcss, n_mcss_atoms, {'st1':keep_idxs['st2'],'st2':keep_idxs['st1']})

            if (2*n_mcss_atoms < min(n_st1_atoms, n_st2_atoms)):
                # or n_mcss_atoms <= 10):
                bad_apples += [(i,j)] #,2*n_mcss_atoms < min(n_st1_atoms, n_st2_atoms),n_mcss_atoms <= 10,st1.GetProp("_Name"),st2.GetProp("_Name"))]
                continue

            rmsds[-1][j] = compute_mcss_rmsd(st1, st2, keep_idxs)

    rmsds_bottom = np.vstack(rmsds)
    if len(bad_apples):
        xind, yind= np.array(bad_apples).T
        rmsds_bottom[(xind,yind)] = -1 #float('inf')
    filled_matrix = rmsds_bottom + rmsds_bottom.T - np.diag(np.diag(rmsds_bottom))
    return np.where(filled_matrix<0, np.inf, filled_matrix)

def mcss_mp(sts1, sts2, processes=1):
    """
    Computes root mean square deviation (RMSD) between the maximum common substructure (MCSS) for atoms in two poseviewer files.

    Parallelized accross `processes`.

    Parameters
    ----------
    sts1 : :class:`list[Mol]<list>`
        Set of ligand poses as :class:`~rdkit.Chem.rdchem.Mol` s to compute the MCSS RMSD with `sts2`
    sts2 : :class:`list[Mol]<list>`
        Set of ligand poses as :class:`~rdkit.Chem.rdchem.Mol` s to compute the MCSS RMSD with `sts1`
    processes : int, default=1
        Number of processes to use for computing the pairwise features, if -1 then use all available cores

    Returns
    -------
    :class:`~numpy.ndarray`
        (# poses in sts1) x (# poses in sts2) of the computed MCSS RMSDs. -1 indicates no RMSD was calculated.

    See Also
    --------
    mcss : non-parallelized
    """

    memo = {}
    # sts1 = [merge_halogens(st.copy()) for st in sts1]
    # sts2 = [merge_halogens(st.copy()) for st in sts2]

    unfinished = []
    for j, st1 in enumerate(sts1):
        n_st1_atoms = st1.GetNumHeavyAtoms()
        sma1 = Chem.MolToSmarts(st1)
        # rmsds += [np.zeros(len(sts2))]
        for i, st2 in enumerate(sts2):
            if i > j:  # only calculate lower triangle
                break
            n_st2_atoms = st2.GetNumHeavyAtoms()
            sma2 = Chem.MolToSmarts(st2)
            if (sma1, sma2) in memo:
                mcss, n_mcss_atoms, keep_idxs = memo[(sma1, sma2)]
            else:
                mcss, n_mcss_atoms, keep_idxs = compute_mcss(st1, st2)
                memo[(sma1, sma2)] = (mcss, n_mcss_atoms, keep_idxs)
                memo[(sma2, sma1)] = (mcss, n_mcss_atoms, {'st1':keep_idxs['st2'],'st2':keep_idxs['st1']})

            retain_inf = False
            if (2*n_mcss_atoms < min(n_st1_atoms, n_st2_atoms)):
                # or n_mcss_atoms <= 10):
                retain_inf = True
            unfinished += [(st1,i,st2,j,keep_idxs,retain_inf)]

    results = mp(compute_mcss_rmsd_mp,unfinished,processes)

    rows, cols, values = zip(*results)
    # row_col_val = np.array([rows,cols,values])
    # non_inf = row_col_val[row_col_val[:,2] >= 0]
    # inf_vals = row_col_val[row_col_val[:,2] < 0][:,:-1]
    rmsds_bottom = np.zeros((len(sts1),len(sts2)))
    # rmsds_bottom[non_inf[:,0].astype(np.int64),non_inf[:,1].astype(np.int64)] = non_inf[:,2]
    # full_simi_mat[inf_vals[:,0].astype(np.int64),inf_vals[:,1].astype(np.int64)] = -1 #float('inf')
    rmsds_bottom[rows,cols] = values
    full_simi_mat = rmsds_bottom + rmsds_bottom.T - np.diag(np.diag(rmsds_bottom))
    # full_simi_mat[inf_vals[:,1].astype(np.int64),inf_vals[:,0].astype(np.int64)] = -1 #float('inf')
    return np.where(full_simi_mat<0,np.inf,full_simi_mat)

def compute_mcss_rmsd(st1, st2, keep_idxs, names=True):
    """
    Compute minimum rmsd between mcss(s).

    Takes into account that the mcss smarts pattern could
    map to multiple atom indices (e.g. symetric groups).

    Parameters
    ----------
    st1 : :class:`~rdkit.Chem.rdchem.Mol`
        Molecule 1
    st2 : :class:`~rdkit.Chem.rdchem.Mol`
        Molecule 2
    keep_idxs : dict
        Atom indices to keep for each molecule with keys named the same as the variables
    names : bool,default=True
        Give the created sub-molecules the same name as the original molecules (helps with errors)

    Returns
    -------
    float
        Minimum RMSD between the MCSS of `st1` and `st2`

    See Also
    --------
    compute_mcss_rmsd_mp : used during multiprocessing
    """
    rmsd = float('inf')
    for kpatom_idx1 in keep_idxs['st1']:
        ss1 = subMol(st1,kpatom_idx1)
        if names:
            ss1.SetProp('_Name', st1.GetProp('_Name'))
        for kpatom_idx2 in keep_idxs['st2']:
            ss2 = subMol(st2,kpatom_idx2)
            if names:
                ss2.SetProp('_Name', st2.GetProp('_Name'))
            _rmsd = calculate_rmsd(ss1, ss2)
            rmsd = min(_rmsd, rmsd)
    return rmsd

def compute_mcss_rmsd_mp(st1, i, st2, j, keep_idxs, retain_inf):
    """
    Compute minimum rmsd between mcss(s).

    Used during multiprocessing

    Parameters
    ----------
    st1 : :class:`~rdkit.Chem.rdchem.Mol`
        Molecule 1
    i : int
        array index of molecule 1
    st2 : :class:`~rdkit.Chem.rdchem.Mol`
        Molecule 2
    j : int
        array index of molecule 2
    keep_idxs : dict
        Atom indices to keep for each molecule with keys named the same as the variables
    retain_inf : bool
        Do not calculate RMSD and set RMSD as -1

    Returns
    -------
    float
        Minimum RMSD between the MCSS of `st1` and `st2`

    See Also
    --------
    compute_mcss_rmsd : not used with multiprocessing
    """

    if retain_inf:
        rmsd = -1
    else:
        rmsd = compute_mcss_rmsd(st1,st2,keep_idxs, names=False)
    return (i,j, rmsd)

def get_info_from_results(mcss_res):
    """
    Check the results of `~rdkit.Chem.rdFMCS.FindMCS` to check if finished successfully.

    If MCSS not found, then MCSS is returned as one carbon atom
    
    Parameters
    ----------
    mcss_res : :class:`~rdkit.Chem.rdFMCS.MCSResult`
        Value returned after computing the MCSS
    
    Returns
    -------
    str
        SMARTS string of the MCSS
    int
        Number of atoms in the MCSS
    :class:`~rdkit.Chem.rdchem.Mol`
        A molecule representing the MCSS
    
    """
    
    if not mcss_res.canceled:
        mcss = mcss_res.smartsString
        num_atoms = mcss_res.numAtoms
    else:
        mcss = '[#6]'
        num_atoms = 1
    mcss_mol = Chem.MolFromSmarts(mcss)
    return mcss, num_atoms, mcss_mol

def compute_mcss(st1, st2):
    """
    Compute SMARTS patterns for MCSS(s) between two structures.

    Parameters
    ----------
    st1 : :class:`~rdkit.Chem.rdchem.Mol`
        Molecule 1
    st2 : :class:`~rdkit.Chem.rdchem.Mol`
        Molecule 2

    Returns
    -------
    str
        SMARTS string of the MCSS
    int
        Number of atoms in the MCSS
    dict
        Which atom indices of each molecule are included in the MCSS

    """
    try:
        res = rdFMCS.FindMCS([st1,st2], ringMatchesRingOnly=True,
                completeRingsOnly=True, bondCompare=rdFMCS.BondCompare.CompareOrderExact)
        mcss, num_atoms, mcss_mol = get_info_from_results(res)
        pose1 = subMol(st1,st1.GetSubstructMatch(mcss_mol))
        pose2 = subMol(st2,st2.GetSubstructMatch(mcss_mol))
        assert pose1.HasSubstructMatch(pose2) or pose2.HasSubstructMatch(pose1)
    except AssertionError:
        # some pesky problem ligands (see SKY vs LEW on rcsb) get around default ringComparison
        # but this is slow, so only should do it when we need to do it (but checking is also slow)
        newres = rdFMCS.FindMCS([st1,st2], ringMatchesRingOnly=True,
                completeRingsOnly=True, bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                ringCompare=rdFMCS.RingCompare.PermissiveRingFusion)
        mcss, num_atoms, mcss_mol = get_info_from_results(newres)
    substruct_idx = {'st1': st1.GetSubstructMatches(mcss_mol),
                    'st2': st2.GetSubstructMatches(mcss_mol)}


    return mcss, num_atoms, substruct_idx#, rmv_idx

def calculate_rmsd(pose1, pose2):
    """
    Calculates the RMSD between the two input molecules. Symmetry of molecules is respected during the RMSD calculation.

    Parameters
    ----------
    pose1 : :class:`~rdkit.Chem.rdchem.Mol`
        Molecule 1
    pose2 : :class:`~rdkit.Chem.rdchem.Mol`
        Molecule 2

    Returns
    -------
    float
        RMSD between the two molecules
    """
    assert pose1.HasSubstructMatch(pose2) or pose2.HasSubstructMatch(pose1), f"{pose1.GetProp('_Name')}&{pose2.GetProp('_Name')}"
    try:
        rmsd = Chem.CalcRMS(pose1,pose2)
    except:
        try:
            rmsd = Chem.CalcRMS(pose2,pose1)
        except:
            print(f"{pose1.GetProp('_Name')} and {pose2.GetProp('_Name')}, CalcRMS doesn't work either way")
    return rmsd

# def merge_halogens(structure):
#     """
#     Sets atomic number for all halogens to be that for flourine.
#     This enable use of ConformerRmsd for atom typing schemes that
#     merge halogens.
#     """
#     for atom in structure.atom:
#         if atom.atomic_number in [9, 17, 35, 53]:
#             atom.atomic_number = 9
#     return structure

def subMol(mol, match):
    """
    Get a substructure, as a molecule, of a molecule given the atom indices of the substructure

    Parameters
    ----------
    mol : :class:`~rdkit.Chem.rdchem.Mol`
        Molecule to get a substructure of
    match : :class:`list[int]<list>`
        Atom indices of the substructure
    
    Returns
    -------
    :class:`~rdkit.Chem.rdchem.Mol`
        Substructure of `mol`
    """

    #not sure why this functionality isn't implemented natively
    #but get the interconnected bonds for the match
    atoms = set(match)
    bonds = set()
    for a in atoms:
        atom = mol.GetAtomWithIdx(a)
        for b in atom.GetBonds():
            if b.GetOtherAtomIdx(a) in atoms:
                bonds.add(b.GetIdx())
    return Chem.PathToSubmol(mol,list(bonds))

# def get_substructure(mol, remove_idxs):
#     """
#     Gets the substructure of mol by removing the atoms with indices
#     in remove_idxs
#     """
#     rw_mol = Chem.RWMol(mol)
#     for idx in sorted(remove_idxs,reverse=True):
#         rw_mol.RemoveAtom(idx)

#     assert rw_mol.GetNumAtoms() == (mol.GetNumAtoms() - len(remove_idxs))

#     substruct = Chem.Mol(rw_mol)
#     Chem.SanitizeMol(substruct)
#     return substruct

# def mcss_to_rmv_idx(mol, mcss_mol):
#     """
#     Finds the atom indices that need to be removed from mol
#     to be left with the mcss
#     """
#     mol_mcss = mol.GetSubstructMatches(mcss_mol)
#     mol_fidx = set(range(mol.GetNumAtoms()))
#     remove_idxs = []
#     for keep_idx in mol_mcss:
#         remove_idxs.append(sorted(list(mol_fidx - set(keep_idx)),reverse=True))

#     return remove_idxs
