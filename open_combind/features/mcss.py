import tempfile
import numpy as np
import subprocess
import os
import itertools
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdmolops import ReplaceSubstructs
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
# from plumbum.cmd import obrms
from open_combind.utils import mp

class CompareHalogens(rdFMCS.MCSAtomCompare):
    """
    Atom comparator for MCS that allows halogens to match with each other.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, p, mol1, atom1, mol2, atom2):
        """
        Checks if two atoms are a match.
        Returns True if they are a match, or if both atoms are halogens.
        Returns False otherwise.
        """
        a1 = mol1.GetAtomWithIdx(atom1)
        a2 = mol2.GetAtomWithIdx(atom2)
        a1_an = a1.GetAtomicNum()
        a2_an = a2.GetAtomicNum()
        if (a1_an != a2_an):
            if (a1_an not in [9, 17, 35, 53]) or (a2_an not in [9, 17, 35, 53]):
                return False
        if (p.MatchValences and a1.GetTotalValence() != a2.GetTotalValence()):
            return False
        if (p.MatchChiralTag and not self.CheckAtomChirality(p, mol1, atom1, mol2, atom2)):
            return False
        if (p.MatchFormalCharge and not self.CheckAtomCharge(p, mol1, atom1, mol2, atom2)):
            return False
        if p.RingMatchesRingOnly:
            return self.CheckAtomRingMatch(p, mol1, atom1, mol2, atom2)
        return True
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
    Computes the RMSD between the maximum common substructure (MCS) between two molecules.

    Parameters
    ----------
    sts1 : list of ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        List of molecules to compare.
    sts2 : list of ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        List of molecules to compare.

    Returns
    -------
    rmsds : np.ndarray
        Matrix of RMSD values between the MCS of each pair of molecules.
    """

    memo = {}
    params = setup_MCS_params()

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
                mcss, n_mcss_atoms, keep_idxs = compute_mcss(st1, st2, params)
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
    Multiprocessing method to compute the RMSD between MCSS for poses in two lists of molecules.

    Parameters
    ----------
    sts1 : list of ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        List of molecules to compare.
    sts2 : list of ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        List of molecules to compare.
    processes : int
        Number of processes to use for computing the MCSS. Default is 1.

    Returns
    -------
    rmsds : np.array
        (# poses in pv1) x (# poses in pv2) np.array of rmsds.
    """
    
    unfinished = []
    mcss_calc_unfinished = []
    group_st1 = group_mols_by_SMARTS(sts1)
    group_st2 = group_mols_by_SMARTS(sts2)
    for g1, g2 in itertools.product(group_st1, group_st2):
        if (g2[0],g2[1],g1[0],g1[1]) in unfinished:
            continue
        mcss_calc_unfinished += [(g1[0][0], g2[0][0])]
        unfinished += [(g1[0],g1[1],g2[0],g2[1])]

    print("calculating mcss first")
    # TODO: make the number of processes for computing the mcss available to the user
    # this should control to some extent, the amount of memory that is used. More processes == more memory
    mcss_results = mp(compute_mcss_mp, mcss_calc_unfinished, min(processes, 3), maxtasksperchild=3)
    keys, vals = zip(*mcss_results)
    global mcss_info
    mcss_info = dict(zip(keys,vals))

    print("now calculating rmsds")
    results = mp(compute_mcss_and_rmsd,unfinished,processes)

    #unpack results into dense matrix
    itr_results = itertools.chain.from_iterable(results)
    rows, cols, values = zip(*itr_results)
    rmsds_bottom = np.zeros((len(sts1),len(sts2)))
    rmsds_bottom[rows,cols] = values
    # matrix is symmetric, so copy over diagonal
    full_simi_mat = rmsds_bottom + rmsds_bottom.T - np.diag(np.diag(rmsds_bottom))
    return np.where(full_simi_mat<0,np.inf,full_simi_mat)

def group_mols_by_SMARTS(mols):
    """
    Group RDKit molecules by SMARTS pattern and return a list of tuples that have the molecules and the indices of the molecules in the original list

    Parameters
    ----------
    mols : list of ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        List of rdkit molecules

    Returns
    -------
    list of tuples
        Each tuple contains a list of molecules, a list of indices, and a SMARTS pattern
    """

    smarts = [Chem.MolToSmarts(mol) for mol in mols]
    unique_smarts = list(set(smarts))
    groups = []
    for us in unique_smarts:
        indices = [i for i, x in enumerate(smarts) if x == us]
        group_mols = [mols[i] for i in indices]
        groups.append((group_mols, indices, us))
    return groups

def compute_mcss_rmsd_mp(mols1, idxs1, mols2, idxs2):
    """
    Compute the RMSD between the MCSS of two ligands with a number of different poses for each ligand

    Parameters
    ----------
    mols1 : list of ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        List of rdkit molecules for the first ligand, specifying the poses of the ligand
    idxs1 : list of int
        List of indices of the poses of the first ligand
    mols2 : list of ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        List of rdkit molecules for the second ligand, specifying the poses of the ligand
    idxs2 : list of int
        List of indices of the poses of the second ligand

    Returns
    -------
    rmsds_bottom : np.array
        A (# poses in pv1) x (# poses in pv2) np.array of rmsds.
    """

    rmsds = []

    n_st1_atoms = mols1[0].GetNumHeavyAtoms()
    n_st2_atoms = mols2[0].GetNumHeavyAtoms()

    smarts1 = Chem.MolToSmarts(mols1[0])
    smarts2 = Chem.MolToSmarts(mols2[0])
    assert (smarts1,smarts2) in mcss_info, f"can't find mcss info the pair: {mols1.GetProp('_Name')} and {mol2.GetProp('_Name')}"
    mcss, n_mcss_atoms, keep_idxs = mcss_info[(smarts1,smarts2)] 
    if (2*n_mcss_atoms < min(n_st1_atoms, n_st2_atoms)):
        # or n_mcss_atoms <= 10):
        for i,j in itertools.product(idxs1,idxs2):
            if i > j:
                rmsds.append((j,i,-1))
            else:
                rmsds.append((i,j,-1))
    else:
        # Get the RMSD for each unique pair with one pose from each group
        for i,j in itertools.product(range(len(mols1)), range(len(mols2))):
            rmsd = compute_mcss_rmsd(mols1[i], mols2[j], keep_idxs, names=False)
            if idxs1[i] > idxs2[j]:
                rmsds.append((idxs2[j], idxs1[i],rmsd))
            else:
                rmsds.append((idxs1[i], idxs2[j], rmsd))

    return rmsds
    

def compute_mcss_rmsd(st1, st2, keep_idxs, names=True):
    """
    Compute minimum RMSD between MCSS(s).

    Parameters
    ----------
    st1 : ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        RDKit molecule for the first ligand
    st2 : ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        RDKit molecule for the second ligand
    keep_idxs : dict
        Dictionary with keys 'st1' and 'st2' that contain lists of indices of atoms to keep in the substructure
    names : bool, optional, default=True
        Whether to keep the names of the molecules

    Returns
    -------
    float
        The minimum RMSD between the MCSS of the two ligands
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

def get_info_from_results(mcss_res):
    """
    Get information from the results of the MCSS calculation.

    Parameters
    ----------
    mcss_res : ` ``RDKit.rdFMCS.MCSResult`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdFMCS.html#rdkit.Chem.rdFMCS.MCSResult>`_
        The results of the MCSS calculation

    Returns
    -------
    str
        The smarts pattern for the MCSS
    int
        The number of heavy atoms in the MCSS
    ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        A molecule that represents the MCSS
    """

    if not mcss_res.canceled:
        mcss = mcss_res.smartsString
        num_atoms = mcss_res.numAtoms
    else:
        mcss = '[#6]'
        num_atoms = 1
    mcss_mol = Chem.MolFromSmarts(mcss)
    return mcss, num_atoms, mcss_mol

def compute_mcss(st1, st2, current_params):
    """
    Compute the MCSS of two ligands.

    Parameters
    ----------
    st1 : ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        RDKit molecule for the first ligand
    st2 : ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        RDKit molecule for the second ligand
    current_params : ` ``RDKit.rdFMCS.MCSParameters`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdFMCS.html#rdkit.Chem.rdFMCS.MCSParameters>`_
        The parameters for the MCSS calculation

    Returns
    -------
    str
        The smarts pattern for the MCSS
    int
        The number of heavy atoms in the MCSS
    dict
        Dictionary with keys 'st1' and 'st2' that contain lists of indices of atoms to keep in the substructure
    """

    try:
        res = rdFMCS.FindMCS([st1,st2], current_params)
        mcss, num_atoms, mcss_mol = get_info_from_results(res)
        if not res.canceled:
            pose1 = subMol(st1,st1.GetSubstructMatch(mcss_mol))
            pose2 = subMol(st2,st2.GetSubstructMatch(mcss_mol))
            assert pose1.HasSubstructMatch(pose2) or pose2.HasSubstructMatch(pose1)
    except AssertionError:
        print("in the assertion")
        # some pesky problem ligands (see SKY vs LEW on rcsb) get around default ringComparison
        # but this is slow, so only should do it when we need to do it (but checking is also slow)

        #This is the same as ringCompare=rdFMC.RingCompare.PermissiveRingFusion
        # see https://github.com/rdkit/rdkit/issues/5438
        current_params.BondCompareParameters.MatchFusedRings = True
        current_params.BondCompareParameters.MatchFusedRingsStrict = False
        newres = rdFMCS.FindMCS([st1, st2], current_params)
        mcss, num_atoms, mcss_mol = get_info_from_results(newres)
        current_params.BondCompareParameters.MatchFusedRings = False
    substruct_idx = {'st1': st1.GetSubstructMatches(mcss_mol),
                    'st2': st2.GetSubstructMatches(mcss_mol)}

    return mcss, num_atoms, substruct_idx#, rmv_idx

def compute_mcss_mp(st1, st2):
    """
    Multiprocessing wrapper for MCSS computation

    st1, st2: rdkit.Mol

    returns: ((smarts1, smarts2), (mcss, num_atoms, substruct_idx))
    """

    p = setup_MCS_params()
    mcss, num_atoms, substruct_idx = compute_mcss(st1,st2, p)
    return ((Chem.MolToSmarts(st1),Chem.MolToSmarts(st2)), (mcss, num_atoms, substruct_idx))

def setup_MCS_params():
    """
    Setup MCS parameters.

    Returns
    -------
    ` ``RDKit.rdFMCS.MCSParameters`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdFMCS.html#rdkit.Chem.rdFMCS.MCSParameters>`_
        The parameters for the MCSS calculation
    """

    params = rdFMCS.MCSParameters()
    params.AtomCompareParameters.RingMatchesRingOnly = True
    params.AtomCompareParameters.CompleteRingsOnly = True
    params.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    params.AtomTyper = CompareHalogens()
    params.Timeout = 420

    return params

def calculate_rmsd(pose1, pose2, eval_rmsd=False):
    """
    Calculates the RMSD between pose1 and pose2.

    Parameters
    ----------
    pose1 : ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        RDKit molecule for the first ligand
    pose2 : ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        RDKit molecule for the second ligand
    eval_rmsd : bool, optional, default=False
        Whether to evaluate the RMSD using the OpenBabel implementation .. .deprecated::
    """

    assert pose1.HasSubstructMatch(pose2) or pose2.HasSubstructMatch(pose1), f"{pose1.GetProp('_Name')}&{pose2.GetProp('_Name')}"
    try:
        rmsd = Chem.CalcRMS(pose1,pose2)
    except:
        try:
            rmsd = Chem.CalcRMS(pose2,pose1)
        except:
            print(f"{pose1.GetProp('_Name')} and {pose2.GetProp('_Name')}, CalcRMS doesn't work either way")
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

    Parameters
    ----------
    structure : ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        RDKit molecule

    Returns
    -------
    structure : ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        RDKit molecule with halogens replaced with fluorine
    """

    for atom in structure.atom:
        if atom.atomic_number in [9, 17, 35, 53]:
            atom.atomic_number = 9
    return structure

def subMol(mol, match, merge_halogens=True):
    """
    Returns a substructure of mol that contains the atoms in match
    and the bonds between them.

    Parameters
    ----------
    mol : ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        RDKit molecule
    match : list of int
        Indices of atoms in mol to include in the substructure
    merge_halogens : bool, optional, default=True
        Whether to merge halogens into fluorine

    Returns
    -------
    new_mol : ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        Substructure of mol
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
    new_mol = Chem.PathToSubmol(mol, list(bonds))
    if merge_halogens:
        new_mol = ReplaceSubstructs(new_mol, Chem.MolFromSmarts('[F,Cl,Br,I]'),
                Chem.MolFromSmiles('F'), replaceAll=True)[0]
    return new_mol
