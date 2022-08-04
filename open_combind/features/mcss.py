import tempfile
import numpy as np
import subprocess
import os
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdmolops import ReplaceSubstructs
# from plumbum.cmd import obrms
from open_combind.utils import mp

class CompareHalogens(rdFMCS.MCSAtomCompare):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, p, mol1, atom1, mol2, atom2):
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
    Computes rmsd between mcss for atoms in two poseviewer files.

    Returns a (# poses in pv1) x (# poses in pv2) np.array of rmsds.
    """
    memo = {}
    params = setup_MCS_params()
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
    Computes rmsd between mcss for atoms in two poseviewer files.

    Returns a (# poses in pv1) x (# poses in pv2) np.array of rmsds.
    """
    memo = {}
    
    params = setup_MCS_params()
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
                mcss, n_mcss_atoms, keep_idxs = compute_mcss(st1, st2, params)
                memo[(sma1, sma2)] = (mcss, n_mcss_atoms, keep_idxs)
                memo[(sma2, sma1)] = (mcss, n_mcss_atoms, {'st1': keep_idxs['st2'],'st2':keep_idxs['st1']})

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

    remove_idxs: dictionary with keys 'st1' and 'st2' that have lists
    of atom indices lists to remove to create the MCSS for each pose
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

    Takes into account that the mcss smarts pattern could
    map to multiple atom indices (e.g. symetric groups).

    keep_idxs: dictionary with keys 'st1' and 'st2' that have lists
    of atom indices lists to keep to create the MCSS for each pose
    """
    if retain_inf:
        rmsd = -1
    else:
        rmsd = compute_mcss_rmsd(st1,st2,keep_idxs, names=False)
    return (i,j, rmsd)

def get_info_from_results(mcss_res):
    if not mcss_res.canceled:
        mcss = mcss_res.smartsString
        num_atoms = mcss_res.numAtoms
    else:
        mcss = '[#6]'
        num_atoms = 1
    mcss_mol = Chem.MolFromSmarts(mcss)
    return mcss, num_atoms, mcss_mol

def compute_mcss(st1, st2, params):
    """
    Compute smarts patterns for mcss(s) between two structures.
    """
    try:
        res = rdFMCS.FindMCS([st1,st2], params)
        mcss, num_atoms, mcss_mol = get_info_from_results(res)
        pose1 = subMol(st1,st1.GetSubstructMatch(mcss_mol))
        pose2 = subMol(st2,st2.GetSubstructMatch(mcss_mol))
        assert pose1.HasSubstructMatch(pose2) or pose2.HasSubstructMatch(pose1)
    except AssertionError:
        # some pesky problem ligands (see SKY vs LEW on rcsb) get around default ringComparison
        # but this is slow, so only should do it when we need to do it (but checking is also slow)

        #This is the same as ringCompare=rdFMC.RingCompare.PermissiveRingFusion
        # see https://github.com/rdkit/rdkit/issues/5438
        params.BondCompareParameters.MatchFusedRings = True
        params.BondCompareParameters.MatchFusedRingsStrict = False
        newres = rdFMCS.FindMCS([st1, st2], params)
        params.BondCompareParameters.MatchFusedRings = False
        mcss, num_atoms, mcss_mol = get_info_from_results(newres)
    substruct_idx = {'st1': st1.GetSubstructMatches(mcss_mol),
                    'st2': st2.GetSubstructMatches(mcss_mol)}


    return mcss, num_atoms, substruct_idx#, rmv_idx

def setup_MCS_params():
    params = rdFMCS.MCSParameters()
    params.AtomCompareParameters.RingMatchesRingOnly = True
    params.AtomCompareParameters.CompleteRingsOnly = True
    params.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    params.AtomTyper = CompareHalogens()

    return params

def calculate_rmsd(pose1, pose2, eval_rmsd=False):
    """
    Calculates the RMSD between pose1 and pose2.

    pose1, pose2: rdkit.Mol
    eval_rmsd: verify that RMSD calculation is the same as obrms
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
    This enable use of ConformerRmsd for atom typing schemes that
    merge halogens.
    """
    for atom in structure.atom:
        if atom.atomic_number in [9, 17, 35, 53]:
            atom.atomic_number = 9
    return structure

def subMol(mol, match, merge_halogens=True):
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
                Chem.MolFromSmiles('F'), replaceAll=True)
    return new_mol

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

def mcss_to_rmv_idx(mol, mcss_mol):
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
