"""
Compute interaction fingerprints for poseviewer files.
"""

import tempfile
import os
import re
import click
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmarts,ForwardSDMolSupplier,MolFromPDBFile, AddHs
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField, OptimizeMolecule
import gzip

def resname(atom):
    """
    Return the residue name of an atom.

    Parameters
    ----------
    atom : :class:`~rdkit.Chem.rdchem.Atom`
        The atom to get the residue name of.

    Returns
    -------
    str
        The residue name of the atom.
    """

    info = atom.GetPDBResidueInfo()
    if info is None:
        return ''
    return ':'.join(map(lambda x: str(x).strip(),
                        [info.GetChainId(), str(info.GetResidueNumber()),
                         info.GetResidueName(), info.GetInsertionCode()]))

def atomname(atom):
    """
    Return the atom name of an atom.

    Parameters
    ----------
    atom : :class:`~rdkit.Chem.rdchem.Atom`
        The atom to get the atom name of.

    Returns
    -------
    str
        The atom name of the atom.
    """

    pdb = atom.GetPDBResidueInfo()
    if pdb is None:
        return str(atom.GetIdx())
    return pdb.GetName().strip()

def coords(atom):
    """
    Return the coordinates of an atom.

    Parameters
    ----------
    atom : :class:`~rdkit.Chem.rdchem.Atom`
        The atom to get the coordinates of.

    Returns
    -------
    :class:`~rdkit.Geometry.rdGeometry.Point3D`
        The coordinates of the atom.
    """
    return atom.GetOwningMol().GetConformer(0).GetAtomPosition(atom.GetIdx())

def centroid_coords(atoms):
    """
    Return the centroid coordinates of a list of atoms.

    Parameters
    ----------
    atoms : :class:`list[Atom]<list>`
        The list of :class:`~rdkit.Chem.rdchem.Atom`s to get the centroid coordinates of.

    Returns
    -------
    :class:`~rdkit.Geometry.rdGeometry.Point3D`
        The centroid coordinates of the atoms.
    """

    _coords = np.array([coords(atom) for atom in atoms])
    _coords = _coords.mean(axis=0)
    return _coords

def distance(atom1, atom2):
    """
    Return the distance between two atoms.

    Parameters
    ----------
    atom1 : :class:`~rdkit.Chem.rdchem.Atom`
        The first atom.
    atom2 : :class:`~rdkit.Chem.rdchem.Atom`
        The second atom.

    Returns
    -------
    float
        The distance between the two atoms.
    """

    return coords(atom1).Distance(coords(atom2))

def angle_atom(atom1, atom2, atom3):
    """
    Return the angle between three atoms.

    Parameters
    ----------
    atom1 : :class:`~rdkit.Chem.rdchem.Atom`
        The first atom.
    atom2 : :class:`~rdkit.Chem.rdchem.Atom`
        The second atom.
    atom3 : :class:`~rdkit.Chem.rdchem.Atom`
        The third atom.

    Returns
    -------
    float
        The angle between the three atoms in degrees.
    """

    v1 = coords(atom1) - coords(atom2)
    v3 = coords(atom3) - coords(atom2)
    return v1.AngleTo(v3) * 180.0 / np.pi

def angle_vector(v1, v2):
    """
    Return the angle between two vectors.

    Parameters
    ----------
    v1 : :class:`~rdkit.Geometry.rdGeometry.Point3D`
        The first vector.
    v2 : :class:`~rdkit.Geometry.rdGeometry.Point3D`
        The second vector.

    Returns
    -------
    float
        The angle between the two vectors in degrees.
    """

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    angle =  np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle *= 180 / np.pi
    if angle > 90:
        angle = 180 - angle
    assert 0 <= angle <= 90, angle
    return angle

class Molecule:
    """
    A class to represent a molecule.

    Attributes
    ----------
    mol : :class:`~rdkit.Chem.rdchem.Mol`
        The molecule.
    is_protein : bool
        Whether the molecule is a protein.
    settings : dict
        The settings for the molecule.
    contacts : :class:`list[dict]<list>`
        The contacts of the molecule.
    hbond_donors : :class:`list[dict]<list>`
        The hydrogen bond donors of the molecule.
    hbond_acceptors : :class:`list[dict]<list>`
        The hydrogen bond acceptors of the molecule.
    charged : :class:`list[dict]<list>`
        The charged atoms of the molecule.
    charge_groups : :class:`list[dict]<list>`
        The charge groups of the molecule.
    """

    def __init__(self, mol, is_protein, settings):
        """
        Parameters
        ----------
        mol : :class:`~rdkit.Chem.rdchem.Mol`
            The molecule.
        is_protein : bool
            Whether the molecule is a protein.
        settings : dict
            The settings for the molecule.
        """

        self.is_protein = is_protein
        self.mol = mol if self.is_protein else self.init_hydrogens(mol)
        self.settings = settings

        self.contacts = self.init_contacts()
        self.hbond_donors, self.hbond_acceptors = self.init_hbond()
        self.charged, self.charge_groups = self.init_saltbridge()

    def init_hydrogens(self, mol, optimize=True):
        """
        Add hydrogens to a molecule.

        Parameters
        ----------
        mol : :class:`~rdkit.Chem.rdchem.Mol`
            The molecule to add hydrogens to.
        optimize : bool
            Whether to optimize the hydrogens with UFF.

        Returns
        -------
        :class:`~rdkit.Chem.rdchem.Mol`
            The molecule with hydrogens added.
        """

        mol_w_Hs = AddHs(mol,addCoords=True)
        if optimize:
            try:
                ff = UFFGetMoleculeForceField(mol_w_Hs)
                for atom in mol_w_Hs.GetAtoms():
                    if atom.GetAtomicNum() > 1:
                        ff.AddFixedPoint(atom.GetIdx())
                OptimizeMolecule(ff)
            except:
                print(f"couldn't optimize hydrogens for {mol.GetProp('_Name')}")
        return mol_w_Hs

    def init_contacts(self):
        """
        Determine the atoms of the molecule that can be involced in hydrophobic contacts
        with other molecules.

        Returns
        -------
        coord: :class:`~numpy.ndarray`
            The coordinates of the atoms that can be involved in hydrophobic contacts.
        vdw: :class:`~numpy.ndarray`
            The van der Waals radii of the atoms that can be involved in hydrophobic contacts.
        res_name: :class:`list[str]<list>`
            The residue names of the atoms that can be involved in hydrophobic contacts.
        atom_name: :class:`list[str]<list>`
            The atom names of the atoms that can be involved in hydrophobic contacts.
        """

        coord, vdw, atom_name, res_name = [], [], [], []
        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() not in self.settings['nonpolar']: continue
            coord += [coords(atom)]
            atom_name += [atomname(atom)]
            res_name += [resname(atom)]
            vdw += [self.settings['nonpolar'][atom.GetAtomicNum()]]
        
        if coord:
            coord = np.vstack(coord)
            vdw = np.array(vdw)
        return coord, vdw, res_name, atom_name

    def get_aromatic_rings(self):
        """
        Get the aromatic rings in the molecule.

        Returns
        -------
        :class:`list[list[int]]<list>`
            The aromatic rings in the molecule.
        """
        return [ring for ring in self.mol.GetRingInfo().AtomRings()
                if self.mol.GetAtomWithIdx(ring[0]).GetIsAromatic()]

    def get_centroid(self, atom_idx):
        """
        Get the centroid of a set of atoms.
        
        Parameters
        ----------
        atom_idx : :class:`list[int]<list>`
            The indices of the atoms to get the centroid of.

        Returns
        -------
        :class:`~rdkit.Geometry.rdGeometry.Point3D`
            The centroid coordinates of the atoms.
        """

        atoms = [self.mol.GetAtomWithIdx(a) for a in atom_idx]
        return centroid_coords(atoms)

    def get_normal(self, ring):
        """
        Get the normal vector of a ring.

        Parameters
        ----------
        ring : :class:`list[int]<list>`
            The indices of the atoms in the ring.

        Returns
        -------
        :class:`~numpy.ndarray`
            The normal vector of the ring.
        """

        centroid = self.get_centroid(ring)
        coords1 = coords(self.mol.GetAtomWithIdx(ring[0])) - centroid
        coords2 = coords(self.mol.GetAtomWithIdx(ring[1])) - centroid

        normal = np.cross(coords1, coords2)
        normal /= np.linalg.norm(normal)
        return normal

    def init_hbond(self):
        """
        Determine the hydrogen bond donors and acceptors of the molecule.

        Returns
        -------
        donors: :class:`list[Atom]<list>`
            The hydrogen bond donors of the molecule.
        acceptors: :class:`list[Atom]<list>`
            The hydrogen bond acceptors of the molecule.
        """

        donors = [atom for atom in self.mol.GetAtoms() if self._is_donor(atom)]
        acceptors = [atom for atom in self.mol.GetAtoms() if self._is_acceptor(atom)]
        return donors, acceptors

    def _is_donor(self, atom):
        """
        Determine whether an atom is a hydrogen bond donor.

        Parameters
        ----------
        atom : :class:`Atom`
            The atom to check.

        Returns
        -------
        bool
            Whether the atom is a hydrogen bond donor.
        """

        if atom.GetAtomicNum() in [7, 8]:
            if _get_bonded_hydrogens(atom):
                return True
        return False

    def _is_acceptor(self, atom):
        """
        Determine whether an atom is a hydrogen bond acceptor.

        Parameters
        ----------
        atom : :class:`Atom`
            The atom to check.

        Returns
        -------
        bool
            Whether the atom is a hydrogen bond acceptor.
        """

        if atom.GetAtomicNum() == 8:
            return True
        if atom.GetAtomicNum() == 7 and atom.GetExplicitValence() < 4:
            return True
        return False

    def init_saltbridge(self):
        """
        Determine the charged atoms and charge groups of the molecule.

        Returns
        -------
        charged: :class:`list[Atom]<list>`
            The charged atoms of the molecule.
        charge_groups: :class:`list[list[Atom]]<list>`
            The charge groups of the molecule.
        """

        charged = [atom for atom in self.mol.GetAtoms()
                   if atom.GetFormalCharge() != 0]
        if self.is_protein:
            charge_groups = self._charged_protein_atoms()
        else:
            charge_groups = self._symmetric_charged_ligand_atoms()
        return charged, charge_groups

    def _charged_protein_atoms(self):
        """
        Determine the charged atoms of the protein.

        Returns
        -------
        :class:`dict[str, list[Atom]]<dict>`
            The charged atoms of the protein.
        """

        protein_groups = {}
        for protein_atom in self.mol.GetAtoms():
            if atomname(protein_atom) in ['OD1', 'OD2', 'OE1', 'OE2', 'NH1', 'NH2','NZ','ND1']:
                # residue_name = resname(protein_atom).split(':')[2]
                # print(residue_name)
                # if residue_name in ['ARG','HIS','LYS','ASP','GLU']:
                #     print('in')
                # else:
                #     print(

                if resname(protein_atom) not in protein_groups:
                    protein_groups[resname(protein_atom)] = []
                protein_groups[(resname(protein_atom))] += [protein_atom]
        return protein_groups

    def _symmetric_charged_ligand_atoms(self):
        """
        Determine the charged atoms of the ligand.

        Returns
        -------
        :class:`dict[str, list[Atom]]<dict>`
            The charged atoms of the ligand.
        """
        ligand_groups = {}
        # smartss = [('[CX3](=O)[O-,OH]', 2, [1, 2]),
        #            ('[CX3](=[NH2X3+])[NH2X3]', 1, [1, 2])]
        # These are pulled from pharmit's positive and negative ions
        pos_smartss = [('[+,+2,+3,+4]', 0, [0]), #positive ions first
                        ('[$(C*)](=,-N)N', 0, [1, 2]),
                        ('C(N)(N)=N', 0, [1, 2,3]),
                        ('[nH]1cncc1', 0, [0, 2])]
        neg_smartss = [('[-,-2,-3,-4]', 0, [0]),
                        ('[S,P,C](=O)[O-,OH,OX1]', 0, [1, 2]),
                        ('c1[nH1]nnn1', 0, [1, 3]),
                        ('c1nn[nH1]n1', 0, [1, 3]),
                        ('C(=O)N[OH1,O-,OX1]', 0, [1, 2]),
                        ('CO(=N[OH1,O-])', 0, [1, 2]),
                        ('[$(N-[SX4](=O)(=O)[CX4](F)(F)F)]', 0, [1, 2])]

        smartss = [(MolFromSmarts(ss), k, v, +1) for ss, k, v in pos_smartss] + [(MolFromSmarts(ss), k, v, -1) for ss, k, v in neg_smartss]
        idx_to_atom = {atom.GetIdx(): atom for atom in self.mol.GetAtoms()}

        for smarts, k, v, charge in smartss:
            matches = self.mol.GetSubstructMatches(smarts)
            # if len(matches):
            #     print(smarts)
            for match in matches:
                ligand_groups[match[k]] = ([idx_to_atom[match[_v]] for _v in v], charge)
        return ligand_groups

################################################################################
# Compute atom-level interactions

def _get_bonded_hydrogens(atom):
    """
    Get the hydrogens bonded to an atom.

    Parameters
    ----------
    atom : :class:`Atom`
        The atom to check.

    Returns
    -------
    :class:`list[Atom]<list>`
        The hydrogens bonded to the atom.
    """

    hydrogens = []
    for bond in atom.GetBonds():
        if bond.GetBeginAtomIdx() != atom.GetIdx():
            hydrogen = bond.GetBeginAtom()
        else:
            hydrogen = bond.GetEndAtom()
            
        if hydrogen.GetAtomicNum() == 1:
            hydrogens += [hydrogen]
    return hydrogens

def _hbond_hydrogen_angle(acceptor, donor):
    """
    Finds the hydrogen that maximizes the angle of the acceptor-donor bond.

    Parameters
    ----------
    acceptor : :class:`~rdkit.Chem.rdchem.Atom`
        The acceptor atom.
    donor : :class:`~rdkit.Chem.rdchem.Atom`
        The donor atom.

    Returns
    -------
    :class:`~rdkit.Chem.rdchem.Atom`
        The hydrogen that maximizes the angle of the acceptor-donor bond.
    float
        The angle of the acceptor-donor bond.
    """

    best_angle, best_hydrogen = 0, None
    for hydrogen in _get_bonded_hydrogens(donor):
        _angle = angle_atom(donor, hydrogen, acceptor)
        if _angle > best_angle:
            best_angle = _angle
            best_hydrogen = hydrogen
    return best_hydrogen, best_angle

def _hbond_compute(donor_mol, acceptor_mol, settings, protein_is_donor):
    """
    Compute the hydrogen bonds between a donor and acceptor molecule.

    Parameters
    ----------
    donor_mol : :class:`~open_combind.features.ifp.Molecule`
        The donor molecule.
    acceptor_mol : :class:`~open_combind.features.ifp.Molecule`
        The acceptor molecule.
    settings : dict
        The settings for the hydrogen bond calculation.
    protein_is_donor : bool
        Whether the protein is the donor.

    Returns
    -------
    :class:`list[dict]<list>`
        The hydrogen bonds between the donor and acceptor molecule.
    """

    hbonds = []
    for donor in donor_mol.hbond_donors:
        for acceptor in acceptor_mol.hbond_acceptors:
            for hydrogen in _get_bonded_hydrogens(donor):
                dist = distance(acceptor, hydrogen)
                if dist > settings['hbond_dist_cut']: continue
                angle = angle_atom(donor, hydrogen, acceptor)
                if angle < settings['hbond_angle_cut']: continue

                if protein_is_donor:
                    label = 'hbond_donor'
                    protein_atom = donor
                    ligand_atom = acceptor
                else:
                    label = 'hbond_acceptor'
                    protein_atom = acceptor
                    ligand_atom = donor

                hbonds += [{'label': label,
                            'protein_res': resname(protein_atom),
                            'protein_atom': atomname(protein_atom),
                            'ligand_atom': atomname(ligand_atom),
                            'dist': dist,
                            'angle': angle,
                            'hydrogen': atomname(hydrogen)}]
    return hbonds

def hbond_compute(protein, ligand, settings):
    """
    Identify and log the hydrogen bonds between the protein and ligand atoms

    Parameters
    ----------
    protein : :class:`~open_combind.features.ifp.Molecule`
        Protein molecule
    ligand : :class:`~open_combind.features.ifp.Molecule`
        Ligand molecule
    settings : dict
        Settings for determining the presence of interactions
    Returns
    -------
    dict
        dict containing all of the hydrogen bonds found between the protein and ligand atoms
    """
    
    donor = _hbond_compute(protein, ligand, settings, True)
    acceptor = _hbond_compute(ligand, protein, settings, False)
    return acceptor + donor

def saltbridge_compute(protein, ligand, settings):
    """
    Identify and log the salt bridges between the protein and ligand atoms
    
    Parameters
    ----------
    protein : :class:`~open_combind.features.ifp.Molecule`
        Protein molecule
    ligand : :class:`~open_combind.features.ifp.Molecule`
        Ligand molecule
    settings : dict
        Settings for determining the presence of interactions

    Returns
    -------
    dict
        dict containing all of the salt bridges found between the protein and ligand atoms
    """
    # Note that much of the complexity here stems from taking into account
    # symetric atoms. Specifically for carboxylate and guanidinium groups,
    # we consider not just the atom that is arbitrarily assigned a formal
    # charge, but also the atom that is charged in the other resonance
    # structure.

    # print(protein.charged)
    saltbridges = []
    for residue_name, protein_atoms in protein.charge_groups.items():
        for ligand_atoms, lig_charge in ligand.charge_groups.values():
            residue_id = resname(protein_atoms[0]).split(':')[2]
            protein_charge = 1 * (residue_id in ['LYS','ARG','HIS']) - 1 * (residue_id in ['ASP','GLU'])
            if lig_charge * protein_charge >= 0: continue

            dist = float('inf')
            for _ligand_atom in ligand_atoms:
                for _protein_atom in protein_atoms:
                    _dist = distance(_protein_atom, _ligand_atom)
                    if _dist < dist:
                        dist = _dist
                        closest_protein_atom = _protein_atom
                        closest_ligand_atom = _ligand_atom

            # print(dist)
            if dist < settings['sb_dist_cut']:
                saltbridges += [{'label': 'saltbridge',
                                 'protein_res': resname(closest_protein_atom),
                                 'protein_atom': atomname(closest_protein_atom),
                                 'ligand_atom': atomname(closest_ligand_atom),
                                 'dist': dist}]
    return saltbridges

def contact_compute(protein, ligand, settings):
    """
    Identify and log the hydrophobic contacts between the protein and ligand atoms
    
    Parameters
    ----------
    protein : :class:`~open_combind.features.ifp.Molecule`
        Protein molecule
    ligand : :class:`~open_combind.features.ifp.Molecule`
        Ligand molecule
    settings : dict
        Settings for determining the presence of interactions

    Returns
    -------
    dict
        dict containing all of the hydrophobic contacts found between the protein and ligand atoms
    """
    
    protein = protein.contacts
    ligand = ligand.contacts

    dists = protein[0].reshape(1, -1, 3) - ligand[0].reshape(-1, 1, 3)
    dists = np.linalg.norm(dists, axis=2)
    vdw = protein[1].reshape(1, -1) + ligand[1].reshape(-1, 1)
    contact_idx = np.argwhere(dists < vdw*settings['contact_scale_cut'])

    contacts = []
    for i, j in contact_idx:
        contacts += [{'label': 'contact',
                      'protein_res': protein[2][j],
                      'protein_atom': protein[3][j],
                      'ligand_atom': ligand[3][i],
                      'dist': dists[i, j],
                      'vdw': vdw[i, j]}]
    return contacts

################################################################################
# Compute residue-level scores.

def _piecewise(data, opt, cut):
    """
    Piecewise linear function that is 0 below cut, 1 above opt, and linear in between.
    
    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        Array of data to be transformed
    opt : float
        Optimal value
    cut : float
        Cutoff value

    Returns
    -------
    :class:`~numpy.ndarray`
        Transformed data
    """

    slope = 1 / (cut-opt)
    intercept = cut * slope

    data = intercept - slope * data
    data[data > 1] = 1
    data[data < 0] = 0
    return data

def _groupby_subset(df, index, col):
    """
    Group a dataframe by a subset of its columns

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Dataframe to be grouped
    index : list
        List of columns to group by
    col : str
        Column to be grouped

    Returns
    -------
    :class:`~pandas.DataFrameGroupBy`
        Grouped dataframe
    """

    return df[index+[col]].groupby(index)

def nodigits(s):
    """
    Remove digits from a string

    Parameters
    ----------
    s : str
        String to remove digits from

    Returns
    -------
    str
        String with digits removed
    """

    return ''.join([i for i in s if not i.isdigit()])

def compute_scores(raw, settings):
    """
    Given atomic level interactions between the protein and ligand compute the compute the residue level interactions.

    Parameters
    ----------
    raw : :class:`~pandas.DataFrame`
        Atomic level interactions between the protein and ligand
    settings : :class:`~pandas.DataFrame`
        Interaction fingerprint settings

    Returns
    -------
    :class:`~pandas.DataFrame`
        Residue level interactions between the protein and ligand

    """
    
    # print("computing scores")
    if settings['level'] == 'atom':
        raw['protein_res'] = [r['protein_res']+':'+nodigits(r['protein_atom'])
                              for _, r in raw.iterrows()]

    scores = []
    for label, group in raw.groupby('label'):
        group = group.copy()
        if label == 'contact':
            group['score'] = _piecewise(group['dist'] / group['vdw'],
                                        settings['contact_scale_opt'],
                                        settings['contact_scale_cut'])
        elif label == 'saltbridge':
            group['score'] = _piecewise(group['dist'],
                                        settings['sb_dist_opt'],
                                        settings['sb_dist_cut'])
        elif label in ['hbond_donor', 'hbond_acceptor']:
            group['score'] = (  _piecewise(group['dist'],
                                           settings['hbond_dist_opt'],
                                           settings['hbond_dist_cut'])
                              * _piecewise(180 - group['angle'],
                                           settings['hbond_angle_opt'],
                                           settings['hbond_angle_cut']))

            # One hydrogen bond per hydrogen
            if label == 'hbond_donor':
                idx = _groupby_subset(group,
                                      ['pose', 'protein_res', 'hydrogen'],
                                      'score').idxmax()
            else:
                idx = _groupby_subset(group,
                                      ['pose', 'hydrogen'],
                                      'score').idxmax()
            idx = idx['score']
            group = group.loc[idx]
        group = _groupby_subset(group, ['pose', 'label', 'protein_res'], 'score').sum()
        scores += [group]
    return pd.concat(scores).sort_index()

################################################################################

def fingerprint(protein, ligand, settings):
    """
    Determine the atomic level interactions present in a protein-ligand complex. The following interactions are detected:

    * Hydrogen bonding
    * Salt bridges
    * Hydrophobic contacts
    
    Parameters
    ----------
    protein : :class:`~open_combind.features.ifp.Molecule`
        Protein molecule
    ligand : :class:`~open_combind.features.ifp.Molecule`
        Ligand molecule
    settings : dict
        Settings for determining the presence of interactions
    Returns
    -------
    :class:`~pandas.DataFrame`
        Dataframe containing all of the interactions found in the complex
    """
    
    fp  = hbond_compute(protein, ligand, settings)
    fp += saltbridge_compute(protein, ligand, settings)
    fp += contact_compute(protein, ligand, settings)
    return pd.DataFrame.from_dict(fp)

def fingerprint_poseviewer(input_file, poses, settings):
    """
    Compute interaction fingerprint at the atomic level of the all of the ligand poses in `input_file`
    
    Parameters
    ----------
    input_file : str
        Path to the gzipped SDF file containing all of the ligand poses
    poses : int
        Number of poses to read from `input_file`
    settings : dict
        Settings of the interaction fingerprint
    
    Returns
    -------
    :class:`~pandas.DataFrame`
        Interaction fingerprint of each pose
    
    """
    
    prot_bname = input_file.split('-to-')[-1]
    prot_fname = re.sub('-docked.*\.sdf\.gz','_prot.pdb',prot_bname)
    prot_file = f"structures/proteins/{prot_fname}"
    # print(prot_file)

    # print(input_file)
    fps = []
    with gzip.open(input_file) as fp:
        mols = ForwardSDMolSupplier(fp, removeHs=False)
        rdk_prot = MolFromPDBFile(prot_file,removeHs=False)
        if rdk_prot is None:
            rdk_prot = MolFromPDBFile(prot_file,sanitize=False,removeHs=False)
            # print(rdk_prot)
        assert rdk_prot is not None, f"RDKit cannot read protein file {prot_file}"

        protein = Molecule(rdk_prot, True, settings)
        assert protein is not None
        # print(len(protein.charged))
        
        for i, ligand in enumerate(mols):
            if i == poses: break
            if ligand is None:
                print('ligand unreadable')
                continue

            ligand = Molecule(ligand, False, settings)
            fps += [fingerprint(protein, ligand, settings)]
            fps[-1]['pose'] = i

    fps = pd.concat(fps, ignore_index=True, sort=False)
    if 'hydrogen' not in fps:
        fps['hydrogen'] = ''
    fps.loc[fps['hydrogen'].isna(), 'hydrogen'] = ''
    return fps

def ifp(settings, input_file, output_file, poses):
    """
    Compute interaction fingerprint of the all of the ligand poses in `input_file` and save to a CSV
    
    Parameters
    ----------
    settings : dict
        Settings of the interaction fingerprint
    input_file : str
        Path to the gzipped SDF file containing all of the ligand poses
    output_file : str
        Path to the output CSV file containing all the interactions 
    poses : int
        Number of poses to read from `input_file`
    
    """
    
    settings['nonpolar'] = {6:1.7, 9:1.47, 17:1.75, 35:1.85, 53:1.98}

    # Compute atom-level interactions.
    fps = fingerprint_poseviewer(input_file, poses, settings)

    # Compute residue-level scores.
    scores = compute_scores(fps, settings)

    # print("done with scores")
    # Write to files
    fps = fps.set_index(['pose', 'label', 'protein_res', 'protein_atom', 'ligand_atom'])
    fps = fps.sort_index()
    base = output_file.split('.')
    base, ext = base[:-1], base[-1]
    raw_file = '.'.join(base) + '_raw.' + ext

    fps.to_csv(raw_file)
    scores.to_csv(output_file)

@click.command()
@click.argument('input_file')
@click.argument('output_file')
@click.argument('poses', default=100)
@click.option('--level', default='residue')
@click.option('--hbond_dist_cut', default=3.0)
@click.option('--hbond_dist_opt', default=2.5)
@click.option('--hbond_angle_cut', default=90.0)
@click.option('--hbond_angle_opt', default=60.0)
@click.option('--sb_dist_cut', default=5.0)
@click.option('--sb_dist_opt', default=4.0)
@click.option('--contact_scale_cut', default=1.75)
@click.option('--contact_scale_opt', default=1.50)
def main(input_file, output_file, poses, **settings):
    ifp(settings, input_file, output_file, poses)

if __name__ == '__main__':
    main()
