import fileinput
from pymol import cmd
from rdkit import Chem
import sys
import pandas as pd
from glob import glob

def load_complexes(protein, n=1):
    """
    USAGE

    load_complexes protein,[n=1]

    loads the first n protein ligand complexes from the directory [protein].
    """
    n = int(n)
    for prot in sorted(glob('{}/structures/proteins/*_prot.pdb'.format(protein)))[:n]:
        pdb = prot.split('/')[-1].split('_')[0]
        load_crystal_protein(protein, pdb)
        load_crystal_pose(protein, pdb)
        
    cmd.util.cbao("prot_*")
    cmd.util.cbay("het and crystal_*")
    cmd.show('sticks', "het and crystal_*")
    cmd.hide('lines', 'element h')
    cmd.show('spheres', 'het and prot* and (crystal* expand 5)')

    cmd.show('cartoon')
    cmd.set('cartoon_oval_length', '0.5')
    cmd.set('cartoon_transparency', '0.5')
    cmd.hide('everything', 'element H and not (element N+O extend 1)')
    cmd.hide('everything', 'name H')

def load_crystal_protein(protein, ligand):
    cmd.load('{}/structures/proteins/{}_prot.pdb'.format(protein, ligand))
    cmd.set_name('{}_prot'.format(ligand), 'prot_{}'.format(ligand))
    return 'prot_' + ligand 

def load_crystal_pose(protein, ligand):
    cmd.load('{}/structures/ligands/{}_lig.sdf'.format(protein, ligand, ligand))
    cmd.set_name('{}_lig'.format(ligand), 'crystal_{}'.format(ligand))

###################################################################
def load_pose(protein, ligand, pose, prefix):
    ligand_file ='{}/dock_out/{}-docked.sdf.gz'.format(protein, ligand)
    pv = glob(ligand_file)[0]
    lig_name = ligand.split('-to-')[0]
    
    cmd.load(pv,ligand)
    cmd.split_states(ligand)
    cmd.delete(ligand)
    if pose == 0:
        pose = ''
    elif pose < 10:
        pose = '0' + str(pose)
    else:
        pose = str(pose)
    print(ligand, pose, prefix, ligand)
    cmd.set_name(lig_name + pose, '{}_{}'.format(prefix, ligand))
    cmd.delete(lig_name + '*')
    return ligand_file

def load_top_docked(protein, n = 1):
    n = int(n)
    grid = None
    for prot in sorted(glob('{}/structures/proteins/*_prot.pdb'.format(protein)))[:n]:
        pdb = prot.split('/')[-1].split('_')[0]
        if grid is None:
            grid = pdb
        print(pdb)
        load_pose(protein, pdb, grid, 0, 'docked')
    cmd.show('sticks', "docked_*")
    cmd.hide('lines', 'element h')
    cmd.hide('everything', 'element H and not (element N+O extend 1)')

def load_results(protein, scores, show_inter:bool=False, show_chembl_inter:bool=False):
    """
    USAGE

    load_results protein, pose_prediction_csv, [show_inter=False], [show_chembl_inter=False]

    load all results from the ComBind docking for the ligands in [pose_prediction_csv]. Additionally, loads crystal structures for non-CHEMBL ligands

    Single ligand docking results are in yellow, ComBind results are in cyan, and crystal structures are in white
    """
    struct = glob('{}/structures/template/*'.format(protein))[0].split('/')[-1].split('.')[0]
    prot_obj = load_crystal_protein(protein, struct)
    with open(scores) as fp:
        fp.readline()
        for line in fp:
            if line[:3] == 'com': continue
            (ligand,
             combind_rank, combind_rmsd,
             docked_rmsd,
             best_rmsd) = line.strip().split(',')
            ligand_file = load_pose(protein, ligand, int(combind_rank), 'combind')
            load_pose(protein, ligand, 0, 'docked')
            if show_inter and ('CHEMBL' not in ligand_file or show_chembl_inter):
                print(ligand_file)
                interaction_file = ligand_file.split('.sdf.gz')[0]+'_ifp_rd1_raw.csv'
                show_interactions(interaction_file, 'all', prot_obj, ligand, ligand_file, int(combind_rank))
                show_interactions(interaction_file, 'all', prot_obj, ligand, ligand_file, 0)
            ligand = ligand.split('-to-')[0].replace('_lig','')
            if ligand[:6] != 'CHEMBL':
                load_crystal_pose(protein, ligand)
            cmd.group(ligand,f'*{ligand}*')

    cmd.show('sticks', "docked_*")
    cmd.show('sticks', "combind_*")
    cmd.show('sticks', "crystal_*")
    cmd.hide('lines', 'element h')
    cmd.hide('everything', 'element H and not (element N+O extend 1)')

    cmd.util.cbaw('*')
    cmd.color('yellow', 'docked* and element c')
    cmd.color('cyan', 'combind* and element c')
    cmd.set('stick_radius', '0.13')
    
    
    cmd.show('cartoon')
    cmd.set('cartoon_oval_length', '0.5')
    cmd.set('cartoon_transparency', '0.5')

###############################################################
def style():
    cmd.show('cartoon')
    cmd.show('lines')
    cmd.hide('sticks')
    cmd.util.cbaw()
    cmd.color('slate', 'het and element C')
    cmd.hide('everything', 'element H and (element C extend 1)')

def pose_name(bn, pose):
    if pose == 0:
        pose = ''
    elif pose < 10:
        pose = '0{}'.format(pose)
    else:
        pose = '{}'.format(pose)
    return '{}{}'.format(bn, pose)

def enable(group, pose, prot=True):
    cmd.enable(group+'*-to-*')
    cmd.disable(group+'*-to-*.*')
    cmd.enable(pose_name(group, pose))
    if prot:
        cmd.enable('{}*.*_prot'.format(group))

def show_interactions(ifp_file, interaction, protein, lig, ligand_file, pose, delete=True, disable=True, lig_mol = None):
    """
    USAGE

    show_interactions interaction_file, interaction, protein, ligand, ligand_file, pose_number,[delete=True],[disable=True]

    interaction is the type of interaction to show; can be one of: all, sb, hbond, contact
    ligand is the name of the ligand object, while ligand_file is the name of the file containing the ligand
    """
    if interaction not in ['all','hbond','contact','sb']: raise IOError('interaction must be one of: all, hbond, contact, sb')

    pose = int(pose)

    if delete:
        cmd.delete('dist*')
        cmd.delete('ps*')
    if disable:
        cmd.disable('*')
    style()

    enable(lig, pose)

    if lig_mol is None:
        if isinstance(ligand_file,str):
            with fileinput.hook_compressed(ligand_file,'rb') as lf:
                lig_mols = [mol for mol in Chem.ForwardSDMolSupplier(ligand_file)]
        assert len(lig_mols), "No ligands found in {}".format(ligand_file)
        lig_mol = lig_mols[pose]

    if interaction == 'all':
        for interaction in ['sb', 'hbond', 'contact']: #, 'pipi']:
             show_interactions(ifp_file, interaction, protein, lig, ligand_file, pose,
                               delete=False, disable=False, lig_mol= lig_mol)

    
    df = pd.read_csv(ifp_file)

    if interaction == 'hbond':
        idx = df['label'] == 'hbond_acceptor'
        idx |= df['label'] == 'hbond_donor'
        thresh = 3.5
        color = 'yellow'
    elif interaction == 'sb':
        idx = df['label'] == 'saltbridge'
        thresh = 4
        color = 'magenta'
    elif interaction == 'contact':
        idx = df['label'] == 'contact'
        thresh = 1.25
        color='green'
    elif interaction == 'pipi':
        idx = df['label'] == 'pipi'
        idx |= df['label'] == 'pi-t'
        thresh = 7.0
        color='smudge'

    idx &= df['pose'] == pose

    for i, row in df[idx].iterrows():
        if interaction == 'contact' and row['dist'] > thresh*row['vdw']: continue
        if interaction != 'contact' and row['dist'] > thresh: continue
        
        chain, resid, _, _ = row['protein_res'].split(':')
        prot = '{} and chain {} and resid {} and name {}'.format(protein,
                                                                       chain,
                                                                       resid,
                                                                       row['protein_atom'].replace(',', '+'))
        ligand = [float(f) for f in lig_mol.GetConformer(-1).GetAtomPosition(row['ligand_atom'])]


        cmd.pseudoatom('ps{}{}prot'.format(interaction, i), prot)
        cmd.pseudoatom('ps{}{}lig'.format(interaction, i), pos=ligand)

        cmd.dist('dist'+interaction+str(i),
                 'ps{}{}prot'.format(interaction, i),
                 'ps{}{}lig'.format(interaction, i))
        cmd.color(color, 'dist'+interaction+str(i))

    cmd.set('dash_width', 6)
    cmd.set('dash_width', 3, 'distcontact*')

    cmd.enable(protein)
    cmd.enable(pose_name(lig, pose))

    
cmd.extend('load_complexes', load_complexes)
cmd.extend('load_top_docked', load_top_docked)
cmd.extend('load_results', load_results)
cmd.extend('show_interactions', show_interactions)
