from pymol import cmd
import sys
import pandas as pd

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

def show_interactions(ifp_file, interaction, protein, lig, pose, delete=True, disable=True):
    """
    USAGE

    show_interactions interaction_file, interaction, prot, ligand, pose_number,[delete=True],[disable=True]

    interaction is the type of interaction to show; can be one of: all, sb, hbond, contact
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

    if interaction == 'all':
        for interaction in ['sb', 'hbond', 'contact']: #, 'pipi']:
             show_interactions(ifp_file, interaction, lig, pose,
                               delete=False, disable=False)
    
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
        color='smudge'
    elif interaction == 'pipi':
        idx = df['label'] == 'pipi'
        idx |= df['label'] == 'pi-t'
        thresh = 7.0
        color='green'

    idx &= df['pose'] == pose

    for i, row in df[idx].iterrows():
        if interaction == 'contact' and row['dist'] > thresh*row['vdw']: continue
        if interaction != 'contact' and row['dist'] > thresh: continue
        
        chain, resid, _, _ = row['protein_res'].split(':')
        prot = '{} and chain {} and resid {} and name {}'.format(protein,
                                                                       chain,
                                                                       resid,
                                                                       row['protein_atom'].replace(',', '+'))
        print(row['dist'])
        ligand = '{} and {}'.format(pose_name(lig, pose), str(row['ligand_atom']).replace(',', '+'))

        print(prot, ligand)

        cmd.pseudoatom('ps{}{}prot'.format(interaction, i), prot)
        cmd.pseudoatom('ps{}{}lig'.format(interaction, i), ligand)

        cmd.dist('dist'+interaction+str(i),
                 'ps{}{}prot'.format(interaction, i),
                 'ps{}{}lig'.format(interaction, i))
        cmd.color(color, 'dist'+interaction+str(i))

    cmd.set('dash_width', 6)
    cmd.set('dash_width', 3, 'distcontact*')

    cmd.enable(protein)
    cmd.enable(pose_name(lig, pose))

cmd.extend('show_interactions', show_interactions)
