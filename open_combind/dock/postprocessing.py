#!/usr/bin/python3

import argparse
import gzip
import os
from rdkit.Chem import ForwardSDMolSupplier, SDWriter
from rdkit.Chem.rdMolAlign import CalcRMS

def coalesce_poses(docked_files, sort_by='CNNscore', filter_RMSD=None, reverse=True):
    """
    .. include :: <isotech.txt>
    
    Coalesce multiple docked poses into a single file, sorted by a given property

    Parameters
    ----------
    docked_files : `list[str]<list>`
        List of docked pose files to coalesce
    sort_by : str
        Property to sort the poses by
    filter_RMSD : str
        Ground truth molecule file, filter poses that are >2 |angst| RMSD from GT
    reverse : bool
        Reverse the sort order, i.e. lowest first

    Returns
    -------
    sorted_mols : `list[Mol]<list>`
        List of sorted :class:`~rdkit.Chem.rdchem.Mol` objects
    """

    mols = []
    if isinstance(docked_files, str):
        docked_files = [docked_files]
    for dock_file in docked_files:
        if dock_file.endswith('.gz'):
            openfile = gzip.open
        else:
            openfile = open
        with openfile(dock_file,'rb') as gz:
            mols += [m for m in ForwardSDMolSupplier(gz)]
    sorted_mols = sorted(mols, key=lambda x: float(x.GetProp(sort_by)),reverse=reverse)

    if filter_RMSD is not None:
        if not os.path.exists(filter_RMSD):
            raise ValueError('Ground truth file does not exist')
        gt_mol = Chem.MolFromMolFile(filter_RMSD)
        sorted_mols = [mol for mol in sorted_mols if CalcRMS(mol,gt_mol) <= 2]

    return sorted_mols

def write_poses(sorted_mols, out_file):
    """
    Write poses to a file

    Parameters
    ----------
    sorted_mols : `list[Mol]<list>`
        List of sorted :class:`~rdkit.Chem.rdchem.Mol` objects
    out_file : str
        Output file to write the poses to
    """

    if out_file.endswith('.gz'):
        openfile=gzip.open
    else:
        openfile=open
    with openfile(out_file,'wt') as gz:
        writer = SDWriter(gz)
        for sm in sorted_mols:
            writer.write(sm)
        writer.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('DOCKED_POSES',nargs='*',help='docked files to coalesce and sort into a single file')
    parser.add_argument('--coalesced','-O',required=True,help='Output file with all of the poses sorted')
    parser.add_argument('--sort_by','-S',default='CNNscore',help='what to sort the poses by')
    parser.add_argument('--no_reverse', action='store_false',help='do not reverse the sort order, i.e. lowest first')
    parser.add_argument('--filter_RMSD','-R',default=None,help='Ground truth molecule, filter poses that >2 Angstrom RMSD from GT')
    return parser.parse_args()

def main():
    args = parse_args()

    sorted_mols = coalese_poses(args.DOCKED_POSES, args.coalesced, args.sort_by,
            args.filter_RMSD, args.no_reverse)
        
    write_poses(sorted_mols, args.coalesced)


if __name__ == '__main__':
    main()
