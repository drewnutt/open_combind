import os
import pkg_resources
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit.Chem.rdmolfiles import MolFromMolFile

CMD = "gnina -r {prot} --autobox_ligand {abox_lig} " 

def make_grid(pdb,
              PROTFILE='structures/proteins/{pdb}_prot.pdb',
              LIGFILE='structures/ligands/{pdb}_lig.sdf',
              DOCKTEMP='structures/template/{pdb}.template',
              CUSTOMATOMS=None,
              box_add=8):
    """
    Make a docking template for crossdocking on a protein with the ``autobox_ligand`` defined by the cognate ligand.

    Parameters
    ----------
    pdb : str
        The PDB code of the protein to make a template for.
    PROTFILE : str
        The path to the protein file.
    LIGFILE : str
        The path to the ligand file.
    DOCKTEMP : str
        The path to the output template file.
    CUSTOMATOMS : str
        The path to a custom atom types file. If `None` and the template requires it, the default will be used.
    box_add : float
        The amount to add to the box size to make sure the ligand is fully enclosed.
    """


    ligfile = os.path.abspath(LIGFILE.format(pdb=pdb))
    protfile = os.path.abspath(PROTFILE.format(pdb=pdb))
    docktemp = os.path.abspath(DOCKTEMP.format(pdb=pdb))
    atom_file = ''
    if '{atom_file}' in CMD:
        if CUSTOMATOMS is None:
            atom_file = pkg_resources.resource_filename(__name__, "crossdock_atom_types.txt")
        else:
            atom_file = os.path.abspath(CUSTOMATOMS)


    cmd = CMD.format(prot=protfile, abox_lig=ligfile, aadd=box_add, atom_file=atom_file)

    if not (os.path.exists(ligfile) and os.path.exists(protfile)):
        print(ligfile, protfile)
        return  # Not ready.

    print('making grid', pdb)

    print(os.path.dirname(os.path.abspath(docktemp)))
    os.makedirs(os.path.dirname(os.path.abspath(docktemp)), exist_ok=True)

    with open(docktemp, 'w') as template:
        template.write(cmd)

def get_mol_center(ligfile):
    mol = MolFromMolFile(ligfile)
    return tuple(ComputeCentroid(mol.GetConformer(0)))
