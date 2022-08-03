import os
import pkg_resources
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit.Chem.rdmolfiles import MolFromMolFile

CMD = "gnina -r {prot} --autobox_ligand {abox_lig} --autobox_add {aadd} --custom_atoms {atom_file} "
# CMD = "gnina -r {prot} --center_x {lig_x:.3f} --center_y {lig_y:.3f} --center_z {lig_z:.3f} --size_x {box_x} --size_y {box_y} --size_z {box_z} "

def make_grid(pdb,
              PROTFILE='structures/proteins/{pdb}_prot.pdb',
              LIGFILE='structures/ligands/{pdb}_lig.sdf',
              DOCKTEMP='structures/template/{pdb}.template',
              CUSTOMATOMS=None,
              box_add=8):

    ligfile = os.path.abspath(LIGFILE.format(pdb=pdb))
    protfile = os.path.abspath(PROTFILE.format(pdb=pdb))
    docktemp = os.path.abspath(DOCKTEMP.format(pdb=pdb))
    if CUSTOMATOMS is None:
        atom_file = pkg_resources.resource_filename(__name__, "crossdock_atom_types.txt")
        print(atom_file)
    else:
        atom_file = os.path.abspath(CUSTOMATOMS)


    # center_tuple = get_mol_center(ligfile)
    cmd = CMD.format(prot=protfile, abox_lig=ligfile, aadd=box_add, atom_file=atom_file)
            # lig_x=center_tuple[0], lig_y=center_tuple[1], lig_z=center_tuple[2],
            # box_x=box_dimension, box_y=box_dimension, box_z=box_dimension)

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
