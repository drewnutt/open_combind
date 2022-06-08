import os
import subprocess
from glob import glob

CMD = "gnina -r {prot} --autobox_ligand {abox_lig} "

def make_grid(pdb,
              PROTFILE='structures/proteins/{pdb}_prot.pdb',
              LIGFILE='structures/ligands/{pdb}_lig.sdf',
              DOCKTEMP='structures/template/{pdb}.template'):

    ligfile = os.path.abspath(LIGFILE.format(pdb=pdb))
    protfile = os.path.abspath(PROTFILE.format(pdb=pdb))
    docktemp = os.path.abspath(DOCKTEMP.format(pdb=pdb))
    cmd = CMD.format(prot=protfile,abox_lig=ligfile)

    if not (os.path.exists(ligfile) and os.path.exists(protfile)):
        print(ligfile, protfile)
        return # Not ready.

    print('making grid', pdb)

    print(os.path.dirname(os.path.abspath(docktemp)))
    os.makedirs(os.path.dirname(os.path.abspath(docktemp)), exist_ok=True)

    with open(docktemp, 'w') as template:
        template.write(cmd)
