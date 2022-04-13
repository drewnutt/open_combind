import os
import subprocess
from glob import glob
# from schrodinger.structure import StructureReader
# from schrodinger.structutils.transform import get_centroid

# GRID_IN = """
# GRID_CENTER {x},{y},{z}
# GRIDFILE {pdb}.zip
# INNERBOX 15,15,15
# OUTERBOX 30,30,30
# RECEP_FILE {prot}
# """

# CMD = "glide -WAIT {infile}"
CMD = "gnina -r {prot} --autobox_ligand {abox_lig} "

# not necessary because Gnina can use ligand to define autobox
# def centroid(ligfile):
#     with StructureReader(ligfile) as st:
#         st = next(st)
#     c = get_centroid(st)
#     x,y,z = c[:3]
#     return x, y, z

def make_grid(pdb,
              PROTFILE='structures/proteins/{pdb}_prot.pdb',
              LIGFILE='structures/ligands/{pdb}_lig.mol2',
              DOCKTEMP='structures/template/{pdb}.template'):
    # if grid_in is None:
    #     grid_in = GRID_IN

    ligfile = os.path.abspath(LIGFILE.format(pdb=pdb))
    protfile = os.path.abspath(PROTFILE.format(pdb=pdb))
    docktemp = os.path.abspath(DOCKTEMP.format(pdb=pdb))
    cmd = CMD.format(prot=protfile,abox_lig=ligfile)

    # if os.path.exists(zipfile):
    #     return # Done.
    if not (os.path.exists(ligfile) and os.path.exists(protfile)):
        print(ligfile, protfile)
        return # Not ready.

    print('making grid', pdb)

    print(os.path.dirname(os.path.abspath(docktemp)))
    os.makedirs(os.path.dirname(os.path.abspath(docktemp)), exist_ok=True)

    with open(docktemp, 'w') as template:
        template.write(cmd)

    # x, y, z = centroid(ligfile)

    # with open(infile, 'w') as fp:
    #     fp.write(grid_in.format(x=x, y=y, z=z, pdb=pdb, prot=protfile))

    # subprocess.run(cmd, cwd=cwd, shell=True)
