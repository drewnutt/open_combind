import os
# from schrodinger.structure import StructureReader
from prody import parsePDB, writePDB
from subprocess import run
import numpy as np

# command = '$SCHRODINGER/utilities/prepwizard -WAIT -rehtreat -watdist 0 {}_in.mae {}_out.mae'

def load_complex(prot_in, lig_id):

    prot_st = parsePDB(prot_in)
    chains = np.unique(prot_st.getChids())
    fchain = chains[0]

    #filter protein to remove waters and other non-protein things
    prot_only = prot_st.select('protein')
    waters = prot_st.select('water')
    heteros = prot_st.select('hetero and not water')
    important_ligand = prot_st.select(f'resname {lig_id} and chain {fchain}')
    if important_ligand is None:
        important_ligand = prot_st.select(f'resname {lig_id}')
    compl = prot_only + important_ligand

    return compl, prot_only, waters, heteros

def struct_process(structs,
                   protein_in='structures/raw/{pdbid}.pdb',
                   ligand_info='structures/raw/{pdbid}.info',
                   filtered_protein='structures/processed/{pdbid}/{pdbid}_prot.pdb',
                   filtered_complex='structures/processed/{pdbid}/{pdbid}_complex.pdb',
                   filtered_hetero='structures/processed/{pdbid}/{pdbid}_het.pdb',
                   filtered_water='structures/processed/{pdbid}/{pdbid}_wat.pdb'):

    for struct in structs:
        _protein_in = protein_in.format(pdbid=struct)
        _ligand_info = ligand_info.format(pdbid=struct)
        _filtered_protein = filtered_protein.format(pdbid=struct)
        _filtered_complex = filtered_complex.format(pdbid=struct)
        _filtered_water = filtered_water.format(pdbid=struct)
        _filtered_hetero = filtered_hetero.format(pdbid=struct)
        _workdir = os.path.dirname(_filtered_protein)

        if os.path.exists(_filtered_protein):
            continue

        lig_id = open(_ligand_info,'r').readlines()[0].strip('\n')
        print('processing', struct)

        os.system('mkdir -p {}'.format(os.path.dirname(_workdir)))
        os.system('rm -rf {}'.format(_workdir))
        os.system('mkdir {}'.format(_workdir))

        compl, prot, waters, het = load_complex(_protein_in, lig_id)
        writePDB(_filtered_protein,prot)
        writePDB(_filtered_water,waters)
        writePDB(_filtered_hetero,het)
        writePDB(_filtered_complex,compl)

        # with open('{}/process_in.sh'.format(_workdir), 'w') as f:
        #     f.write('#!/bin/bash\n')
        #     f.write(command.format(struct, struct))
        # run('sh process_in.sh', shell=True, cwd=_workdir)
