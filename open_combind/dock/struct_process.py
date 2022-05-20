import os
from prody import parsePDB, writePDB
from subprocess import run
import numpy as np

def load_complex(prot_in, lig_id, other_lig=None):

    prot_st = parsePDB(prot_in, altloc='all')
    chains = np.unique(prot_st.getChids())
    fchain = chains[0]

    #filter protein to remove waters and other non-protein things
    prot_only = prot_st.select('protein')
    if len(np.unique(prot_only.getAltlocs())) > 1:
        altlocs = np.unique(prot_only.getAltlocs())
        altlocs = [ alt if alt != ' ' else '_' for alt in altlocs]
        prot_only = prot_only.select(f'altloc {altlocs[0]} or altloc {altlocs[1]}')
    waters = prot_st.select('water')
    heteros = prot_st.select('hetero and not water')
    if len(lig_id) in [3,4]:
        important_ligand = prot_st.select(f'resname {lig_id} and chain {fchain}')
        if important_ligand is None:
            important_ligand = prot_st.select(f'resname {lig_id}')
        assert important_ligand is not None, f"no ligand found with resname {lig_id} for {prot_in}"
    else:
        important_ligand = prot_st.select(f'{lig_id}')
        if other_lig is not None:
            prot_only = prot_only.select(f'not {other_lig}')
        assert important_ligand is not None, f"nothing found with {lig_id} for {prot_in} to select as ligand"
    if len(np.unique(important_ligand.getAltlocs())) > 1:
        altlocs = np.unique(important_ligand.getAltlocs())
        altlocs = [ alt if alt != ' ' else '_' for alt in altlocs]
        important_ligand = important_ligand.select(f'altloc {altlocs[0]}')
        important_ligand.setAltlocs(' ')
    important_ligand = important_ligand.select('not water')
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

        if os.path.exists(_filtered_complex):
            continue

        other_lig = None
        lig_info = open(_ligand_info,'r').readlines()
        lig_info = [info_line.strip('\n') for info_line in lig_info]
        if len(lig_info[0]) not in [3,4]:
            lig_id = lig_info[1]
            print(f"Non-standard RCSB ligand name:{lig_info[0]}")
            print(f"Using {lig_id} as the selection criterion")
            if len(lig_info) > 2:
                other_lig = lig_info[2]
        else:
            lig_id = lig_info[0]
        assert lig_id is not None
        print(f'processing {struct} with ligand {lig_id}')

        os.system('mkdir -p {}'.format(os.path.dirname(_workdir)))
        os.system('rm -rf {}'.format(_workdir))
        os.system('mkdir {}'.format(_workdir))

        compl, prot, waters, het = load_complex(_protein_in, lig_id, other_lig=other_lig)
        writePDB(_filtered_protein,prot)
        if waters is not None:
            writePDB(_filtered_water,waters)
        if het is not None:
            writePDB(_filtered_hetero,het)
        writePDB(_filtered_complex,compl)
