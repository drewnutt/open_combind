import os
from prody import parsePDB, writePDB
from plumbum.cmd import obabel

def split_complex(st, pdb_id, structname):
    os.system('mkdir -p structures/proteins structures/ligands')
    lig_path = 'structures/ligands/{}_lig.sdf'.format(pdb_id)
    prot_path = 'structures/proteins/{}_prot.pdb'.format(pdb_id)
    lig_pdb_path = 'structures/ligands/{}_lig.pdb'.format(pdb_id)

    if not os.path.exists(lig_path):
        lig_st = st.select('hetatm') # 'hetero' == 'not (protein or nucleic)' so breaks if ATP or similar ligand
        # therefore switched to 'hetatm'
        assert lig_st is not None, f"no hetero in {structname}"
        writePDB(lig_pdb_path,lig_st)
        obabel[lig_pdb_path, '-O', lig_path]()
        os.remove(lig_pdb_path)

    if not os.path.exists(prot_path):
        prot_st = st.select('protein')
        writePDB(prot_path,prot_st)

def struct_sort(structs):
    for struct in structs:
        opt_complex = 'structures/aligned/{}/{}_aligned.pdb'.format(struct, struct)

        if os.path.exists(opt_complex):
            comp_st = parsePDB(opt_complex)
            split_complex(comp_st, struct, opt_complex)
