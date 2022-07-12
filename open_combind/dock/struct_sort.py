import os
import shutil
from prody import parsePDB, writePDB
from rdkit.Chem import MolFromPDBFile, SDWriter
# from plumbum.cmd import obabel
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def split_complex(complex_loc, pdb_id, structname,ligand_select='hetatm',
        structures_loc="structures/"):
    """
    Splits the complex

    Parameters
    ----------
    complex_loc : str
    pdb_id : str
    structname : str
    ligand_select : str, default='hetatm'
    structures_loc : str, default="structures/"

    """
    st = parsePDB(complex_loc)
    lig_path = '{structures_loc}ligands/{pdb_id}_lig.sdf'.format(structures_loc=structures_loc, pdb_id=pdb_id)
    aligned_lig_path = '{structures_loc}aligned/{pdb_id}/{pdb_id}_lig.sdf'.format(structures_loc=structures_loc, pdb_id=pdb_id)
    prot_path = '{structures_loc}proteins/{pdb_id}_prot.pdb'.format(structures_loc=structures_loc, pdb_id=pdb_id)
    lig_pdb_path = '{structures_loc}ligands/{pdb_id}_lig.pdb'.format(structures_loc=structures_loc, pdb_id=pdb_id)

    if not os.path.exists(lig_path):
        if not os.path.exists(aligned_lig_path):
            print(f"No separate aligned ligand found for {structname}")
            lig_st = st.select(f'{ligand_select}') 
            assert lig_st is not None, f"no ligand found in {structname} using {ligand_select}"
            writePDB(lig_pdb_path,lig_st)

            rdk_lig = MolFromPDBFile(lig_pdb_path)
            lig_writer = SDWriter(lig_path)
            lig_writer.write(rdk_lig)
            lig_writer.close()
            # obabel[lig_pdb_path, '-O', lig_path]()
            os.remove(lig_pdb_path)
        else:
            shutil.copy(aligned_lig_path,lig_path)

    if not os.path.exists(prot_path):
        prot_st = st.select('protein')
        if ligand_select != "hetatm" or ligand_select != "hetero":
            prot_st = prot_st.select(f"not {ligand_select}")
        writePDB(prot_path,prot_st)
        protonate_protein(prot_path)

def struct_sort(structs, opt_path='structures/aligned/{pdbid}/{pdbid}_aligned.pdb'):
    structures_loc = opt_path.replace("aligned/{pdbid}/{pdbid}_aligned.pdb","")
    os.system(f'mkdir -p {structures_loc+"proteins"} {structures_loc+"ligands"}')
    for struct in structs:
        opt_complex = opt_path.format(pdbid=struct)

        if os.path.exists(opt_complex):
            ligand_select = 'hetatm' # 'hetero' == 'not (protein or nucleic)' so breaks if ATP or similar ligand
            # therefore switched to 'hetatm'
            liginfo_path = opt_complex.replace(f'aligned/{struct}','raw').replace('_aligned.pdb','.info')
            liginfo = open(liginfo_path,'r').readlines()
            if len(liginfo[0].strip('\n')) > 4:
                ligand_select = liginfo[1]
            split_complex(opt_complex, struct, opt_complex,
                    ligand_select=ligand_select,
                    structures_loc = structures_loc)

def protonate_protein(path_to_protein, pH=7.0):
    print(path_to_protein)
    pdb = PDBFixer(filename=path_to_protein)
    # pdb.findMissingResidues()
    pdb.missingResidues = {}
    pdb.findMissingAtoms()
    pdb.addMissingAtoms()
    pdb.addMissingHydrogens(pH=pH)
    PDBFile.writeFile(pdb.topology,pdb.positions,open(path_to_protein,'w'))
