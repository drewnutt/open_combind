import os
import shutil
from prody import parsePDB, writePDB
from rdkit.Chem import MolFromPDBFile, SDWriter
# from plumbum.cmd import obabel
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def split_complex(complex_loc, pdb_id, ligand_select='hetatm',
                structures_loc="structures/"):
    """
    Splits the complex into the object selected by `ligand_select`, the ligand, and everything else, the protein.

    Parameters
    ----------
    complex_loc : str
        Path to protein-ligand complex PDB file
    pdb_id : str
        PDB ID of the protein-ligand complex, used for naming
    ligand_select : str, default='hetatm'
        `ProDy atom selection <http://prody.csb.pitt.edu/manual/reference/atomic/select.html#selections>`_ that defines the ligand atoms
    structures_loc : str, default="structures/"
        Base directory, containing directories ``proteins/`` and ``ligands/``, to place the separated protein and ligands, respectively

    """

    st = parsePDB(complex_loc)
    lig_path = '{structures_loc}ligands/{pdb_id}_lig.sdf'.format(structures_loc=structures_loc, pdb_id=pdb_id)
    aligned_lig_path = '{structures_loc}aligned/{pdb_id}/{pdb_id}_lig.sdf'.format(structures_loc=structures_loc, pdb_id=pdb_id)
    prot_path = '{structures_loc}proteins/{pdb_id}_prot.pdb'.format(structures_loc=structures_loc, pdb_id=pdb_id)
    lig_pdb_path = '{structures_loc}ligands/{pdb_id}_lig.pdb'.format(structures_loc=structures_loc, pdb_id=pdb_id)

    if not os.path.exists(lig_path):
        if not os.path.exists(aligned_lig_path):
            print(f"No separate aligned ligand found for {complex_loc}")
            lig_st = st.select(f'{ligand_select}') 
            assert lig_st is not None, f"no ligand found in {complex_loc} using {ligand_select}"
            writePDB(lig_pdb_path, lig_st)

            rdk_lig = MolFromPDBFile(lig_pdb_path)
            lig_writer = SDWriter(lig_path)
            lig_writer.write(rdk_lig)
            lig_writer.close()
            # obabel[lig_pdb_path, '-O', lig_path]()
            os.remove(lig_pdb_path)
        else:
            shutil.copy(aligned_lig_path, lig_path)

    if not os.path.exists(prot_path):
        prot_st = st.select('protein')
        if ligand_select != "hetatm" or ligand_select != "hetero":
            prot_st = prot_st.select(f"not {ligand_select}")
        writePDB(prot_path,prot_st)
        protonate_protein(prot_path)

def struct_sort(structs, opt_path='structures/aligned/{pdbid}/{pdbid}_aligned.pdb'):
    """
    Looks for each PDB ID in `structs` at the format string path given by `opt_path` and then splits that file based
    on the ligand information provided in::
        
        `opt_path`.replace("aligned/<PDB ID>", "raw").replace("_aligned.pdb",".info")

    Parameters
    ----------
    structs : iterable of str
        PDB IDs of all the protein-ligand complexes that should be split
    opt_path : str, default='structures/aligned/{pdbid}/{pdbid}_aligned.pdb'
        Format string of the protein-ligand complex paths, where ``pdbid`` will be replaced with each PDB ID.
    """
    structures_loc = opt_path.replace("aligned/{pdbid}/{pdbid}_aligned.pdb", "")
    os.system(f'mkdir -p {structures_loc+"proteins"} {structures_loc+"ligands"}')
    for struct in structs:
        opt_complex = opt_path.format(pdbid=struct)

        if os.path.exists(opt_complex):
            ligand_select = 'hetatm'  # 'hetero' == 'not (protein or nucleic)' so breaks if ATP or similar ligand
            # therefore switched to 'hetatm'
            liginfo_path = opt_complex.replace(f'aligned/{struct}', 'raw').replace('_aligned.pdb', '.info')
            liginfo = open(liginfo_path, 'r').readlines()
            if len(liginfo[0].strip('\n')) > 4:
                ligand_select = liginfo[1]
            split_complex(opt_complex, struct, opt_complex,
                    ligand_select=ligand_select,
                    structures_loc = structures_loc)

def protonate_protein(path_to_protein, pH=7.0):
    """
    Protonates the protein at the given pH using PDBFixer.

    Protein is loaded from `path_to_protein` and then protonated at `pH` using PDBFixer. The protonated protein is then
    written back to `path_to_protein`.

    Parameters
    ----------
    path_to_protein : str
        Path to the protein PDB file
    pH : float, default=7.0
        pH at which to protonate the protein
    """

    print(path_to_protein)
    pdb = PDBFixer(filename=path_to_protein)
    # pdb.findMissingResidues()
    pdb.missingResidues = {}
    pdb.findMissingAtoms()
    pdb.addMissingAtoms()
    pdb.addMissingHydrogens(pH=pH)
    PDBFile.writeFile(pdb.topology,pdb.positions,open(path_to_protein,'w'))
