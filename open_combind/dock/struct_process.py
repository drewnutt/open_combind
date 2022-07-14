import os
import re
import requests
from prody import parsePDB, writePDB
from rdkit.Chem import AllChem as Chem
import numpy as np
from open_combind.dock.ligand_handling import get_ligands_from_RCSB, ligand_selection_to_mol, get_ligand_from_SMILES, RDKitParseException


def load_complex(prot_in, lig_id, other_lig=None):
    """
    Loads the protein and separates it into separate ``ProDy.AtomGroups`` for the protein-ligand complex only, protein only, waters, heteroatoms, and ligand only.

    Parameters
    ----------
    prot_in : str
        Path to PDB file containing protein and ligand
    lig_id : str
        Three letter residue name or ProDy selection string to identify the ligand
    other_lig : str, default=None
        ProDy selection string that identifies any ligands other the one in the desired pocket
    """

    lig_chain = None
    prot_st = parsePDB(prot_in, altloc='all')
    chains = np.unique(prot_st.getChids())
    fchain = chains[0]

    # filter protein to remove waters and other non-protein things
    prot_only = prot_st.select('protein')
    if len(np.unique(prot_only.getAltlocs())) > 1:
        altlocs = np.unique(prot_only.getAltlocs())
        altlocs = [alt if alt != ' ' else '_' for alt in altlocs]
        prot_only = prot_only.select(f'altloc {altlocs[0]} or altloc {altlocs[1]}')
    waters = prot_st.select('water')
    heteros = prot_st.select('hetero and not water')
    if len(lig_id) < 4:
        important_ligand = prot_st.select(f'resname {lig_id} and chain {fchain}')
        lig_chain = fchain
        if important_ligand is None:
            lig_chain = None
            important_ligand = prot_st.select(f'resname {lig_id}')
        assert important_ligand is not None, f"no ligand found with resname {lig_id} for {prot_in}"
    else:
        important_ligand = prot_st.select(f'{lig_id} and not {other_lig}')
        if other_lig is not None:
            prot_only = prot_only.select(f'not {other_lig} and not {lig_id}')
        assert important_ligand is not None, f"nothing found with {lig_id} for {prot_in} to select as ligand"
        chains = np.unique(important_ligand.getChids())
        if len(chains) == 1:
            lig_chain = chains[0]
    if len(np.unique(important_ligand.getAltlocs())) > 1:
        altlocs = np.unique(important_ligand.getAltlocs())
        altlocs = [alt if alt != ' ' else '_' for alt in altlocs]
        important_ligand = important_ligand.select(f'altloc {altlocs[0]}')
        important_ligand.setAltlocs(' ')
    important_ligand = important_ligand.select('not water')

    return prot_only, waters, heteros, important_ligand, lig_chain


def struct_process(structs,
                   protein_in='structures/raw/{pdbid}.pdb',
                   ligand_info='structures/raw/{pdbid}.info',
                   filtered_protein='structures/processed/{pdbid}/{pdbid}_prot.pdb',
                   filtered_complex='structures/processed/{pdbid}/{pdbid}_complex.pdb',
                   filtered_ligand='structures/processed/{pdbid}/{pdbid}_lig.sdf',
                   filtered_hetero='structures/processed/{pdbid}/{pdbid}_het.pdb',
                   filtered_water='structures/processed/{pdbid}/{pdbid}_wat.pdb'):
    """
    Filters a list of raw PDB file into its main separate components.
    Creates a pdb file for the following:

        * Protein and ligand atoms only (only one ligand molecule)
        * Protein only atoms
        * Heteroatoms and not water
        * Water only

    Additionally, a ligand SDF is pulled from the PDB, if possible.

    Parameters
    -----------
    structs : iterable of str
        PDB IDs of the raw PDB files that need to be processed
    protein_in : str, default='structures/raw/{pdbid}.pdb'
        Format string of the path to the protein, given the PDBID as `pdbid`
    ligand_info : str, default='structures/raw/{pdbid}.info'
        Format string of the path to the ``.info`` file, given the PDBID as `pdbid`
    filtered_protein : str, default='structures/processed/{pdbid}/{pdbid}_prot.pdb'
        Format string of the path to the processed protein only PDB file, given the PDBID as `pdbid`
    filtered_complex : str, default='structures/processed/{pdbid}/{pdbid}_complex.pdb'
        Format string of the path to the processed protein-ligand only only PDB file, given the PDBID as `pdbid`
    filtered_ligand : str, default='structures/processed/{pdbid}/{pdbid}_lig.sdf'
        Format string of the path to the processed ligand only only SDF file, given the PDBID as `pdbid`
    filtered_hetero : str, default='structures/processed/{pdbid}/{pdbid}_het.pdb'
        Format string of the path to the processed heteroatom only (no waters) PDB file, given the PDBID as `pdbid`
    filtered_water : str, default='structures/processed/{pdbid}/{pdbid}_wat.pdb')
        Format string of the path to the processed water only PDB file, given the PDBID as `pdbid`
    """

    for struct in structs:
        _protein_in = protein_in.format(pdbid=struct)
        _ligand_info = ligand_info.format(pdbid=struct)
        _filtered_protein = filtered_protein.format(pdbid=struct)
        _filtered_complex = filtered_complex.format(pdbid=struct)
        _filtered_ligand = filtered_ligand.format(pdbid=struct)
        _filtered_water = filtered_water.format(pdbid=struct)
        _filtered_hetero = filtered_hetero.format(pdbid=struct)
        _workdir = os.path.dirname(_filtered_protein)

        if os.path.exists(_filtered_complex):
            continue

        os.system('mkdir -p {}'.format(os.path.dirname(_workdir)))
        # os.system('rm -rf {}'.format(_workdir))
        os.system('mkdir {}'.format(_workdir))

        other_lig = None
        lig_info = open(_ligand_info,'r').readlines()
        lig_info = [info_line.strip('\n') for info_line in lig_info]
        if len(lig_info[0]) > 3:
            lig_id = lig_info[1]
            print(f"Non-standard RCSB ligand name:{lig_info[0]}")
            print(f"Using {lig_id} as the selection criterion")
            if len(lig_info) > 2:
                other_lig = lig_info[2]
        else:
            lig_id = lig_info[0]
        assert lig_id is not None
        print(f'processing {struct} with ligand {lig_id}')

        prot, waters, het, ligand, lig_chain = load_complex(_protein_in, lig_id, other_lig=other_lig)
        compl = prot + ligand
        if not os.path.exists(_filtered_ligand):
            create_correct_ligand_sdf(struct, lig_info[0], ligand, _filtered_ligand, ligand_chain=lig_chain)
        writePDB(_filtered_protein, prot)
        if waters is not None:
            writePDB(_filtered_water, waters)
        if het is not None:
            writePDB(_filtered_hetero, het)

        writePDB(_filtered_complex, compl)

def create_correct_ligand_sdf(pdb_id, lig_id, ligand, save_file, ligand_chain=None):
    """
    Create a SDF file containing only the ligand specified by `lig_id` with the correct coordinates and bond orders as specified in `pdb_id`

    Parameters
    ----------
    pdb_id : str
        PDB ID of the protein-ligand complex that the ligand is coming from
    lig_id : str
        Chemical Component Identifier (CCI) of the ligand, according to RCSB
    ligand : `ProDy.AtomGroups`
        Ligand atom group extracted from the PDB File
    save_file : str
        Path to save SDF of ligand
    ligand_chain : str
        Specify which author chain ligand to save to SDF, if not specified then will download each chain of the ligand with `lig_id`
    """

    mol = None
    if len(lig_id) < 4:
        try:
            mol = get_ligands_from_RCSB(pdb_id, lig_code=lig_id, specific_chain=ligand_chain,
                                    save_file=save_file, first_only=True)
        except FileNotFoundError as fnfe:
            if "Unable to retrieve instance coordinates" in str(fnfe):
                print(str(fnfe))
            else:
                raise fnfe
        except RDKitParseException as rdkpe:
            print(str(rdkpe))
    if mol is None:
        print(f"WARNING: Unable to download ligand SDF directly from RCSB for {lig_id}, extracting from PDB and assigning bonds from RCSB SMILES entry")
        try:
            mol_from_smiles = get_ligand_from_SMILES(lig_id)
            _ = ligand_selection_to_mol(ligand, mol_from_smiles, outfile=save_file)
        except FileNotFoundError as fnfe:
            if "Unable to retrieve instance coordinates" in str(fnfe):
                print(str(fnfe))
            else:
                raise fnfe
        except RDKitParseException as rdkpe:
            print(str(rdkpe))
