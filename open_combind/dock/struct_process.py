import os
import re
import requests
from prody import parsePDB, writePDB
from rdkit.Chem import AllChem as Chem
import numpy as np
from open_combind.dock.ligand_handling import get_ligands_from_RCSB, ligand_selection_to_mol, get_ligand_from_SMILES, RDKitParseException


def load_complex(prot_in, lig_id, other_lig=None):
    """
    Loads the protein and separates it into separate :class:`~prody.atomic.atomgroup.AtomGroup`s for the protein-ligand complex only, protein only, waters, heteroatoms, and ligand only.

    Parameters
    ----------
    prot_in : str
        Path to PDB file containing protein and ligand
    lig_id : str
        Three letter residue name or ProDy selection string to identify the ligand
    other_lig : str, default=None
        ProDy selection string that identifies any ligands other the one in the desired pocket

    Returns
    -------
    prot_only : :class:`~prody.atomic.atomgroup.AtomGroup`
        Protein only atoms
    waters : :class:`~prody.atomic.atomgroup.AtomGroup`
        Water only atoms
    heteros : :class:`~prody.atomic.atomgroup.AtomGroup`
        Heteroatoms and not water atoms
    important_ligand : :class:`~prody.atomic.atomgroup.AtomGroup`
        Ligand only
    lig_chain : str
        Chain ID of the ligand
    """

    lig_chain = None
    prot_st = parsePDB(prot_in, altloc='all')
    assert prot_st is not None, f"could not load {prot_in}"
    chains = np.unique(prot_st.getChids())
    fchain = chains[0]

    # filter protein to remove waters and other non-protein things
    prot_only = prot_st.select('protein')
    assert prot_only is not None, f"no protein found in {prot_in}"
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
        important_ligand = prot_st.select(f'{lig_id} and not ( {other_lig} ) ' if other_lig is not None else f'{lig_id}')
        if other_lig is not None:
            prot_only = prot_only.select(f'not ( {other_lig} ) and not {lig_id}')
        else:
            prot_only = prot_only.select(f'not {lig_id}')
        assert important_ligand is not None, f"nothing found with {lig_id} for {prot_in} to select as ligand"
        chains = np.unique(important_ligand.getChids())
        if len(chains) == 1:
            lig_chain = chains[0]
    if len(np.unique(important_ligand.getAltlocs())) > 1:
        altlocs = np.unique(important_ligand.getAltlocs())
        altlocs = [alt if alt != ' ' else '_' for alt in altlocs]
        fin_alt_lig = None
        for alt in altlocs:
            imp_ligand = important_ligand.select(f'altloc {alt}')
            imp_ligand.setAltlocs(' ')
            if fin_alt_lig is None or len(imp_ligand) > len(fin_alt_lig):
                fin_alt_lig = imp_ligand
        important_ligand = fin_alt_lig 
    important_ligand = important_ligand.select('not water')
    # remove the ligand from the protein

    return prot_only, waters, heteros, important_ligand, lig_chain


def struct_process(structs,
                   raw_dir='structures/raw',
                   processed_dir='structures/processed',
                   protein_in='{raw_dir}/{pdbid}.pdb',
                   ligand_info='{raw_dir}/{pdbid}.info',
                   filtered_protein='{processed_dir}/{pdbid}/{pdbid}_prot.pdb',
                   filtered_complex='{processed_dir}/{pdbid}/{pdbid}_complex.pdb',
                   filtered_ligand='{processed_dir}/{pdbid}/{pdbid}_lig.sdf',
                   filtered_hetero='{processed_dir}/{pdbid}/{pdbid}_het.pdb',
                   filtered_water='{processed_dir}/{pdbid}/{pdbid}_wat.pdb'):
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
    processed_dir : str, default='structures/processed'
        Path to the directory where the processed structures will be saved
    protein_in : str, default='{raw_dir}/{pdbid}.pdb'
        Format string of the path to the protein, given the PDBID as `pdbid`
    ligand_info : str, default='{raw_dir}/{pdbid}.info'
        Format string of the path to the ``.info`` file, given the PDBID as `pdbid`
    filtered_protein : str, default='{processed_dir}/{pdbid}/{pdbid}_prot.pdb'
        Format string of the path to the processed protein only PDB file, given the PDBID as `pdbid`
    filtered_complex : str, default='{processed_dir}/{pdbid}/{pdbid}_complex.pdb'
        Format string of the path to the processed protein-ligand only only PDB file, given the PDBID as `pdbid`
    filtered_ligand : str, default='{processed_dir}/{pdbid}/{pdbid}_lig.sdf'
        Format string of the path to the processed ligand only only SDF file, given the PDBID as `pdbid`
    filtered_hetero : str, default='{processed_dir}/{pdbid}/{pdbid}_het.pdb'
        Format string of the path to the processed heteroatom only (no waters) PDB file, given the PDBID as `pdbid`
    filtered_water : str, default='{processed_dir}/{pdbid}/{pdbid}_wat.pdb')
        Format string of the path to the processed water only PDB file, given the PDBID as `pdbid`
    """

    for struct in structs:
        _protein_in = protein_in.format(pdbid=struct, raw_dir=raw_dir)
        _ligand_info = ligand_info.format(pdbid=struct, raw_dir=raw_dir)
        _filtered_protein = filtered_protein.format(pdbid=struct, processed_dir=processed_dir)
        _filtered_complex = filtered_complex.format(pdbid=struct, processed_dir=processed_dir)
        _filtered_ligand = filtered_ligand.format(pdbid=struct, processed_dir=processed_dir)
        _filtered_water = filtered_water.format(pdbid=struct, processed_dir=processed_dir)
        _filtered_hetero = filtered_hetero.format(pdbid=struct, processed_dir=processed_dir)
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
        if ' ' not in lig_id:
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
