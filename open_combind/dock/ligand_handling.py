import requests
# import re
import urllib.parse
from rdkit import Chem
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
from prody import writePDBStream
import gzip
# from bs4 import BeautifulSoup


class RDKitParseException(Exception):
    """
    Exception raised when RDKit cannot parse a molecule
    """
    def __init__(self, message):
        super().__init__(message)


def get_ligands_from_RCSB(pdbid, lig_code=None, specific_chain=None,
                        save_file="{pdbid}_{chem_name}_{chain_id}_{seq_id}.sdf", first_only=False):
    """
    Given a PDBFile (or a PDB Header file), download the ligands present in the PDB file directly
    from the RCSB as a SDF file

    Parameters
    ----------
    pdbid : str
       PDB ID containing the small molecules
    lig_code : str, default=None
        Three letter chemical component identifier (CCI) used by the RCSB for your ligand of interest
    specific_chain : str, default=None
        If specified, will only retrieve the ligand(s) with the author chain ID 
    save_file : str, default="{pdbid}_{chem_name}_{seq_id}.sdf"
        Path to output SDF file for your ligand of interest. Defaults to "<PDB ID>_<CCI>_<Sequence Number>.sdf" for each ligand in the PDB File.
    first_only : bool, default=False
        Only make SDF file of the first HET chemical found in the PDB file. If `lig_code` is provided, only generates an SDF for that.

    Returns
    -------
    :class:`list[Mol]<list>`
        Ligands downloaded directly from the RCSB webpage as a :class:`~rdkit.Chem.rdchem.Mol`
    """

    lig_retrieve_format = "https://models.rcsb.org/v1/{pdbid}/ligand?auth_seq_id={seq_id}&label_asym_id={chain_id}&encoding=sdf&filename={pdbid}_{chem_name}_{seq_id}.sdf"

    all_ligand_info = get_ligand_info_RCSB(pdbid)
    assert (lig_code is None) ^ (lig_code in all_ligand_info.keys()), f"{lig_code} not a valid ligand in {pdbid}"

    molecules = []
    for idx, (chem_name, chem_info) in enumerate(all_ligand_info.items()):
        if (lig_code is not None) and (chem_name != lig_code):
            continue

        for chain_id, chain_info in chem_info.items():
            auth_chain_id = chain_info['auth_chain']
            if (specific_chain is not None) and (auth_chain_id != specific_chain):
                continue
            seq_id = chain_info['seq_id']

            page = requests.get(lig_retrieve_format.format(pdbid=pdbid, seq_id=seq_id,
                                chain_id=chain_id, chem_name=chem_name)).text
            if not page.startswith(chem_name):
                raise FileNotFoundError(f"Unable to retrieve instance coordinates for {chem_name} in entry {pdbid}")
            mol = Chem.MolFromMolBlock(page)
            if mol is None:
                raise RDKitParseException(f"RDKit cannot parse the molecule, {chem_name}, downloaded from the PDB for entry {pdbid}")

            save_lig_path = save_file.format(pdbid=pdbid, chem_name=chem_name, seq_id=seq_id,chain_id=chain_id)
            # print(save_lig_path)
            mol_to_sdf(mol, save_lig_path)
            if first_only:
                return mol
            else:
                molecules.append(mol)
    return molecules

def ligand_selection_to_mol(ligand_selection, query_ligand, outfile=None):
    """
    Converts a :class:`~prody.atomic.atomgroup.AtomGroup` to a :class:`~rdkit.Chem.rdchem.Mol` using another :class:`~rdkit.Chem.rdchem.Mol` as a template for the correct bond orders of the :class:`~prody.atomic.atomgroup.AtomGroup`

    Parameters
    ----------
    ligand_selection : :class:`~prody.atomic.atomgroup.AtomGroup`
        Selection from a PDB file containing the atoms of the ligand
    query_ligand : :class:`~rdkit.Chem.rdchem.Mol`
        Template molecule with correct bond orders (atomic coordinates are irrelevant)
    outfile : str
        Path to the output SDF file of ligand, if not specified then the ligand is not saved to a file

    Returns
    -------
    :class:`~rdkit.Chem.rdchem.Mol`
        Ligand with proper coordinates and bond orders
    """
    
    template_lig = ligand_selection.select("not water")

    mol_holder = DummyMolBlock()
    writePDBStream(mol_holder, template_lig)

    tmpl_lig = Chem.MolFromPDBBlock(mol_holder.getBlock())
    tmpl_natoms, query_natoms = tmpl_lig.GetNumHeavyAtoms(), query_ligand.GetNumHeavyAtoms()
    if tmpl_natoms != query_natoms:
        print(f"WARNING:Ligand with bond orders has different number of atoms from ligand extracted from PDB file ({query_natoms} and {tmpl_natoms}, respectively)")
    proper_lig = AssignBondOrdersFromTemplate(query_ligand, tmpl_lig)

    if outfile is not None:
        mol_to_sdf(proper_lig, outfile)
        
    return proper_lig


def get_ligand_from_SMILES(lig_id):
    """
    Using the RCSB API, pulls the SMILES string containin stereochemical features for the given ligand ID.

    If a ligand has a dedicated page at ``https://www.rcsb.org/ligand/{ligand_id}`` then it likely has a SMILES string
    (does not need to be a 2 or 3 letter identifier)

    Parameters
    ----------
    lig_id : str
        Chemical component identifier provided by the RCSB database

    Returns
    -------
    :class:`~rdkit.Chem.rdchem.Mol`
        RDKit molecule created by the SMILES string

    """

    api_base_url = "https://data.rcsb.org/graphql?query="
    query = '{{chem_comp(comp_id:"{LIG_ID}")\n {{\nrcsb_chem_comp_descriptor {{\nSMILES_stereo\n}}\n}}\n}}'

    response = requests.get(api_base_url + urllib.parse.quote(query.format(LIG_ID=lig_id)))

    if not (response and ("SMILES_stereo" in response.text)):
        raise FileNotFoundError(f"Unable to retrieve model sdf file of ligand {lig_id}")
    smiles_string = response.json()['data']['chem_comp']['rcsb_chem_comp_descriptor']['SMILES_stereo']
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise RDKitParseException(f"RDKit cannot parse the molecule, {lig_id}, downloaded from the PDB as a SMILES string")

    return mol



# def scrape_rcsb_webpage(pdb_id):
#     rec_page_url = "https://www.rcsb.org/structure/{receptor}"
#     receptor_page = requests.get(rec_page_url.format(receptor=pdb_id))
#     soup = BeautifulSoup(receptor_page.text, 'html.parser')
#     ligand_rows = soup.find_all('tr', id=re.compile('ligand_row_\w{1,3}'))
#     ligand_chain_info = dict()
#     for lig_row in ligand_rows:
#         lig_cols = lig_row.find_all('td')
#         lig_name = lig_cols[0].a.text
#         chain_info = dict()
#         for link_tag in lig_cols[0].find_all('ul', class_='dropdown-menu')[0].find_all('a'):
#             url = link_tag.get('href')
#             if 'encoding=sdf' not in url:
#                 continue
#             text = link_tag.text.split(',')[-1]
#             chain = text.split()[1]  # first word is always "chain"
#             auth_chain = re.findall('\[auth ([A-Z])\]', text)
#             if len(auth_chain) == 1:
#                 auth_chain = auth_chain[0]
#             elif len(auth_chain) > 1:
#                 raise IndexError(f"more than one auth chain for {text} in {pdb_id}")
#             else:
#                 auth_chain = chain
#             chain_info[chain] = {"auth_chain": auth_chain, "url": url}

#         ligand_chain_info[lig_name] = chain_info

#     return ligand_chain_info


def get_ligand_info_RCSB(pdb_id):
    """
    Downloads the information for all of the nonpolymer entities in the given `pdb_id` from RCSB.

    Parameters
    ----------
    pdb_id : str
        PDB ID of the RCSB entry

    Returns
    -------
    dict
        Dictionary containing all of the nonpolymer entities with their chain, author specified chain, and residue number.
    """

    api_base_url = "https://data.rcsb.org/graphql?query="
    query = '{{entry(entry_id:"{PDB_ID}")\n {{\nnonpolymer_entities {{\nnonpolymer_entity_instances {{\nrcsb_nonpolymer_entity_instance_container_identifiers {{asym_id auth_asym_id comp_id auth_seq_id\n}}\n}}\n}}\n}}\n}}'

    response = requests.get(api_base_url + urllib.parse.quote(query.format(PDB_ID=pdb_id)))

    all_ligs_info = response.json()['data']['entry']['nonpolymer_entities']
    entry_ligands_info = dict()
    for lig in all_ligs_info:
        ligand_chain_info = dict()
        for lig_instance in lig['nonpolymer_entity_instances']:
            lig_inst = lig_instance['rcsb_nonpolymer_entity_instance_container_identifiers']
            lig_name = lig_inst['comp_id']
            auth_chain = lig_inst['auth_asym_id']
            chain = lig_inst['asym_id']
            seq_id = lig_inst['auth_seq_id']
            ligand_chain_info[chain] = {"auth_chain": auth_chain, "seq_id": seq_id}
        entry_ligands_info[lig_name] = ligand_chain_info

    return entry_ligands_info

def mol_to_sdf(mol, path_to_sdf):
    if path_to_sdf.endswith('.gz'):
        with gzip.open(path_to_sdf,'wt') as f:
            with Chem.SDWriter(f) as w:
                w.write(mol)
    else:
        with Chem.SDWriter(path_to_sdf) as w:
                w.write(mol)


class DummyMolBlock():
    def __init__(self):
        self.molblock = ""

    def write(self, string):
        self.molblock += string

    def getBlock(self):
        return self.molblock
