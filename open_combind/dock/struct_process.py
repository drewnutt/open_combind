import os
import re
import requests
from prody import parsePDB, parsePDBHeader, writePDB
from rdkit.Chem import AllChem as Chem
import numpy as np
from bs4 import BeautifulSoup

class RDKitParseException(Exception):
    def __init__(self, message):
        super().__init__(message)

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
    if len(lig_id) == 3:
        important_ligand = prot_st.select(f'resname {lig_id} and chain {fchain}')
        if important_ligand is None:
            important_ligand = prot_st.select(f'resname {lig_id}')
        assert important_ligand is not None, f"no ligand found with resname {lig_id} for {prot_in}"
    else:
        important_ligand = prot_st.select(f'{lig_id}')
        if other_lig is not None:
            prot_only = prot_only.select(f'not {other_lig} and not {lig_id}')
        assert important_ligand is not None, f"nothing found with {lig_id} for {prot_in} to select as ligand"
    if len(np.unique(important_ligand.getAltlocs())) > 1:
        altlocs = np.unique(important_ligand.getAltlocs())
        altlocs = [alt if alt != ' ' else '_' for alt in altlocs]
        important_ligand = important_ligand.select(f'altloc {altlocs[0]}')
        important_ligand.setAltlocs(' ')
    important_ligand = important_ligand.select('not water')
    compl = prot_only + important_ligand

    return compl, prot_only, waters, heteros, important_ligand

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

    Paraameters
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
        os.system('rm -rf {}'.format(_workdir))
        os.system('mkdir {}'.format(_workdir))

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
            try:
                _ = get_ligands_frompdb(_protein_in,lig_code=lig_id,
                        save_file=_filtered_ligand,first_only=True)
            except FileNotFoundError as fnfe:
                if "Unable to retrieve instance coordinates" in str(fnfe):
                    print(str(fnfe))
            except RDKitParseException as rdkpe:
                print(str(rdkpe))
        assert lig_id is not None
        print(f'processing {struct} with ligand {lig_id}')

        compl, prot, waters, het, ligand = load_complex(_protein_in, lig_id, other_lig=other_lig)
        writePDB(_filtered_protein,prot)
        if waters is not None:
            writePDB(_filtered_water,waters)
        if het is not None:
            writePDB(_filtered_hetero,het)

        writePDB(_filtered_complex,compl)

def get_ligands_frompdb(pdbfile, lig_code=None, save_file="{pdbid}_{chem_name}_{seq_id}.sdf", first_only=False):
    """
    Given a PDBFile (or a PDB Header file), download the ligands present in the PDB file directly from the RCSB as a SDF file

    Parameters
    ----------
    pdbfile : str
        Path to PDB file or PDB header file to pull information about the contained small molecules
    lig_code : str, default=None
        Three letter chemical component identifier (CCI) used by the RCSB for your ligand of interest
    save_file : str, default="{pdbid}_{chem_name}_{seq_id}.sdf"
        Path to output SDF file for your ligand of interest. Defaults to "<PDB ID>_<CCI>_<Sequence Number>.sdf" for each ligand in the PDB File.
    first_only : bool, default=False
        Only make SDF file of the first HET chemical found in the PDB file. If `lig_code` is provided, only generates an SDF for that.

    Returns
    -------
    list of ` ``RDKit.rdchem.Mol`` <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol>`_
        Ligands downloaded directly from the RCSB webpage
    """
    lig_retrieve_format = "https://models.rcsb.org/v1/{pdbid}/ligand?auth_seq_id={seq_id}&label_asym_id={chain_id}&encoding=sdf&filename={pdbid}_{chem_name}_{seq_id}.sdf"
    header = parsePDBHeader(pdbfile)
    assert (lig_code is None) ^ (lig_code in header.keys()), f"{lig_code} not a valid ligand in {pdbfile}"
    # ligands_in_order = get_ligand_order(pdbfile)
    # print(ligands_in_order)
    # last_prot_chain = header['polymers'][-1].chid
    chemicals = header['chemicals']
    pdbid = header['identifier']
    all_ligand_info = scrape_rcsb_webpage(pdbid)
    molecules = []
    for idx, chemical in enumerate(chemicals):
        chem_name = chemical.resname
        if (lig_code is not None) and (chem_name != lig_code):
            continue
        seq_id = chemical.resnum
        auth_chain_id = chemical.chain
        current_ligand = all_ligand_info[chem_name]
        for chain_id, info in current_ligand.items():
            if auth_chain_id == info['auth_chain']:
                assert f"auth_seq_id={seq_id}" in info['url']
                break
        # add_to_pchain = ligands_in_order.index((chem_name,seq_id)) + 1
        # chain_id = chr(ord(last_prot_chain) + add_to_pchain)
        # print(f"Page:{lig_retrieve_format.format(pdbid=pdbid,seq_id=seq_id,chain_id=chain_id,chem_name=chem_name)}")

        # Maybe RCSB will get a better API for this, but for now we just have to comb the webpage
        page = requests.get(lig_retrieve_format.format(pdbid=pdbid, seq_id=seq_id, chain_id=chain_id, chem_name=chem_name)).text
        if not page.startswith(chem_name):
            raise FileNotFoundError(f"Unable to retrieve instance coordinates for {chem_name} in entry {pdbid}")
        mol = Chem.MolFromMolBlock(page)
        if mol is None:
            raise RDKitParseException(f"RDKit cannot parse the molecule, {chem_name}, downloaded from the PDB for entry {pdbid}")

        save_lig_path = save_file.format(pdbid=pdbid, chem_name=chem_name, seq_id=seq_id)
        writer = Chem.SDWriter(save_lig_path)
        writer.write(mol)
        writer.close()
        if first_only:
            molecules = mol
            break
        else:
            molecules.append(mol)
    return molecules

def get_ligands(pdbfile):
    liginfo_path = pdbfile.replace('.pdb','.info')
    liginfo = open(liginfo_path,'r').readlines()
    ligand_name = liginfo[0].strip('\n')
    if len(ligand_name) < 4:
        return get_ligands_frompdb(pdbfile,lig_code=ligand_name,first_only=True)
    else:
        return "ligand is protein"

def get_ligand_order(pdbfile):
    ligand_ordering = []
    with open(pdbfile) as read_file:
        for line in read_file:
            if line.startswith('HET '):
                chemname = line[7:10].strip()
                resnum = int(line[13:17])
                chain = line[10:13].strip()
                ligand_ordering.append((chemname,resnum,chain))
            elif line.startswith('ATOM '):
                break
    ordered_ligands = sorted(ligand_ordering, key=lambda x: (x[1], x[2]))
    return ordered_ligands

def scrape_rcsb_webpage(pdb_id):
    rec_page_url = "https://www.rcsb.org/structure/{receptor}"
    receptor_page = requests.get(rec_page_url.format(receptor=pdb_id))
    soup = BeautifulSoup(receptor_page.text, 'html.parser')
    ligand_rows = soup.find_all('tr', id=re.compile('ligand_row_\w{1,3}'))
    ligand_chain_info = dict()
    for lig_row in ligand_rows:
        lig_cols = lig_row.find_all('td')
        lig_name = lig_cols[0].a.text
        chain_info = dict()
        for link_tag in lig_cols[0].find_all('ul', class_='dropdown-menu')[0].find_all('a'):
            url = link_tag.get('href')
            if 'encoding=sdf' not in url:
                continue
            text = link_tag.text.split(',')[-1]
            chain = text.split()[1] # first word is always "chain"
            auth_chain = re.findall('\[auth ([A-Z])\]', text)
            if len(auth_chain) == 1:
                auth_chain = auth_chain[0]
            elif len(auth_chain) > 1:
                raise IndexError(f"more than one auth chain for {text} in {pdb_id}")
            else:
                auth_chain = chain
            chain_info[chain] = {"auth_chain": auth_chain, "url": url}

        # chain_ids = lig_cols[1].text.replace(' Less', '').split(',')
        # auth_id_lookup = dict()
        # for line in chain_ids:
        #     line = line.strip()
        #     chain = line.split()[0]
        #     auth_chain = re.findall('\[auth ([A-Z])\]',line)
        #     if len(auth_chain) == 1:
        #         auth_chain = auth_chain[0]
        #     elif len(auth_chain) > 1:
        #         raise IndexError(f"more than one auth chain for {line} in {pdb_id}")
        #     else:
        #         auth_chain = chain
        #     auth_id_lookup[chain] = {"chains": auth_chain, "links": lig_urls}
        ligand_chain_info[lig_name] = chain_info

    return ligand_chain_info
