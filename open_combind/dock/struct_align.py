import os
import numpy as np
from pymol.cmd import load, select, align, save, delete, get_object_matrix, reinitialize, get_chains
from rdkit.Chem import ForwardSDMolSupplier, SDWriter
from rdkit.Chem.rdMolTransforms import TransformConformer

def align_successful(out_dir, struct):
    """
    Check if a given PDB ID has been aligned

    Only checks for the presence of the file. Does not check if the file is valid.

    Parameters
    ----------
    out_dir : str
        Path to the aligned protein directory
    struct : str
        PDB ID of the protein-ligand complex to check
    
    Returns
    -------
    bool
        If the structure with the provided PDB ID has been aligned
    """

    if not os.path.isfile('{0}/{1}/{1}_aligned.pdb'.format(out_dir, struct)):
        return False
    else:
        return True

def align_separate_ligand(struct, trans_matrix,
        downloaded_ligand="structures/processed/{pdbid}/{pdbid}_lig.sdf",
        aligned_lig = "structures/aligned/{pdbid}/{pdbid}_lig.sdf"):
    """
    Transform the the ligand using the provided transformation matrix and write it to a new file.

    Parameters
    ----------
    struct : str
		PDB ID of the original complex
    trans_matrix : :class:`~numpy.ndarray`
        Transformation matrix describing the transformation of the ligand
	downloaded_ligand : str, default="structures/processed/{pdbid}/{pdbid}_lig.sdf"
		Format string for the path to the ligand SDF file given the PDB ID as `pdbid`
    aligned_lig: str, default="structures/aligned/{pdbid}/{pdbid}_lig.sdf"
        Format string for the path to the transformed ligand SDF file for output

    Returns
    -------
    bool
        If the ligand file existed and the transformation was performed
    """
    ligand_path = downloaded_ligand.format(pdbid=struct)
    # print(ligand_path)
    if not os.path.isfile(ligand_path):
        return False

    lig_mol = next(ForwardSDMolSupplier(ligand_path))
    TransformConformer(lig_mol.GetConformer(), trans_matrix)

    # print("writing lig")
    writer = SDWriter(aligned_lig.format(pdbid=struct))
    writer.write(lig_mol)
    writer.close()
    return True
    

def struct_align(template, structs, dist=15.0, retry=True,
                 filtered_protein='structures/processed/{pdbid}/{pdbid}_complex.pdb',
                 ligand_info='structures/raw/{pdbid}.info',
                 aligned_prot='structures/aligned/{pdbid}/{pdbid}_aligned.pdb',
                 align_dir='structures/aligned'):
    """
    Align protein-ligand complexes based on the atoms less than `dist` |angst| from the ligand heavy atoms.
    
    
    Aligned files are put in :file:`{align_dir}/{<PDB ID>}/`

    If a separate ligand exists for a given PDB ID, then it will be transformed with the same matrix as used for the protein alignment.

    Parameters
    ----------
    template : str
        PDB ID of the protein-ligand complex to align all of the other complexes to
    structs : iterable of str
        PDB IDs of the protein-ligand complexes to align to `template`
    dist : float,default=15.0
        Distance, in |angst|, from the ligand for which to select the alignment atoms of the protein
    retry : bool,default=True
        If alignment is unsuccessful, try again with a `dist` of 25 |angst|
    filtered_protein : str,default='structures/processed/{pdbid}/{pdbid}_complex.pdb'
        Format string for the path to the protein-ligand complexes given the PDB ID as `pdbid`
    ligand_info : str, default='structures/raw/{pdbid}.info'
        Format string for the path to the ligand `.info` files given the PDB ID as `pdbid`
    aligned_prot : str, default='structures/aligned/{pdbid}/{pdbid}_aligned.pdb'
        Format string for the path to the output, aligned protein-ligand complex given the PDB ID as `pdbid`
    align_dir : str, default='structures/aligned'
        Path to the aligned protein directory, all parent directories will be created if they do not exist

    Returns
    -------
    :class:`~numpy.ndarray`
        Transformation matrix of the last alignment performed


    .. include :: <isotech.txt>
    """

    template_path = filtered_protein.format(pdbid=template)
    if not os.path.isfile(template_path):
        print('template not processed', template_path)
        return

    if retry:
        reinitialize()
        load(f'{template_path}', 'target')
        template_liginfo_path = template_path.replace(f'processed/{template}', 'raw').replace('_complex.pdb', '.info')
        selection_text, templ_lig_chain = get_selection_texts(template_liginfo_path, 'target')
        select('target_to_align', f'(target and not hydrogens and not hetatm) within {dist} of (( {selection_text} ) and target )')

    for struct in structs:
        transform_matrix = np.identity(4)
        query_path = filtered_protein.format(pdbid=struct)
        if align_successful(align_dir, struct):
            continue

        print('align', struct, template)

        os.system('rm -rf {}/{}'.format(align_dir, struct))
        os.system('mkdir -p {}/{}'.format(align_dir, struct))

        _workdir = '{}/{}/'.format(align_dir, struct)
        _template_fname = '{}_template.pdb'.format(template)
        _query_fname = '{}_query.pdb'.format(struct)

        os.system('cp {} {}/{}'.format(template_path, _workdir, _template_fname))
        os.system('cp {} {}/{}'.format(query_path, _workdir, _query_fname))

        load(f'{_workdir}/{_query_fname}', 'query')
        query_liginfo_path = query_path.replace(f'processed/{struct}', 'raw').replace('_complex.pdb', '.info')

        selection_text, query_lig_chain = get_selection_texts(query_liginfo_path, 'query')
        select('query_to_align', f'(query and not hydrogens and not hetatm) within {dist} of (( {selection_text} ) and query )')

        rms_aft, _,_, rms_bef, _, _, _ = align('query_to_align','target_to_align')

        transform_matrix = get_object_matrix('query')
        assert transform_matrix is not None
        transform_matrix = np.array(transform_matrix).reshape(4,4)
        if rms_bef == 0 or rms_aft < rms_bef:
            save(aligned_prot.format(pdbid=struct),'query')

        delete('query')
        delete('query_to_align')

        if retry and not align_successful(align_dir, struct):
            print('Alignment failed. Trying again with a larger radius.')
            transform_matrix = struct_align(template, [struct], dist=15.0, retry=False,
                     filtered_protein=filtered_protein,aligned_prot=aligned_prot,
                     align_dir=align_dir)
        
        if retry:
            aligned_lig = align_separate_ligand(struct, transform_matrix,
                    downloaded_ligand= filtered_protein.replace("_complex.pdb","_lig.sdf"),
                    aligned_lig= align_dir+"/{pdbid}/{pdbid}_lig.sdf")
        if aligned_lig:
            print("Successfully aligned separate ligand")
        else:
            print("No separate ligand found to align")

    return transform_matrix

def get_selection_texts(liginfo_path, prot):
    """
    Get the selection text for the ligand and the chain of the ligand

    Selects the ligand chain as the protein chain that is within 5 |angst| of the ligand

    Parameters
    ----------
    liginfo_path : str
        Path to the ligand info file
    prot : :class:`~prody.atomic.atomgroup.AtomGroup`
        Protein structure

    Returns
    -------
    selection_text : str
        Selection text for the ligand
    lig_chain : str
        Chain of the ligand
    """

    liginfo = open(liginfo_path, 'r').readlines()
    if len(liginfo[0].strip('\n')) < 4:
        selection_text = 'hetatm'
    else:
        selection_text = liginfo[1].strip()
    # lig_chain = prot.select(selection_text).getChids()[0]
    # if selection_text == f'chain {lig_chain}':
    unique_chains, counts = np.unique(prot.select(f'not {selection_text} and protein within 5 of {selection_text}').getChids(), return_counts=True)
    lig_chain = unique_chains[counts.argmax()]

    return selection_text, lig_chain
