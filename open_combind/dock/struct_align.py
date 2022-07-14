import os
from prody import parsePDB, writePDB, matchChains, calcTransformation
from rdkit.Chem import ForwardSDMolSupplier, SDWriter
from rdkit.Chem.rdMolTransforms import TransformConformer

def align_successful(out_dir, struct):
    if not os.path.isfile('{0}/{1}/{1}_aligned.pdb'.format(out_dir, struct)):
        return False
    else:
        return True

def align_separate_ligand(ligand_path, trans_matrix, transformed_lig_path):
    """
    Transform the the ligand using the provided transformation matrix

    Parameters
    ----------
    ligand_path : str
        Path to the ligand SDF file that needs to be transformed
    trans_matrix : ``ndarray``
        Transformation matrix describing the transformation of the ligand
    transformed_lig_path : str
        Path to the transformed ligand SDF file for output

    Returns
    -------
    bool
        If the ligand file existed and the transformation was performed
    """
    print(ligand_path)
    if not os.path.isfile(ligand_path):
        return False

    lig_mol = next(ForwardSDMolSupplier(ligand_path))
    TransformConformer(lig_mol.GetConformer(), trans_matrix)

    print("writing lig")
    writer = SDWriter(transformed_lig_path)
    writer.write(lig_mol)
    writer.close()
    return True
    

def struct_align(template, structs, dist=15.0, retry=True,
                 filtered_protein='structures/processed/{pdbid}/{pdbid}_complex.pdb',
                 ligand_info='structures/raw/{pdbid}.info',
                 aligned_prot='{pdbid}_aligned.pdb',
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
    aligned_prot : str, default='{pdbid}_aligned.pdb'
        Format string for the new filename of the aligned protein-ligand complex given the PDB ID as `pdbid`
    align_dir : str, default='structures/aligned'
        Path to the aligned protein directory, all parent directories will be created if they do not exist

    Returns
    -------
    `ProDy.measure.Transformation`
        Transformation object of the last alignment performed


    .. include :: <isotech.txt>
    """

    template_path = filtered_protein.format(pdbid=template)
    if not os.path.isfile(template_path):
        print('template not processed', template_path)
        return
    # .. |angst| replace:: \u212B

    template_st = parsePDB(template_path)
    # template_liginfo_path = template_path.replace(f'processed/{template}', 'raw').replace('_complex.pdb','.info')
    template_liginfo_path = ligand_info.format(pdbid=template)
    temp_liginfo = open(template_liginfo_path,'r').readlines()
    if len(temp_liginfo[0].strip('\n')) < 4:
        selection_text = 'hetatm'
    else:
        selection_text = temp_liginfo[1]
    template_to_align = template_st.select(f'calpha within {dist} of {selection_text}')
    transform_matrix = 0
    for struct in structs:
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

        query = parsePDB(f'{_workdir}/{_query_fname}')
        # query_liginfo_path = query_path.replace(f'processed/{struct}','raw').replace('_complex.pdb','.info')
        query_liginfo_path = ligand_info.format(pdbid=struct)
        q_liginfo = open(query_liginfo_path, 'r').readlines()
        if len(q_liginfo[0].strip('\n')) < 4:
            selection_text = 'hetatm'
        else:
            selection_text = q_liginfo[1]
        query_to_align = query.select(f'calpha within {dist} of {selection_text}')
        try:
            query_match, template_match, _, _ = matchChains(query_to_align, template_to_align, pwalign=True, seqid=10, overlap=10)[0]
        except IndexError:
            query_match, template_match, _, _ = matchChains(query_to_align,template_to_align,pwalign=False,seqid=10,overlap=10)[0]
        transform = calcTransformation(query_match,template_match)
        query_aligned = transform.apply(query)

        transform_matrix = transform.getMatrix()

        writePDB(align_dir+f"/{struct}/"+aligned_prot.format(pdbid=struct), query_aligned)

        if retry and not align_successful(align_dir, struct):
            print('Alignment failed. Trying again with a larger radius.')
            transform_matrix = struct_align(template, [struct], dist=25.0, retry=False,
                     filtered_protein=filtered_protein, aligned_prot=aligned_prot,
                     align_dir=align_dir)
        
        aligned_lig = align_separate_ligand(filtered_protein.replace("_complex.pdb", "_lig.sdf").format(pdbid=struct),
                transform_matrix, (align_dir+"/{pdbid}/{pdbid}_lig.sdf").format(pdbid=struct))
        if aligned_lig:
            print("Successfully aligned separate ligand")
        else:
            print("No separate ligand found to align")

    return transform_matrix

