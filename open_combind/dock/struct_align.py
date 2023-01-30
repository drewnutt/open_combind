import os
import numpy as np
from prody import parsePDB, writePDB, matchChains, calcTransformation
from rdkit.Chem import ForwardSDMolSupplier, SDWriter
from rdkit.Chem.rdMolTransforms import TransformConformer

def align_successful(out_dir, struct):
    if not os.path.isfile('{0}/{1}/{1}_aligned.pdb'.format(out_dir, struct)):
        return False
    else:
        return True

def align_separate_ligand(struct, trans_matrix,
        downloaded_ligand="structures/processed/{pdbid}/{pdbid}_lig.sdf",
        aligned_lig = "structures/aligned/{pdbid}/{pdbid}_lig.sdf"):
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
    .. include ::<isotech.txt>
    Align protein-ligand complexes based on the atoms less than `dist` |angst| from the ligand heavy atoms.

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
    ` ``ProDy.Transformation`` <http://prody.csb.pitt.edu/manual/reference/measure/transform.html#module-prody.measure.transform>`_
        Transformation object of the last alignment performed
    """

    template_path = filtered_protein.format(pdbid=template)
    if not os.path.isfile(template_path):
        print('template not processed', template_path)
        return

    template_st = parsePDB(template_path)
    template_liginfo_path = template_path.replace(f'processed/{template}', 'raw').replace('_complex.pdb', '.info')
    selection_text, templ_lig_chain = get_selection_texts(template_liginfo_path, template_st)
    templ_prot_chain = template_st.select(f'not {selection_text} and (chain {templ_lig_chain} within {dist} of {selection_text}) and heavy')
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
        query_liginfo_path = query_path.replace(f'processed/{struct}', 'raw').replace('_complex.pdb', '.info')

        selection_text, query_lig_chain = get_selection_texts(query_liginfo_path, query)
        # query_to_align = query.select(f'calpha within {dist} of {selection_text}')
        # print(f'not {selection_text} and (chain {query_lig_chain} within {dist} of {selection_text}) and heavy')
        query_prot_chain = query.select(f'not {selection_text} and (chain {query_lig_chain} within {dist} of {selection_text}) and heavy')
        try:
            # query_match, template_match, _, _ = matchChains(query_to_align, template_to_align, pwalign=True,
            #                                                 seqid=1, overlap=1)[0]
            # query_match, template_match, _, _ = matchChains(query_prot_chain, templ_prot_chain, pwalign=True,
            #                                                 seqid=1, overlap=1)[0]
            matches = matchChains(query_prot_chain, templ_prot_chain, pwalign=True,
                                                            seqid=1, overlap=1)
            # print([[match[2],match[3]] for match in matches])
            query_match, template_match, _, _ = matches[0]
        except IndexError as ie:
            print(str(ie))
            query_match, template_match, _, _ = matchChains(query_prot_chain, templ_prot_chain, pwalign=False,
                                                            seqid=1, overlap=1)[0]
        # if len(query_match) < 0.5 * min(len(query_prot_chain), len(templ_prot_chain)):
        #     # print(len(query_match))
        #     print(f"WARNING: Bad quality chain alignment of {struct}, "
        #             "trying with only protein atoms on ligand chain")
            # print(len(query_match))
        transform = calcTransformation(query_match, template_match)
        query_aligned = transform.apply(query)

        transform_matrix = transform.getMatrix()

        writePDB(aligned_prot.format(pdbid=struct), query_aligned)

        if retry and not align_successful(align_dir, struct):
            print('Alignment failed. Trying again with a larger radius.')
            transform_matrix = struct_align(template, [struct], dist=15.0, retry=False,
                     filtered_protein=filtered_protein,aligned_prot=aligned_prot,
                     align_dir=align_dir)
        
        aligned_lig = align_separate_ligand(struct, transform_matrix,
                downloaded_ligand= filtered_protein.replace("_complex.pdb","_lig.sdf"),
                aligned_lig= align_dir+"/{pdbid}/{pdbid}_lig.sdf")
        if aligned_lig:
            print("Successfully aligned separate ligand")
        else:
            print("No separate ligand found to align")

    return transform_matrix

def get_selection_texts(liginfo_path, prot):
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
