import os
from prody import parsePDB, writePDB, matchChains, calcTransformation

def align_successful(out_dir, struct):
    if not os.path.exists('{0}/{1}/{1}_aligned.pdb'.format(out_dir, struct)):
        return False
    else:
        return True

def struct_align(template, structs, dist=15.0, retry=True,
                 filtered_protein='structures/processed/{pdbid}/{pdbid}_complex.pdb',
                 aligned_prot='structures/aligned/{pdbid}/{pdbid}_aligned.pdb',
                 align_dir='structures/aligned'):

    template_path = filtered_protein.format(pdbid=template)
    if not os.path.exists(template_path):
        print('template not processed', template_path)
        return

    template_st = parsePDB(template_path)
    template_liginfo_path = template_path.replace(f'processed/{template}', 'raw').replace('_complex.pdb','.info')
    temp_liginfo = open(template_liginfo_path,'r').readlines()
    if len(temp_liginfo[0].strip('\n')) < 4:
        selection_text = 'hetatm'
    else:
        selection_text = temp_liginfo[1]
    template_to_align = template_st.select(f'calpha within {dist} of {selection_text})')
    for struct in structs:
        query_path = filtered_protein.format(pdbid=struct)
        if align_successful(align_dir, struct):
            continue

        print('align', struct, template)

        os.system('mkdir -p {}'.format(align_dir))
        os.system('rm -rf {}/{}'.format(align_dir, struct))
        os.system('mkdir -p {}/{}'.format(align_dir, struct))

        _workdir = '{}/{}/'.format(align_dir, struct)
        _template_fname = '{}_template.pdb'.format(template)
        _query_fname = '{}_query.pdb'.format(struct)

        os.system('cp {} {}/{}'.format(template_path, _workdir, _template_fname))
        os.system('cp {} {}/{}'.format(query_path, _workdir, _query_fname))

        query = parsePDB(f'{_workdir}/{_query_fname}')
        query_liginfo_path = query_path.replace(f'processed/{struct}','raw').replace('_complex.pdb','.info')
        q_liginfo = open(query_liginfo_path,'r').readlines()
        if len(q_liginfo[0].strip('\n')) < 4:
            selection_text = 'hetatm'
        else:
            selection_text = q_liginfo[1]
        query_to_align = query.select(f'calpha within {dist} of {selection_text}')
        query_match, template_match, _ ,_ = matchChains(query_to_align,template_to_align,pwalign=True,seqid=10,overlap=10)[0]
        transform = calcTransformation(query_match,template_match)
        query_aligned = transform.apply(query)

        writePDB(aligned_prot.format(pdbid=struct),query_aligned)

        if retry and not align_successful(align_dir, struct):
            print('Alignment failed. Trying again with a larger radius.')
            struct_align(template, [struct], dist=25.0, retry=False,
                     filtered_protein=filtered_protein,aligned_prot=aligned_prot,
                     align_dir=align_dir)
