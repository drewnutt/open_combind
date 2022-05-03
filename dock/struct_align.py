import os
from pymol.cmd import load, select, align, save, delete

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

    load(f'{template_path}', 'target')
    select('target_to_align', f'(target and name CA) within {dist} of het')
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

        load(f'{_workdir}/{_query_fname}', 'query')
        select('query_to_align', f'(query and name CA) within {dist} of het')
        rms_aft, _,_, rms_bef, _, _, _ = align('query_to_align','target_to_align')

        if rms_bef == 0 or rms_aft < rms_bef:
            save(aligned_prot.format(pdbid=struct),'query')

        # Doesn't seem to overload objects, so just delete the query
        delete('query')

        if retry and not align_successful(align_dir, struct):
            print('Alignment failed. Trying again with a larger radius.')
            struct_align(template, [struct], dist=25.0, retry=False,
                     filtered_protein=filtered_protein,aligned_prot=aligned_prot,
                     align_dir=align_dir)
