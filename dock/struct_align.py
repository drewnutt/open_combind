import os
from prody import parsePDB, writePDB, calcTransformation

# command = ('$SCHRODINGER/utilities/structalign '
#            '-asl "(not chain. L and not atom.element H) and (fillres within {0} chain. L)" '
#            '-asl_mobile "(not chain. L and not atom.element H) and (fillres within {0} chain. L)" '
#            '{1} {2}')

def align_successful(out_dir, struct):
    if not os.path.exists('{0}/{1}/{1}_aligned.pdb'.format(out_dir, struct)):
        return False
    else:
        return True
    
    # if os.path.exists('{}/{}/{}_template.mae'.format(out_dir, struct, struct)):
    #     return True # query = template so we don't need to check alignment

    # with open('{}/{}/align.out'.format(out_dir, struct), 'r') as f:
    #     for line in f:
    #         tmp = line.strip().split()
    #         if len(tmp) > 0 and tmp[0] == 'Alignment':
    #             if float(tmp[2]) > 0.4:
    #                 print('-- Alignment warning!', struct, float(tmp[2]))
    #                 return False
    #             return True
    #     else:
    #         print('alignment failure', struct)
    #         return False

def struct_align(template, structs, dist=15.0, retry=True,
                 filtered_protein='structures/processed/{pdbid}/{pdbid}_complex.pdb',
                 aligned_prot='structures/aligned/{pdbid}/{pdbid}_aligned.pdb',
                 align_dir='structures/aligned'):

    template_path = filtered_protein.format(pdbid=template)
    if not os.path.exists(template_path):
        print('template not processed', template_path)
        return

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

        # with open('{}/align_in.sh'.format(_workdir), 'w') as f:
        #     f.write(command.format(dist, _template_fname, _query_fname))
        # run('sh align_in.sh > align.out', shell=True, cwd=_workdir)
        
        query = parsePDB(f'{_workdir}/{_query_fname}')
        query_to_align = query.select(f'calpha and within {dist} of hetero')
        target = parsePDB(f'{_workdir}/{_template_fname}')
        target_to_align = target.select(f'calpha and within {dist} of hetero')
        t = calcTransformation(query_to_align, target_to_align)
        t.apply(query)
        writePDB(aligned_prot.format(pdbid=struct),query)

        if retry and not align_successful(align_dir, struct):
            print('Alignment failed. Trying again with a larger radius.')
            struct_align(template, [struct], dist=25.0, retry=False,
                     filtered_protein=filtered_protein,aligned_prot=aligned_prot,
                     align_dir=align_dir)
