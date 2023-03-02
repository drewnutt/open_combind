from pymol import cmd
from glob import glob

def load_complexes(protein, n=1):
    """
    USAGE

    load_complexes protein,[n=1]

    loads the first n protein ligand complexes from the directory [protein].
    """
    n = int(n)
    for prot in sorted(glob('{}/structures/proteins/*_prot.pdb'.format(protein)))[:n]:
        pdb = prot.split('/')[-1].split('_')[0]
        load_crystal_protein(protein, pdb)
        load_crystal_pose(protein, pdb)
        
    cmd.util.cbao("prot_*")
    cmd.util.cbay("het and crystal_*")
    cmd.show('sticks', "het and crystal_*")
    cmd.hide('lines', 'element h')
    cmd.show('spheres', 'het and prot* and (crystal* expand 5)')

    cmd.show('cartoon')
    cmd.set('cartoon_oval_length', '0.5')
    cmd.set('cartoon_transparency', '0.5')
    cmd.hide('everything', 'element H and not (element N+O extend 1)')
    cmd.hide('everything', 'name H')

def load_crystal_protein(protein, ligand):
    cmd.load('{}/structures/proteins/{}_prot.pdb'.format(protein, ligand))
    cmd.set_name('{}_prot'.format(ligand), 'prot_{}'.format(ligand))

def load_crystal_pose(protein, ligand):
    cmd.load('{}/structures/ligands/{}_lig.sdf'.format(protein, ligand, ligand))
    cmd.set_name('{}_lig'.format(ligand), 'crystal_{}'.format(ligand))

###################################################################
def load_pose(protein, ligand, pose, prefix):
    pv = glob('{}/dock_out/{}.sdf.gz'.format(protein, ligand))[0]
    lig_name = ligand.split('-to-')[0]
    
    cmd.load(pv,ligand)
    cmd.split_states(ligand)
    cmd.delete(ligand)
    if pose == 0:
        pose = ''
    elif pose < 10:
        pose = '0' + str(pose)
    else:
        pose = str(pose)
    print(ligand, pose, prefix, ligand)
    cmd.set_name(lig_name + pose, '{}_{}'.format(prefix, ligand))
    cmd.delete(lig_name + '*')

def load_top_docked(protein, n = 1):
    n = int(n)
    grid = None
    for prot in sorted(glob('{}/structures/proteins/*_prot.pdb'.format(protein)))[:n]:
        pdb = prot.split('/')[-1].split('_')[0]
        if grid is None:
            grid = pdb
        print(pdb)
        load_pose(protein, pdb, grid, 0, 'docked')
    cmd.show('sticks', "docked_*")
    cmd.hide('lines', 'element h')
    cmd.hide('everything', 'element H and not (element N+O extend 1)')

def load_results(protein, scores):
    """
    USAGE

    load_results protein, pose_prediction_csv

    load all results from the ComBind docking for the ligands in [pose_prediction_csv]. Additionally, loads crystal structures for non-CHEMBL ligands

    Single ligand docking results are in yellow, ComBind results are in cyan, and crystal structures are in white
    """
    struct = glob('{}/structures/template/*'.format(protein))[0].split('/')[-1].split('.')[0]
    load_crystal_protein(protein, struct)
    with open(scores) as fp:
        fp.readline()
        for line in fp:
            if line[:3] == 'com': continue
            (ligand,
             combind_rank, combind_rmsd,
             docked_rmsd,
             best_rmsd) = line.strip().split(',')
            load_pose(protein, ligand, int(combind_rank), 'combind')
            load_pose(protein, ligand, 0, 'docked')
            ligand = ligand.split('-to-')[0].replace('_lig','')
            if ligand[:6] != 'CHEMBL':
                load_crystal_pose(protein, ligand)
            cmd.group(ligand,f'*{ligand}*')

    cmd.show('sticks', "docked_*")
    cmd.show('sticks', "combind_*")
    cmd.show('sticks', "crystal_*")
    cmd.hide('lines', 'element h')
    cmd.hide('everything', 'element H and not (element N+O extend 1)')

    cmd.util.cbaw('*')
    cmd.color('yellow', 'docked* and element c')
    cmd.color('cyan', 'combind* and element c')
    cmd.set('stick_radius', '0.13')
    
    
    cmd.show('cartoon')
    cmd.set('cartoon_oval_length', '0.5')
    cmd.set('cartoon_transparency', '0.5')

###############################################################

## not sure what this stuff is for?
def parse_fp_file(fp_file):
    ifps = {}
    try:
        with open(fp_file) as f:
            pose_num = 0
            for line in f:
                if line.strip() == '': continue
                if line[:4] == 'Pose':
                    pose_num = int(line.strip().split(' ')[1])
                    ifps[pose_num] = {}
                    continue
                sc_key, sc = line.strip().split('=')
                i,r,ss = sc_key.split('-')
                i = int(i)
                sc = float(sc)
                prev_sc = ifps[(i, r)] if (i,r) in ifps[pose_num] else 0
                ifps[pose_num][(i,r)] = max(prev_sc, sc)

    except Exception as e:
        print(e)
        print(fp_file, 'fp not found')
    if len(ifps) == 0:
        print('check', fp_file)
        return {}
    return ifps

def show_interactions(protein, ligand, struct, ifp, pose):
    ifp_file = '{}/ifp/{}/{}_lig-to-{}-confgen_es4.fp'.format(protein, ifp, ligand, struct)
    print(ifp_file)
    ifp = parse_fp_file(ifp_file)[int(pose)]
    cmd.hide('labels')
    cmd.set('label_size', 50)
    for (i, r), score in ifp.items():
        if i not in [2, 3]: continue
        if score < 0.5: continue
        res = r.split(':')[1].split('(')[0]
        cmd.label('{}/ca'.format(res), score)
    
cmd.extend('load_complexes', load_complexes)
cmd.extend('load_top_docked', load_top_docked)
cmd.extend('load_results', load_results)
# cmd.extend('show_interactions', show_interactions)
