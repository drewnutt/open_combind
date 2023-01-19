import pandas as pd
import argparse
import os
import sys
from glob import glob
sys.path.append(os.path.expanduser('~anm329/git/combind'))

# Defaults
mcss_version = 'mcss16'
shape_version = 'pharm_max'
ifp_version = 'rd1'

def get_unique_helper_ligs(helper_list_root):
    helper_ligs_affinity = pd.read_csv(f'{helper_list_root}helper_best_affinity_diverse.csv')
    helper_ligs_mcss = pd.read_csv(f'{helper_list_root}helper_best_mcss.csv')
    combined_helper_ligs = pd.concat([helper_ligs_affinity,helper_ligs_mcss])
    combined_helper_ligs.rename({'smiles':'SMILES','helper':'ID'},axis='columns',inplace=True)
    combined_hl_prot_gp = combined_helper_ligs.groupby('protein')

    for prot_name, gp in combined_hl_prot_gp:
        dedupped_prot = gp.drop_duplicates(subset=['SMILES'])
        print(f"{prot_name}:{dedupped_prot.shape[0]}")
        dedupped_prot.to_csv(f"combind_paper_dataset/{prot_name}/helper_ligands_{prot_name}.csv",
                sep=',',index=False, columns=['ID','SMILES'])

def query_to_helpers(query_ligand, prot_name, helper_list_root, selection_criterion="affinity"):
    if selection_criterion == 'affinity':
        helper_ligs_list = pd.read_csv(f'{helper_list_root}/helper_best_affinity_diverse.csv')
    elif selection_criterion == 'mcss':
        helper_ligs_list = pd.read_csv(f'{helper_list_root}/helper_best_mcss.csv')
    else:
        print(f"{selection_criterion} is not a valid selection criterion")
        return
    helper_for_query = helper_ligs_list[(helper_ligs_list['protein'] == prot_name) & (helper_ligs_list['query'] == query_ligand)]

    return helper_for_query['helper'].tolist()

def run_featurization(root, helper_ligands, query_fname, protein_name, helper_list_root, native_loc='.', selection_criterion="affinity",processes=1):
    query_ligand = query_fname.split('/')[-1].split('-')[0].split('_')[0]
    helpers_to_use = query_to_helpers(query_ligand, protein_name, helper_list_root, selection_criterion)
    helper_ligands = [ligand for ligand in helper_ligands if ligand.split('/')[-1].split('-')[0] in helpers_to_use] + [query_fname]
    print(helper_ligands)
    native_poses = {}
    for native_path in native_loc:
        name = native_path.split('/')[-1].replace('.sdf', '')
        # sts = Chem.SDMolSupplier(native_path)
        native_poses[name] = native_path
    print(native_poses)
    return featurize(root, helper_ligands, native_poses, ifp_version, mcss_version, shape_version, False, False, processes, 100, False)


def featurize(root, poseviewers, native_loc, ifp_version, mcss_custom,
              shape_version, no_mcss, use_shape, processes, max_poses, no_cnn):
    from open_combind.features.features import Features
    if use_shape:
        print("Shape is not currently implemented outside of Schrodinger\n Shape has not been evaluated for performance in pose-prediction")

    features = Features(root, ifp_version=ifp_version, shape_version=shape_version,
                        max_poses=max_poses, cnn_scores=not no_cnn)

    features.compute_single_features(poseviewers,native_poses=native_loc)

    features.compute_pair_features(poseviewers,
                                   mcss=not no_mcss, shape=use_shape, processes=processes)

    return features

def merge_correct_stats(prot_name,stats_root, interactions):
    stats_root = stats_root + "/protein_statistics"
    merged_stats = stats_root + '/merged_combind_stats_excl_' + prot_name + '/{}_{}.txt'
    if len(glob(merged_stats.replace('{}','*'))) != len(interactions) * 2:
        from open_combind.score.statistics import merge_stats
        all_prot_names = ['/'.join(directory.split('/')[-2:]).replace('/','') for directory in glob(f"{stats_root}/*/") if "stats" not in directory]
        correct_stats = [prot for prot in all_prot_names if prot != prot_name ]
        correct_stats.sort()
        print(correct_stats)
        if not os.path.isdir(os.path.dirname(merged_stats)):
            os.makedirs(os.path.dirname(merged_stats))
        merge_stats(correct_stats, stats_root, merged_stats, interactions)

    return '/'.join(merged_stats.split('/')[:-1])

def pose_prediction(prot_features, out, stats_root, alpha=-0.6,
                    features='mcss,hbond,saltbridge,contact', restart=500, max_iterations=1000):
    """
    Run ComBind pose prediction.
    """
    from open_combind.score.pose_prediction import PosePrediction
    from open_combind.score.statistics import read_stats
    from open_combind.features.features import Features

    if isinstance(features,str):
        features = features.split(',')

    ligands = set(prot_features.raw['name1'])
    ligands = sorted(ligands)

    data = prot_features.get_view(ligands, features)
    stats = read_stats(stats_root, features)
    
    ps = PosePrediction(ligands, features, data, stats, alpha)
    best_poses = ps.max_posterior(max_iterations, restart)

    with open(out, 'w') as fp:
        fp.write('ID,POSE,COMBIND_RMSD,GNINA_RMSD,BEST_RMSD\n')
        for ligand in best_poses:
            rmsds = data['rmsd'][ligand]
            grmsd = rmsds[0]
            crmsd = rmsds[best_poses[ligand]]
            brmsd = min(rmsds)
            fp.write(','.join(map(str, [ligand.replace('_pv', ''),
                                        best_poses[ligand],
                                        crmsd, grmsd, brmsd]))+ '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_root', required=True, help='directory to place features in')
    parser.add_argument('--helper_ligands',nargs='*', required=True, help='helper ligand docked poses')
    parser.add_argument('--query_ligand', required=True,help='query ligand docked poses')
    parser.add_argument('--protein_name', required=True, help='name of protein class')
    parser.add_argument('--stats_root',required=True,help='Directory that contains helper lists and protein statistics are in `<stats_root>/protein_statistics`')
    parser.add_argument('--merged_stats_root',default=None,help='Directory that contains merged protein statistics')
    parser.add_argument('--selection_criterion',choices=['affinity','mcss'], default='affinity',help='how to select the helper ligands')
    parser.add_argument('--native', nargs='*', default=[],help='location of native poses')
    parser.add_argument('--interactions',default='mcss,hbond,saltbridge,contact',help='interactions to use for featurization and pose prediction')
    parser.add_argument('--processes',type=int,default=1,help='# of processes to use')
    parser.add_argument('--pose_csv',help='name of pose_csv to use, if not set then `<protein_name>_<query_ligand>_<selection_criterion>.csv`')

    args = parser.parse_args()

    interactions = args.interactions.split(',')
    if 'mcss' in interactions:
        no_mcss = False
    prot_features = run_featurization(args.feat_root,args.helper_ligands,args.query_ligand,args.protein_name,
            args.stats_root,native_loc=args.native, selection_criterion=args.selection_criterion,processes=args.processes)
    if args.pose_csv is None:
        args.pose_csv = f"{args.protein_name}_{args.query_ligand.split('/')[-1].split('-')[0].split('_')[0]}_{args.selection_criterion}.csv"
    if args.merged_stats_root is None:
        print("merging stats")
        args.merged_stats_root = merge_correct_stats(args.protein_name,args.stats_root,interactions)
    prot_features.load_features()
    pose_prediction(prot_features,args.pose_csv,args.merged_stats_root,features=interactions)
