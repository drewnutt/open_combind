#!/usr/bin/env python3
import argparse
from glob import glob
import numpy as np
from open_combind.features.features import Features
from scipy.stats import linregress
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def get_xy_from_feats(feats,ligand_names,score='gscore',sigmoid=True):
    all_poses = []
    for i, ligand in enumerate(ligand_names):
        for s, rmsd in zip(feats[score][ligand],feats['rmsd'][ligand]):
            if sigmoid:
                s= sigmoid(s)            
            pose_good = 0
            if rmsd <= 2:
                pose_good = 1
            all_poses.append((s,pose_good))
    return all_poses

def nll_to_ncorrect(nll):
    neg_exp =np.exp(-nll)
    return 100*neg_exp / (1+neg_exp)

def compare_to_nll(score_pose_arr,col='gscore',xlims=None):
    sorting_order = np.argsort(score_pose_arr[:,0],axis=0)
    X = score_pose_arr[:,0]
    y = score_pose_arr[:,-1]

    groups_stats = []
    score_sum = 0
    good_pose = 0
    count = 0
    for i, sort_idx in enumerate(sorting_order):
        score, pose_score = score_pose_arr[sort_idx,[0,-1]]
        if pose_score:
            good_pose += 1
        count += 1
        score_sum += score
        if (i + 1) % 100 == 0:
            # print(i)
            assert count == 100
            groups_stats.append((score_sum/100,-np.log1p(good_pose)+np.log1p(100-good_pose)))
            count = 0
            score_sum = 0
            good_pose = 0


    x,y = [],[]
    for score, nll in groups_stats:
        if np.isinf(nll):
            continue
        x.append(score)
        y.append(nll)

    a,b, r, tt, stderr = linregress(np.array(x),y)

    return x,y,a,b

def poses_to_arr(root,seed=42,feature_dir_bn="features_stats_",**kwargs):
    proteins = [dirname.split('/')[-2] for dirname in glob(f'{root}/*/') if 'stats' not in dirname and 'test' not in dirname]
    all_poses = []
    for protein in proteins:
        print(protein)
        features = Features(f'{root}/{protein}/{feature_dir_bn}{seed}/',max_poses=100)
        features.load_features()
        ligand_names = sorted(set(features.raw['name1']))
        interactions = []
        feats = features.get_view(ligand_names,interactions)
        prot_poses = get_xy_from_feats(feats,ligand_names,**kwargs)
        all_poses += prot_poses

    return np.asarray(all_poses)

def plot_figure(xvals, yvals, slope, intercept, figname):
    xlims = [min(xvals), max(xvals)]
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(x,y,alpha=0.5)
    xx = np.linspace(*xlims, 500)
    yy = m*xx+b
    plt.plot(xx, yy, '--', label=f'm={m:.2f},b={b:.2f}')
    plt.xlabel('Average Score')
    plt.ylabel('NLL of the cluster being correct')
    plt.legend()
    plt.savefig(figname, dpi='figure')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Determine Alpha for Open-ComBind given a set of docked poses')
    parser.add_argument('root',type=str,help='Root directory of the dataset')
    parser.add_argument('--seed',type=int,default=42,help='Seed used for the dataset')
    parser.add_argument('--feature_dir_bn',type=str,default='features_stats_',help='Base name of the feature directory')
    # add an argument for the output directory
    parser.add_argument('--newscore',action='store_true',help='Use the new scoring function')
    parser.add_argument('--sigmoid',action='store_true',help='Use sigmoid on the scores')
    parser.add_argument('--output',type=str,default='.',help='Output directory for the data and plots')
    parser.add_argument('--figname',type=str,help='Name of the figure')
    args = parser.parse_args()

    score_pose_arr = poses_to_arr(args.root,seed=args.seed,feature_dir_bn=args.feature_dir_bn,score='gscore' if not args.newscore else 'vaff',sigmoid=args.sigmoid)
    x,y, m, b = compare_to_nll(score_pose_arr,col='gscore' if not args.newscore else 'vaff')

    print(f'm={m}, b={b}')
    if args.figname:
        plot_figure(x,y,m,b,f"{args.output}/{args.figname}")

    


