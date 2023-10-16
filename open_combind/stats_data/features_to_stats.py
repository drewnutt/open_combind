import argparse
import os.path
from glob import glob
from open_combind.score.statistics import pair_features,compute_stats, merge_stats, read_stats
from open_combind import featurize
import matplotlib.pyplot as plt


def get_args():
    parser =argparse.ArgumentParser()
    parser.add_argument('--stats_dir','-D',help="which directory to use to write all the stats")
    parser.add_argument('--interactions','-I',default=['hbond','saltbridge','contact','mcss','fg_simi'],nargs="+",help="what interactions to calculate statistics for")
    parser.add_argument('--stats_figure',help="file name of statistics figure")
    parser.add_argument('--suffix',default="",help="suffix of features directory")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if not os.path.isdir(args.stats_dir):
        os.mkdir(args.stats_dir)
    for directory in sorted(glob("*/")):
        if not os.path.isdir(f"{directory}/features_stats{args.suffix}"):
            if os.path.isdir(f"{directory}/dock_out"):
                print(directory)
            continue
        dirname = directory.replace('/','')
        if os.path.isfile(f'{args.stats_dir}/{dirname}/reference_mcss.de'):
            continue
        print(dirname)
        if not os.path.isfile(f'{args.stats_dir}/{dirname}.csv'):
            pair_features(dirname,'.',args.stats_dir,interactions=args.interactions,features_dir=f"features_stats{args.suffix}")
        compute_stats(dirname,args.stats_dir,args.stats_dir,args.interactions)

    protnames = [directory.replace('/','') for directory in glob('*/') if os.path.isdir(f"{directory}/features_stats{args.suffix}")]
    if not os.path.isdir(f"{args.stats_dir}/merged_combind_stats/"):
        os.mkdir(f"{args.stats_dir}/merged_combind_stats/")
    merge_stats(protnames,args.stats_dir,args.stats_dir + '/merged_combind_stats/{}_{}.txt',args.interactions)

    stats_dict_num200 = read_stats(f'{args.stats_dir}/merged_combind_stats/',args.interactions)
    fig, axes = plt.subplots(1,len(args.interactions))
    fig.set(size_inches=(16,4))
    for idx, feature in enumerate(args.interactions):
        axes[idx].set_title(feature)
        for dist, color in [('native','orange'),('reference','k')]:
            axes[idx].plot(stats_dict_num200[feature][dist].x,stats_dict_num200[feature][dist].fx, color=color)
    plt.savefig(args.stats_figure)
