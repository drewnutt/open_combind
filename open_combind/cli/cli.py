#!/bin/env python3

import pandas as pd
import numpy as np
import click
import os
import sys
from glob import glob

from open_combind.utils import *
import open_combind as oc
###############################################################################

# Defaults
SHAPE_VERSION = 'pharm_max'
IFP_VERSION = 'rd1'


class RunGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        # TODO: check if cmd_name is a file in the current dir and not require `run`?
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        return None

@click.command(cls=RunGroup, invoke_without_command=True)
@click.version_option(version=oc.__version__)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument('struct', default='')
@click.option('--templ-struct')
def structprep(templ_struct='', struct='', raw_dir='structures/raw',
        align_dir='structures/aligned', processed_dir='structures/processed',
        template_dir='structures/dir', ligand_dir='structures/ligands',
        protein_dir='structures/proteins' ):
    """
    Prepare structures and make a docking template file.

    "struct" specifies the name of the structure for which to make a docking
    template file. (Not the full path, generally just the PDB code.) Defaults to the
    structure with alphabetically lowest name.

    "templ_struct" specifies the name of the structure to use as a template for
    docking. Defaults to the structure with alphabetically lowest name.

    raw_dir, align_dir, processed_dir, template_dir, ligand_dir, and protein_dir
    specify the directories where the raw structures, aligned structures,
    processed structures, docking templates, ligands, and proteins will be
    written, respectively. Defaults to "structures/raw", "structures/aligned",
    "structures/processed", "structures/templates", "structures/ligands", and
    "structures/proteins", respectively.

    The following directory structure is required:

            <raw_dir>/
                structure_name.pdb
                structure_name.info (first line is Resname of ligand)
                ...
            <processed_dir>/
                structure_name/structure_name_prot.pdb
                ...
            <align_dir>/
                structure_name/structure_name_aligned.pdb
                ...
            <protein_dir>/
                structure_name_prot.pdb
                ...
            <ligand_dir>/
                structure_name_lig.pdb
                ...
            <template_dir>/
                structure_name/structure_name
                ...

    The process can be started from any step, e.g. if you have processed
    versions of your structures, you can place these in the processed directory.

    Files ending with _lig contain only the small molecule ligand present in the
    structure, and files ending with _prot contain everything else.
    """
    oc.structprep(templ_struct=templ_struct, struct=struct, raw_dir=raw_dir,
            align_dir=align_dir, processed_dir=processed_dir,
            template_dir=template_dir, ligand_dir=ligand_dir,
            protein_dir=protein_dir)


@cli.command()
@click.argument('smiles')
@click.option('--root', default='ligands')
@click.option('--multiplex', is_flag=True)
@click.option('--sdffile', is_flag=True)
@click.option('--ligand-names', default='ID')
@click.option('--ligand-smiles', default='SMILES')
@click.option('--delim', default=',')
@click.option('--num-out-confs', default=10)
@click.option('--num-confs', default=50)
@click.option('--confgen', default='etkdg_v2')
@click.option('--ff', default="UFF")
@click.option('--max-iterations', default=1000)
@click.option('--seed', default=-1)
@click.option('--processes', default=1)
def ligprep(smiles, root, multiplex, ligand_names, ligand_smiles,
            delim, processes, sdffile, num_out_confs, num_confs, confgen,
            max_iterations, seed, ff):
    """
    Prepare ligands for docking, from smiles or sdf.

    Specifically, this will utilize RDKit to generate a 3D conformer for each of the ligands 

    "smiles" should be a `delim` delimited file with columns "ligand-names"
    and "ligand-smiles". Alternatively, "smiles" can be a SDF file with multiple
    ligands as different entries, but you must specify "--sdffile".

    "root" specifies where the processed ligands will be written.

    By default, an individual file will be made for each ligand. If multiplex is
    set, then only one file, containing all the ligands, will be produced.

    Multiprocessing is only supported for non-multiplexed mode.
    """
    oc.ligprep(smiles, root=root, multiplex=multiplex, ligand_names=ligand_names,
            ligand_smiles=ligand_smiles, delim=delim, sdffile=sdffile,
            num_out_confs=num_out_confs, num_confs=num_confs, confgen=confgen,
            max_iterations=max_iterations, ff=ff, processes=processes, seed=seed)


@cli.command()
@click.argument('ligands', nargs=-1)
@click.option('--root', default='docking')
@click.option('--template')
@click.option('--screen', is_flag=True)
@click.option('--slurm', is_flag=True)
@click.option('--now', is_flag=True)
@click.option('--dock-file')
def dock_ligands(template, root, ligands, screen, slurm, now, dock_file):
    """
    Dock "ligands" to "grid".

    "root" specifies where the docking results will be written.

    Setting "screen" limits the thoroughness of the pose sampling. Recommended
    for screening, but not pose prediction.

    "ligands" are paths to prepared ligand files. Multiple can be specified.

    "dock_file" is a format string that will be used with all of the ligands to 
    create a docking file. The default looks like:
     "-l {lig} -o {out} --exhaustiveness {exh} --num_modes 200 > {log} \n"
    """
    oc.dock_ligands(ligands, template=template, dock_file=dock_file, root=root, screen=screen, slurm=slurm, now=now)

################################################################################


@cli.command()
@click.argument('root')
@click.argument('poseviewers', nargs=-1)
@click.option('--native', default='structures/ligands/*_lig.sdf')
@click.option('--ifp-version', default=IFP_VERSION)
@click.option('--shape-version', default=SHAPE_VERSION)
@click.option('--screen', is_flag=True)
@click.option('--max-poses', default=100)
@click.option('--no-mcss', is_flag=True)
@click.option('--no-cnn', is_flag=True)
@click.option('--use-shape', is_flag=True)
@click.option('--processes', default=1)
@click.option('--template', default='structures/template/*.template')
@click.option('--check-center-ligs', is_flag=True)
def featurize(root, poseviewers, native, ifp_version,
            shape_version, screen, no_mcss,
            use_shape, processes, max_poses, no_cnn, template, check_center_ligs):

    oc.featurize(root, poseviewers, native=native, no_mcss=no_mcss, use_shape=use_shape,
                max_poses=max_poses, no_cnn=no_cnn, screen=screen, ifp_version=ifp_version,
                shape_version=shape_version, processes=processes, template=template, check_center_ligs=check_center_ligs)
################################################################################


@cli.command()
@click.argument('root')
@click.argument('out', default="poses.csv")
@click.argument('ligands', nargs=-1)
@click.option('--features', default='mcss,hbond,saltbridge,contact')
@click.option('--alpha', default=1.0)
@click.option('--stats-root', default=None)
@click.option('--restart', default=500)
@click.option('--max-iterations', default=1000)
def pose_prediction(root, out, ligands,
                    alpha, stats_root, features, restart,
                    max_iterations):
    """
    Run ComBind pose prediction.
    """
    features = features.split(',')

    oc.pose_prediction(root, out=out, ligands=ligands, features=features,
            alpha=alpha, stats_root=stats_root, restart=restart, max_iterations=max_iterations)

# @main.command()
# @click.argument('score-fname')
# @click.argument('root')
# @click.option('--stats-root', default=stats_root)
# @click.option('--alpha', default=1.0)
# @click.option('--features', default='shape,hbond,saltbridge,contact')
# def screen(score_fname, root, stats_root, alpha, features):
#     """
#     Run ComBind screening.
#     """
#     from open_combind.score.screen import screen, load_features_screen
#     from open_combind.score.statistics import read_stats

#     features = features.split(',')
#     stats = read_stats(stats_root, features)
#     single, raw = load_features_screen(features, root)

#     combind_energy = screen(single, raw, stats, alpha)
#     np.save(score_fname, combind_energy)

################################################################################

# @main.command()
# @click.argument('scores')
# @click.argument('original_pvs', nargs=-1)
# def extract_top_poses(scores, original_pvs):
#     """
#     Write top-scoring poses to a single file.
#     """
#     out = scores.replace('.csv', '.sdf.gz')
#     scores = pd.read_csv(scores).set_index('ID')

#     writer = Chem.SDWriter(out)

#     counts = {}
#     written = []
#     for pv in original_pvs:
#         sts = Chem.ForwardSDMolSupplier(gzip.open(pv))
#         for st in sts:
#             name = st.GetProp("_Name")
#             if name not in counts:
#                 counts[name] = 0
#             else:
#                 # counts is zero indexed.
#                 counts[name] += 1

#             if name in scores.index and scores.loc[name, 'POSE'] == counts[name]:
#                 writer.append(st)
#                 written += [name]

#     assert len(written) == len(scores), written
#     for name in scores.index:
#         assert name in written

# @main.command()
# @click.argument('pv')
# @click.argument('scores')
# @click.argument('out', default=None)
# def apply_scores(pv, scores, out):
#     """
#     Add ComBind screening scores to a poseviewer.
#     """
#     from open_combind.score.screen import apply_scores
#     if out is None:
#         out = pv.replace('_pv.maegz', '_combind_pv.maegz')
#     apply_scores(pv, scores, out)

@cli.command()
@click.argument('pv')
@click.argument('out', default=None)
def scores_to_csv(pv, out):
    """
    Write docking and ComBind scores to text.
    """
    oc.scores_to_csv(pv, out)

@cli.command()
@click.argument('smiles')
@click.option('--ligand-names', default='ID')
@click.option('--ligand-smiles', default='SMILES')
@click.option('--features', default='mcss,hbond,saltbridge,contact', help='which features to use for pose prediction')
@click.option('--processes', default=-1, help='number of processes to use, -1 means use all available processes')
def prep_dock_and_predict(smiles,features,ligand_names,ligand_smiles,processes):
    """
    Run the whole Open-Combind pipeline, from structure preparation all the way to pose prediction

    Raw complexes are assumed to be in 'structures/raw/*.pdb' with associated info files in 'structures/raw/*.info'

    SMILES is a comma delimited file containing the name and smiles strings of all of your docking ligands
    """
    from rdkit.Chem import ForwardSDMolSupplier
    from prody import confProDy
    confProDy(verbosity="critical")
    oc.structprep(None)
    oc.ligprep(smiles, root='ligands', multiplex=False, ligand_names=ligand_names,
            ligand_smiles=ligand_smiles, delim=',', sdffile=False,
            num_confs=10, confgen='etkdg_v2', max_iterations=500, processes=processes)
    for lig_file in sorted(glob('structures/ligands/*.sdf'))[1:]:
        in_lig = next(ForwardSDMolSupplier(lig_file))
        out_ligname = f"ligands/{lig_file.split('/')[-1]}"
        oc.dock.ligprep.write3DConf(in_lig, out_ligname, num_confs=100)
    oc.dock_ligands(glob('ligands/*.sdf'), template=None, dock_file=None, root='docked', screen=False, slurm=False, now=True, processes=processes)
    no_mcss = not ('mcss' in features)
    use_shape = 'shape' in features
    oc.featurize('features', glob('docked/*.sdf.gz'), no_mcss=no_mcss, use_shape=use_shape,
                max_poses=100, processes=processes)
    features = features.split(',')
    oc.pose_prediction('features', 'poses.csv', None, features=features)

