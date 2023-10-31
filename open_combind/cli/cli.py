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
MCSS_PARAM = 'strict'
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
@click.argument('struct', default='', help='Structure to use for alignment. Defaults to alphabetically lowest.')
@click.option('--templ-struct', help='Structure to use as template for alignment. Defaults to alphabetically lowest.')
def structprep(templ_struct='', struct='', raw_dir='structures/raw',
        align_dir='structures/aligned', processed_dir='structures/processed',
        template_dir='structures/dir', ligand_dir='structures/ligands',
        protein_dir='structures/proteins' ):
    """
    Prepare structures and make a docking template file.

    raw_dir, align_dir, processed_dir, template_dir, ligand_dir, and protein_dir
    specify the directories where the raw structures, aligned structures,
    processed structures, docking templates, ligands, and proteins will be
    written, respectively. 
    Defaults to "structures/raw", "structures/aligned", "structures/processed", 
    "structures/templates", "structures/ligands", and "structures/proteins", respectively.

    The following directory structure is recommended:

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
    """
    oc.structprep(templ_struct=templ_struct, struct=struct, raw_dir=raw_dir,
            align_dir=align_dir, processed_dir=processed_dir,
            template_dir=template_dir, ligand_dir=ligand_dir,
            protein_dir=protein_dir)


@cli.command()
@click.argument('smiles', required=True, help='Smiles file to prepare.')
@click.option('--root', default='ligands',
        help='Root directory for output of ligprep', type=click.Path(), show_default=True)
@click.option('--multiplex', is_flag=True, help='Write all ligands to one file.')
@click.option('--sdffile', is_flag=True,
        help='`smiles` is a SDF file containing all the ligands')
@click.option('--ligand-names', default='ID', help='Column name for ligand names',
        show_default=True)
@click.option('--ligand-smiles', default='SMILES', help='Column name for ligand smiles',
        show_default=True)
@click.option('--delim', default=',', help='Delimiter for smiles file', show_default=True)
@click.option('--num-out-confs', default=10, help='Number of output conformers', show_default=True)
@click.option('--num-confs', default=50,
        help='Number of conformers to generate and query energy', show_default=True)
@click.option('--confgen', default='etkdg_v2', help='Conformer generation method',
        show_default=True)
@click.option('--ff', default="UFF",
        help='Force field for minimization and energy computation', show_default=True)
@click.option('--max-iterations', default=1000,
        help='Maximum number of iterations for minimization', show_default=True)
@click.option('--seed', default=-1, help='Random seed for conformer generation')
@click.option('--processes', default=1 , help='Number of processes to use', show_default=True)
def ligprep(smiles, root, multiplex, ligand_names, ligand_smiles,
            delim, processes, sdffile, num_out_confs, num_confs, confgen,
            max_iterations, seed, ff):
    """
    Prepare ligands for docking, from SMILES or SDF.

    Specifically, this will utilize RDKit to generate a 3D conformer for each of the ligands 
    """
    oc.ligprep(smiles, root=root, multiplex=multiplex, ligand_names=ligand_names,
            ligand_smiles=ligand_smiles, delim=delim, sdffile=sdffile,
            num_out_confs=num_out_confs, num_confs=num_confs, confgen=confgen,
            max_iterations=max_iterations, ff=ff, processes=processes, seed=seed)


@cli.command()
@click.argument('ligands', nargs=-1, required=True, help='Ligand files to dock.')
@click.option('--root', default='docking',
        help='Root directory for output of docking', type=click.Path(), show_default=True)
@click.option('--template',
        help='Template file for docking, defaults to the alphabetically first file in `structures/templates/*.template`',
        type=click.Path())
@click.option('--screen', is_flag=True,
        help='Screening mode, for faster docking (i.e. less sampling).')
@click.option('--slurm', is_flag=True,
        help='If running on a cluster, outputs tarball of docking inputs and makes dockfile use paths that tarball has.')
@click.option('--now', is_flag=True, help='Run docking now, instead of just writing dockfile.')
@click.option('--processes', default=1, help='Number of processes to use for docking.', show_default=True, type=int)
@click.option('--dock-file',
        help='Format string for dockfile. Defaults to "-l {lig} -o {out} --exhaustiveness {exh} --num_modes 30 --min_rmsd_filter 0.01 > {log}", should include `{lig}` and `{out}` at a minimum.')
def dock_ligands(template, root, ligands, screen, slurm, now, dock_file, processes):
    """
    Dock "ligands" using template.

    """
    oc.dock_ligands(ligands, template=template, dock_file=dock_file, root=root, screen=screen, slurm=slurm, now=now, processes=processes)

################################################################################


@cli.command()
@click.argument('root', required=True, help='Root directory to place feature files')
@click.argument('poseviewers', nargs=-1, required=True, help='Docked ligand files to featurize')
@click.option('--native', default='structures/ligands/*_lig.sdf', help='Native ligand files')
@click.option('--ifp-version', default=IFP_VERSION, help='IFP version to use')
@click.option('--mcss-param', default=MCSS_PARAM, help='MCSS parameter setup to use')
@click.option('--max-poses', default=100,
        help='Maximum number of poses to featurize per ligand', show_default=True)
@click.option('--no-mcss', is_flag=True, help='Do not use RMSD of MCSS as a feature')
@click.option('--no-cnn', is_flag=True, help='Do not extract CNN scores from poses')
@click.option('--use-shape', is_flag=True, help='Use shape comparison as a feature')
@click.option('--processes', default=1, help='Number of processes to use', show_default=True)
@click.option('--template', default='structures/template/*.template',
        help='Template files used for docking')
@click.option('--check-center-ligs', is_flag=True,
        help='Check that the center of the ligands poses are near the center of the native ligand')
@click.option('--newscore', default=None,
        help='If using a score other than Vina score, specify the name of the score here')
@click.option('--no-reverse', is_flag=False,
        help='Sort the poses from lowest (first) to highest score')
def featurize(root, poseviewers, native, ifp_version,
            mcss_param, no_mcss, use_shape, processes, max_poses,
            no_cnn, template, check_center_ligs, newscore, no_reverse):
    """
    Featurize docking poses.

    """

    oc.featurize(root, poseviewers, native=native, no_mcss=no_mcss, use_shape=use_shape,
                max_poses=max_poses, no_cnn=no_cnn, ifp_version=ifp_version,
                mcss_param=mcss_param, processes=processes, template=template,
                check_center_ligs=check_center_ligs, newscore=newscore, reverse=not no_reverse)
################################################################################


@cli.command()
@click.argument('root', required=True, help='Root directory of the feature files')
@click.argument('out', default="poses.csv", help='Output file', show_default=True)
@click.argument('ligands', nargs=-1, help='Ligands to predict poses for')
@click.option('--features', default='mcss,hbond,saltbridge,contact',
        help='Features to use for pose prediction', show_default=True)
@click.option('--alpha', default=-1.0, help='Weight of the docking score', show_default=True)
@click.option('--stats-root', default=None, help='Root directory of the statistics files, defaults to Open-ComBind default statistics')
@click.option('--restart', default=500, help='Number of likelihood minimization runs to run from random initialization', show_default=True)
@click.option('--max-iterations', default=1000, help='Maximum number of iterations for the likelihood minimization for each run', show_default=True)
@click.option('--newscore', default=None, help='If using a score other than Vina score, specify the name of the score here')
def pose_prediction(root, out, ligands, alpha, stats_root, features, restart,
                    max_iterations, newscore):
    """
    Run ComBind pose prediction.
    """
    features = features.split(',')

    oc.pose_prediction(root, out=out, ligands=ligands, features=features,
            alpha=alpha, stats_root=stats_root, restart=restart, max_iterations=max_iterations,
            newscore=newscore)

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

