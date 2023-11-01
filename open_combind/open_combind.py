#!/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
import click
import importlib.resources
from glob import glob
from collections import namedtuple

from open_combind.utils import *
from prody import confProDy, LOGGER

###############################################################################

# Defaults
MCSS_PARAM = 'strict'
IFP_VERSION = 'rd1'

# Ligprep arguments for multiprocessing
LigprepArgs = namedtuple('LigprepArgs', ['num_out_confs', 'num_confs', 'confgen', 'maxIters', 'ff', 'seed'])

def main():
    pass

def structprep(templ_struct='', struct='', raw_dir='structures/raw',
        align_dir='structures/aligned', processed_dir='structures/processed',
        template_dir='structures/dir', ligand_dir='structures/ligands',
        protein_dir='structures/proteins' ):
    """
    Prepare structures and make a docking template file.

    Parameters
    ----------
    templ_struct : str, optional
        PDB ID to use as the template for docking. (defaults to PDB ID with alphabetically lowest name or `struct` if set)
    struct : str, optional
        PDB ID to use as the template for alignment, and docking if `templ_struct` not set. (Defaults to the structure with alphabetically lowest name)
    raw_dir : str, optional, default='structures/raw'
        Directory containing raw structures
    align_dir : str, optional, default='structures/aligned'
        Directory containing aligned structures
    processed_dir : str, optional, default='structures/processed'
        Directory containing processed structures
    template_dir : str, optional, default='structures/template'
        Directory containing docking templates
    ligand_dir : str, optional, default='structures/ligands'
        Directory containing ligands
    protein_dir : str, optional, default='structures/proteins'
        Directory containing proteins

    Notes
    -----
    The following directory structure is recomended::

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

    Files ending with `_lig` contain only the small molecule ligand present in the
    structure, and files ending with `_prot` contain everything else.
    """
    confProDy(verbosity='none')
    from open_combind.dock.struct_align import struct_align
    from open_combind.dock.struct_sort import struct_sort
    from open_combind.dock.struct_process import struct_process
    from open_combind.dock.grid import make_grid

    #iterate over the directories and check that the directories exist and make them if not
    for directory in [align_dir, processed_dir, template_dir, ligand_dir, protein_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    structs = sorted(glob(f'{raw_dir}/*.pdb*'))
    assert len(structs) > 0, f'No structures found in {raw_dir}'
    structs = [struct.split('/')[-1].split('.pdb')[0] for struct in structs]
    
    if not struct:
        struct = structs[0]

    if not templ_struct:
        templ_struct = struct

    print(f'Processing {structs}, aligning to {struct}, and creating a docking'
          f' templ for {templ_struct}')

    struct_process(structs, raw_dir=raw_dir, processed_dir=processed_dir)
    struct_align(struct, structs, align_dir=align_dir,
            process_dir=processed_dir, raw_dir=raw_dir)
    struct_sort(structs, align_dir=align_dir, raw_dir=raw_dir,
            protein_dir=protein_dir, ligand_dir=ligand_dir)
    make_grid(templ_struct, protein_dir=protein_dir, ligand_dir=ligand_dir, template_dir=template_dir)

# Not super sure what is absolutely necessary in this step, especially if starting from sdf
# instead of starting from smiles
def ligprep(smiles, root='ligands', multiplex=False, ligand_names="ID",
        ligand_smiles="SMILES", delim=",", sdffile=False, num_out_confs=10,
        num_confs=50, confgen='etkdg_v2', ff="UFF", max_iterations=1000,
        seed=-1, processes=1):
    """
    Prepare ligands for docking, from smiles or sdf.

    Specifically, this will utilize RDKit to generate a 3D conformer for each of the ligands in the provided CSV or SDF

    Parameters
    ----------
    smiles : str
        A `delim` delimited file with columns `ligand-names` and `ligand-smiles`. `smiles` can also be a SDF file with multiple
    ligands as different entries, but you must specify `sdffile`.
    root : str, default="ligands"
        Specifies where the processed ligands will be written. By default, an individual file will be made for each ligand. If `multiplex` is set, then only one file, containing all the ligands, will be produced.
    multiplex : bool, default=False
        Write all ligands into a single sdf file
    ligand_names: str, default="ID"
        Column ID of the CSV column containing the ligand names
    ligand_smiles: str, default="SMILES"
        Column ID of the CSV column containing the ligand smiles
    delim: str, default=","
        CSV Delimiter
    sdffile: bool, default=False
        `smiles` is a SDF file containing all of the ligands
    num_out_confs: int, default=10
        Number of conformations to output for each ligand
    num_confs: int, default=50
        Number of conformations for RDKit to generate initially with rdkit.EmbedMultipleConfs for later minimization
    confgen: str, default="etkdg_v2"
        Conformation generation embedded parameters model to use (see: :class:`~rdkit.Chem.rdDistGeom`)
    ff: str, default="UFF"
        Force field to use for minimization (see: :class:`~rdkit.Chem.rdForceFieldHelpers`)
    max_iterations: int, default=1000
        Number of iterations to minimize the generated conformations with the UFF force field
    seed: int, default=-1
        Random seed to use for RDKit conformer generation (`-1` implies no random seed)
    processes: int, default=1
        Number of processes to use, -1 implies all available cores. Cannot use with `multiplex`.
    """

    from open_combind.dock.ligprep import ligprep, ligprep_mp, ligsplit
    mkdir(root)
    if not sdffile:
        ligands = pd.read_csv(smiles, sep=delim)
        print('Prepping {} mols from {} in {}'.format(len(ligands), smiles, root))
        if multiplex:
            _name = os.path.splitext(os.path.basename(smiles))[0]
            _smiles = f'{root}/{_name}.smi'
            _sdf = os.path.splitext(_smiles)[0] + '.sdf'

            if not os.path.exists(_sdf):
                # mkdir(_root)
                with open(_smiles, 'w') as fp:
                    for _, ligand in ligands.iterrows():
                        fp.write('{} {}\n'.format(ligand[ligand_smiles], ligand[ligand_names]))
                ligprep(_smiles, num_out_confs=num_out_confs, num_confs=num_confs, confgen=confgen, maxIters=max_iterations, ff=ff, seed=seed)
        else:
            unfinished = []
            for _, ligand in ligands.iterrows():
                _name = ligand[ligand_names]
                _smiles = f'{root}/{_name}.smi'
                _sdf = os.path.splitext(_smiles)[0] + '.sdf'

                if not os.path.exists(_sdf):
                    with open(_smiles, 'w') as fp:
                        fp.write('{} {}\n'.format(ligand[ligand_smiles], ligand[ligand_names]))
                    ligprep_args = LigprepArgs(num_out_confs=num_out_confs, num_confs=num_confs, confgen=confgen, maxIters=max_iterations, ff=ff, seed=seed)
                    unfinished += [(_smiles, ligprep_args)]
            mp(ligprep_mp, unfinished, processes)
    else:
        raise NotImplementedError
        # ligsplit(smiles, root, multiplex=multiplex, processes=processes,
        #         num_confs=num_confs, confgen=confgen, maxIters=max_iterations)

def dock_ligands(ligands, template=None, dock_file=None, root='docking', screen=False, slurm=False, now=False, processes=1):
    """
    Generate GNINA docking commands to dock `ligands` to `template`.

    Parameters
    ----------
    ligands : list of str
        Paths to prepared ligand files. Multiple can be specified.
    template : str
        Path to template file to use for docking (Defaults to alphabetically first `.template` file in `structures/template/`)
    dock_file : str
        Format string that will be used with all of the ligands to create a docking file. Defaults to::
         -l {lig} -o {out} --exhaustiveness {exh} --num_modes 30 --min_rmsd_filter 0.01 > {log}
    root : str, default="docking"
        specifies where the docking results will be written.
    screen : bool, default=False
        Limits the thoroughness of the pose sampling (exhaustiveness=8). Recommended for screening, but not pose prediction.
    slurm : bool, default=False
        Create a tarball including all files needed for docking and update docking command to use tarball paths
    now : bool, default=False
        After generating the docking commands, run the docking
    """

    from open_combind.dock.dock import dock

    if template is None:
        template = sorted(glob('structures/template/*.template'))
        if template:
            template = template[0]
        else:
            # print(''
            #       '')
            raise ValueError("No templates in default location (structures/template)",", please specify path.")
            # sys.exit()


    ligands = [os.path.abspath(lig) for lig in ligands if 'nonames' not in lig]
    template = os.path.abspath(template)
    root = os.path.abspath(root)

    mkdir(root)
    ligs = []
    names = []
    for ligand in ligands:
        name = '{}-to-{}'.format(basename(ligand), basename(template))
        ligs.append(ligand)
        names.append(name)
    print(f"Writing docking file for {len(ligs)} ligands")
    dock(template, ligands, root, names, not screen, slurm=slurm, now=now, infile=dock_file, processes=processes)

################################################################################

def featurize(root, poseviewers, native='structures/ligands/*_lig.sdf',
            no_mcss=False, use_shape=False, max_poses=100, no_cnn=False,
            ifp_version=IFP_VERSION, mcss_param=MCSS_PARAM, processes=1,
            check_center_ligs=False, template='structures/template/*.template',
            newscore=None, reverse=True):
    """
    Featurize the set of docked ligand poses, `poseviewers`

    Parameters
    ----------
    root : str
        Path of where to place the computed features
    poseviewers : list of str
        paths to all of the docked ligand poses that need featurization
    native : str, default='structures/ligands/*_lig.sdf'
        Glob-able path of the ground-truth ligand poses (if available)
    no_mcss : bool, default=False
        Do not compute the RMSD of the maximum common substructure
    use_shape : bool, default=False
        Compute the shape similarity of ligand poses
    max_poses : int, default=100
        Maximum number of poses to featurize per ligand
    no_cnn : bool, default=False
        Do not use CNN scores for featurization
    screen : bool, default=False
    ifp_version : str
        Interaction fingerprint version
    shape_version : str
        Version of the shape similarity calculator
    processes : int, default=1
        Number of processes to use, -1 implies all cores.
    newscore : str, default=None
        Name of the new score to use for featurization
    reverse : bool, default=True
        Reverse the order of the poses before featurization, highest score first
    """

    from open_combind.dock.postprocessing import coalesce_poses, write_poses
    from open_combind.features.features import Features
    if use_shape:
        print("Shape is not currently implemented outside of Schrodinger\n Shape has not been evaluated for performance in pose-prediction")
    if newscore is not None:
        no_cnn = True
        print("Using new score for featurization and not CNN scores")
    
    for poseviewer in poseviewers:
        sorted_poses = coalesce_poses(poseviewer, sort_by=newscore if newscore is not None else 'CNNscore',reverse=reverse)
        write_poses(sorted_poses, poseviewer)

    native_poses = {}
    for native_path in glob(native):
        name = native_path.split('/')[-1].replace('.sdf', '')
        # sts = Chem.SDMolSupplier(native_path)
        native_poses[name] = native_path
    print(native_poses)

    template_file = sorted(glob(template))[0]
    features = Features(root, ifp_version=ifp_version, mcss_param=mcss_param,max_poses=max_poses, 
                        cnn_scores=not no_cnn, template=template_file,
                        check_center_ligs=check_center_ligs, newscore=newscore)

    print(poseviewers)
    features.compute_single_features(poseviewers, native_poses=native_poses, processes=processes)

    features.compute_pair_features(poseviewers,
                                   mcss=not no_mcss, shape=use_shape, processes=processes)

################################################################################

def pose_prediction(root, out="poses.csv", ligands=None, features=['mcss', 'hbond', 'saltbridge', 'contact'],
                    alpha=-1, stats_root=None, restart=500, max_iterations=1000, newscore=None):
    """
    Run ComBind pose prediction and generate a CSV, `out` with the selected pose numbers.

    Parameters
    ----------
    root : str
        Path to the ligand pose features
    out : str, default="poses.csv"
        Name of the output CSV containing the ComBind pose predictions
    ligands : list of str
        ligands to predict poses for. Defaults to all ligands that have been featurized.
    features : list of str, default=['mcss', 'hbond', 'saltbridge', 'contact']
        Which features to utilize during the pose prediction process
    alpha : float, default=1
        Hyperparameter to multiply the per pose score
    stats_root : str
        Path to root of the statistics directory containing the pairwise statistics used for scoring pairs of poses
    restart : int, default=500
        Number of times to run the optimization process
    max_iterations : int, default=1000
        Maximum number of iterations to attempt before exiting each optimization process
    """
    from open_combind.score.pose_prediction import PosePrediction
    from open_combind.score.statistics import read_stats
    from open_combind.features.features import Features
    from importlib_resources import files


    cnn_scores = True
    if newscore is not None:
        cnn_scores = False
    protein = Features(root, newscore=newscore, cnn_scores=cnn_scores)
    protein.load_features()

    if not ligands:
        ligands = set(protein.raw['name1'])
    ligands = sorted(ligands)

    data = protein.get_view(ligands, features)
    if stats_root is None:
        stats_root = files("open_combind").joinpath("stats_data/default/")
    stats = read_stats(stats_root, features)
    
    ps = PosePrediction(ligands, features, data, stats, alpha, newscore=newscore)
    best_poses, best_score = ps.max_posterior(max_iterations, restart)

    with open(out, 'w') as fp:
        fp.write('ID,POSE,COMBIND_RMSD,GNINA_RMSD,BEST_RMSD\n')
        for ligand in best_poses:
            rmsds = data['rmsd'][ligand]
            grmsd = rmsds[0]
            crmsd = rmsds[best_poses[ligand]]
            brmsd = min(rmsds)
            fp.write(','.join(map(str, [ligand.replace('_pv', ''),
                                        best_poses[ligand],
                                        crmsd, grmsd, brmsd])) + '\n')

#def screen(score_fname, root, stats_root, alpha, features):
#    """
#    Run ComBind screening.
#    """
#    from open_combind.score.screen import screen, load_features_screen
#    from open_combind.score.statistics import read_stats

#    features = features.split(',')
#    stats = read_stats(stats_root, features)
#    single, raw = load_features_screen(features, root)

#    combind_energy = screen(single, raw, stats, alpha)
#    np.save(score_fname, combind_energy)

#################################################################################

#def extract_top_poses(scores, original_pvs):
#    """
#    Write top-scoring poses to a single file.
#    """
#    from rdkit import Chem
#    import gzip

#    out = scores.replace('.csv', '.sdf.gz')
#    scores = pd.read_csv(scores).set_index('ID')

#    writer = Chem.SDWriter(out)

#    counts = {}
#    written = []
#    for pv in original_pvs:
#        sts = Chem.ForwardSDMolSupplier(gzip.open(pv))
#        for st in sts:
#            name = st.GetProp("_Name")
#            if name not in counts:
#                counts[name] = 0
#            else:
#                # counts is zero indexed.
#                counts[name] += 1

#            if name in scores.index and scores.loc[name, 'POSE'] == counts[name]:
#                writer.append(st)
#                written += [name]

#    assert len(written) == len(scores), written
#    for name in scores.index:
#        assert name in written

#def apply_scores(pv, scores, out):
#    """
#    Add ComBind screening scores to a poseviewer.
#    """
#    from open_combind.score.screen import apply_scores
#    if out is None:
#        out = pv.replace('_pv.maegz', '_combind_pv.maegz')
#    apply_scores(pv, scores, out)

#def scores_to_csv(pv, out):
#    """
#    Write docking and ComBind scores to text.
#    """
#    from open_combind.score.screen import scores_to_csv
#    scores_to_csv(pv, out)

if __name__ == "__main__":
    main()
