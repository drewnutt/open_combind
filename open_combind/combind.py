#!/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
import click
import importlib.resources
from glob import glob

from open_combind.utils import *

###############################################################################

# Defaults
STATS_ROOT = 'stats_data/default'
SHAPE_VERSION = 'pharm_max'
IFP_VERSION = 'rd1'

def main():
    pass

def structprep(templ_struct, struct=''):
    """
    Prepare structures and make a docking template file.

    "struct" specifies the name of the structure for which to make a docking
    template file. (Not the full path, generally just the PDB code.) Defaults to the
    structure with alphabetically lowest name.

    The following directory structure is required:

    \b
    structures/
        raw/
            structure_name.pdb
            structure_name.info (first line is Resname of ligand)
            ...
        processed/
            structure_name/structure_name_prot.pdb
            ...
        aligned/
            structure_name/structure_name_aligned.pdb
            ...
        proteins/
            structure_name_prot.pdb
            ...
        ligands/
            structure_name_lig.pdb
            ...
        template/
            structure_name/structure_name
            ...

    The process can be started from any step, e.g. if you have processed
    versions of your structures, you can place these in the processed directory.

    Files ending with _lig contain only the small molecule ligand present in the
    structure, and files ending with _prot contain everything else.
    """
    from open_combind.dock.struct_align import struct_align
    from open_combind.dock.struct_sort import struct_sort
    from open_combind.dock.struct_process import struct_process
    from open_combind.dock.grid import make_grid

    assert os.path.exists('structures'), 'No structures directory.'

    structs = sorted(glob('structures/raw/*.pdb*'))
    structs = [struct.split('/')[-1].split('.pdb')[0] for struct in structs]
    
    if not struct:
        struct = structs[0]

    if not templ_struct:
        templ_struct = struct

    print(f'Processing {structs}, aligning to {struct}, and creating a docking'
          f' templ for {templ_struct}')

    struct_process(structs)
    struct_align(struct, structs)
    struct_sort(structs)
    make_grid(templ_struct)

# Not super sure what is absolutely necessary in this step, especially if starting from sdf
# instead of starting from smiles
def ligprep(smiles, root='ligands', multiplex=False, ligand_names="ID",
        ligand_smiles="SMILES", delim=",", sdffile=False,
        num_confs=10, confgen='etkdg_v2', max_iterations=200, processes=1):
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
    from open_combind.dock.ligprep import ligprep, ligsplit
    mkdir(root)
    if not sdffile:
        ligands = pd.read_csv(smiles, sep=delim)
        print('Prepping {} mols from {} in {}'.format(len(ligands), smiles, root))
        if multiplex:
            _name = os.path.splitext(os.path.basename(smiles))[0]
            # _root = f'{root}/{_name}'
            _smiles = f'{root}/{_name}.smi'
            _sdf = os.path.splitext(_smiles)[0] + '.sdf'

            if not os.path.exists(_sdf):
                # mkdir(_root)
                with open(_smiles, 'w') as fp:
                    for _, ligand in ligands.iterrows():
                        fp.write('{} {}\n'.format(ligand[ligand_smiles], ligand[ligand_names]))
                ligprep(_smiles, num_confs=num_confs, confgen=confgen, maxIters=max_iterations)
        else:
            unfinished = []
            for _, ligand in ligands.iterrows():
                _name = ligand[ligand_names]
                # _root = f'{root}/{_name}'
                _smiles = f'{root}/{_name}.smi'
                _sdf = os.path.splitext(_smiles)[0] + '.sdf'

                if not os.path.exists(_sdf):
                    # mkdir(_root)
                    with open(_smiles, 'w') as fp:
                        fp.write('{} {}\n'.format(ligand[ligand_smiles], ligand[ligand_names]))
                    unfinished += [(_smiles, num_confs, confgen, max_iterations)]
            mp(ligprep, unfinished, processes)
    else:
        ligsplit(smiles, root, multiplex=multiplex, processes=processes,
                num_confs=num_confs, confgen=confgen, maxIters=max_iterations)

def dock_ligands(template, ligands, dock_file, root='docking', screen=False, slurm=False, now=True):
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
    from open_combind.dock.dock import dock

    if template is None:
        template = glob('structures/template/*.template')
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
    _roots = []
    names = []
    for ligand in ligands:
        name = '{}-to-{}'.format(basename(ligand), basename(template))
        _root = '{}/{}'.format(root, name)
        ligs.append(ligand)
        _roots.append(_root)
        names.append(name)
    print(f"Writing docking file for {len(ligs)} ligands")
    dock(template, ligands, _roots, names, not screen, slurm=slurm, now=now, infile=dock_file)

################################################################################

def featurize(root, poseviewers, native='structures/ligands/*_lig.sdf',
            no_mcss=False, use_shape=False, max_poses=100, no_cnn=False,
            screen=False, ifp_version=IFP_VERSION, shape_version=SHAPE_VERSION,
            processes=1):
    from open_combind.features.features import Features
    if use_shape:
        print("Shape is not currently implemented outside of Schrodinger\n Shape has not been evaluated for performance in pose-prediction")

    native_poses = {}
    for native_path in glob(native):
        name = native_path.split('/')[-1].replace('.sdf','')
        # sts = Chem.SDMolSupplier(native_path)
        native_poses[name] = native_path
    print(native_poses)

    features = Features(root, ifp_version=ifp_version, shape_version=shape_version,
                        max_poses=max_poses, cnn_scores=not no_cnn)

    features.compute_single_features(poseviewers, native_poses=native_poses)

    if screen:
        assert len(poseviewers) == 2
        features.compute_pair_features(poseviewers[:1],
                                       pvs2 = poseviewers[1:],
                                       mcss=not no_mcss, shape=use_shape)
    else:
        features.compute_pair_features(poseviewers,
                                       mcss=not no_mcss, shape=use_shape, processes=processes)

################################################################################

def pose_prediction(root, out, ligands, features=['mcss', 'hbond', 'saltbridge', 'contact'],
                    alpha=1, stats_root=None, restart=500, max_iterations=1000):
    """
    Run ComBind pose prediction.
    """
    from open_combind.score.pose_prediction import PosePrediction
    from open_combind.score.statistics import read_stats
    from open_combind.features.features import Features
    import pkg_resources


    protein = Features(root)
    protein.load_features()

    if not ligands:
        ligands = set(protein.raw['name1'])
    ligands = sorted(ligands)

    data = protein.get_view(ligands, features)
    if stats_root is None:
        stats_root = pkg_resources.resource_filename(__name__, "stats_data/default/")
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
                                        crmsd, grmsd, brmsd])) + '\n')

def screen(score_fname, root, stats_root, alpha, features):
    """
    Run ComBind screening.
    """
    from open_combind.score.screen import screen, load_features_screen
    from open_combind.score.statistics import read_stats

    features = features.split(',')
    stats = read_stats(stats_root, features)
    single, raw = load_features_screen(features, root)

    combind_energy = screen(single, raw, stats, alpha)
    np.save(score_fname, combind_energy)

################################################################################

def extract_top_poses(scores, original_pvs):
    """
    Write top-scoring poses to a single file.
    """
    from rdkit import Chem
    import gzip

    out = scores.replace('.csv', '.sdf.gz')
    scores = pd.read_csv(scores).set_index('ID')

    writer = Chem.SDWriter(out)

    counts = {}
    written = []
    for pv in original_pvs:
        sts = Chem.ForwardSDMolSupplier(gzip.open(pv))
        for st in sts:
            name = st.GetProp("_Name")
            if name not in counts:
                counts[name] = 0
            else:
                # counts is zero indexed.
                counts[name] += 1

            if name in scores.index and scores.loc[name, 'POSE'] == counts[name]:
                writer.append(st)
                written += [name]

    assert len(written) == len(scores), written
    for name in scores.index:
        assert name in written

def apply_scores(pv, scores, out):
    """
    Add ComBind screening scores to a poseviewer.
    """
    from open_combind.score.screen import apply_scores
    if out is None:
        out = pv.replace('_pv.maegz', '_combind_pv.maegz')
    apply_scores(pv, scores, out)

def scores_to_csv(pv, out):
    """
    Write docking and ComBind scores to text.
    """
    from open_combind.score.screen import scores_to_csv
    scores_to_csv(pv, out)

if __name__ == "__main__":
    main()
