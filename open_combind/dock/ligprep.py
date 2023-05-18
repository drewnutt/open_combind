import os
import argparse
import gzip
from rdkit.Chem import AllChem as Chem

def construct_set_conformers(mol, *, num_confs, confgen, seed=-1):
    """
    Construct a set of 3D conformers for a molecule.

    Parameters
    ----------
    mol : :class:`~rdkit.Chem.rdchem.Mol`
        The molecule to construct conformers for.
    num_confs : int
        The number of conformers to generate.
    confgen : str
        The conformer generation algorithm to use. One of ``'etkdg_v1'``, ``'etkdg_v2'``, or ``'etkdg_v3'``.
    seed : int, optional
        The random seed to use for conformer generation. If not specified, a random seed will be used.

    Returns
    -------
    `list[int]<list>`
        The conformer IDs of the generated conformers.
    """

    if confgen == 'etkdg_v1':
        ps = Chem.ETKDG()
    elif confgen == 'etkdg_v2':
        ps = Chem.ETKDGv2()
    elif confgen == 'etkdg_v3':
        ps = Chem.ETKDGv3()
        ps.useSmallRingTorsions = True
    else:
        print('confgen not `etkdg_v[1-3]` so using ETDG')
        ps = Chem.ETDG()
    ps.randomSeed = seed
    cids = Chem.EmbedMultipleConfs(mol, num_confs, ps)
    if len(cids) == 0:
        ps.useRandomCoords = True
        cids = Chem.EmbedMultipleConfs(mol, num_confs, ps)
    if len(cids) == 0:
        ps = Chem.ETDG()
        ps.useRandomCoords = True
        cids = Chem.EmbedMultipleConfs(mol, num_confs, ps)
    return cids

def make3DConf(inmol, confgen='etkdg_v2', ff='UFF', num_confs=50, maxIters=1000, seed=-1):
    """
    Generate a 3D conformer for a molecule.

    Parameters
    ----------
    inmol : :class:`~rdkit.Chem.rdchem.Mol`
        The molecule to generate a conformer for.
    confgen : str
        The conformer generation algorithm to use. One of ``'etkdg_v1'``, ``'etkdg_v2'``, or ``'etkdg_v3'``.
    ff : str
        The force field to use for energy minimization. One of ``'UFF'`` or ``'MMFF'``.
    num_confs : int
        The number of conformers to generate.
    maxIters : int
        The maximum number of iterations for energy minimization.
    seed : int, optional
        The random seed to use for conformer generation. If not specified, a random seed will be used.
    
    Returns
    -------
    :class:`~rdkit.Chem.rdchem.Mol`
        The molecule with all of the generated conformers.
    int
        The conformer ID of the lowest enegy generated conformer.
    `list[float]<list>`
        The energies of the generated conformers.
    """

    mol = Chem.Mol(inmol)
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    if num_confs > 0:
        cids = construct_set_conformers(mol, num_confs=num_confs, confgen=confgen, seed=seed)
    else:
        cids = [-1]
    assert len(cids) > 0
    cenergy = []
    for conf in cids:
        if ff == 'UFF':
            converged = not Chem.UFFOptimizeMolecule(mol, confId=conf, maxIters=maxIters)
            cenergy.append(Chem.UFFGetMoleculeForceField(mol, confId=conf).CalcEnergy())
        elif ff == 'MMFF':
            converged = not Chem.MMFFOptimizeMolecule(mol, confId=conf)
            mp = Chem.MMFFGetMoleculeProperties(mol)
            cenergy.append(
                    Chem.MMFFGetMoleculeForceField(mol, mp, confId=conf).CalcEnergy())
    mol = Chem.RemoveHs(mol)
    best_conf = min(cids, key=lambda cid: cenergy[cid])

    assert mol.GetConformer(best_conf).Is3D(), f"can't make {mol.GetProp('_Name')} into 3d"

    return mol, best_conf, cenergy


def write3DConf(inmol, out_fname, **kwargs):
    """
    Generate a 3D conformer for a molecule and write it to a file.

    Parameters
    ----------
    inmol : :class:`~rdkit.Chem.rdchem.Mol`
        The molecule to generate a conformer for.
    out_fname : str
        The name of the file to write the conformer to.
    kwargs
        Additional keyword arguments to pass to :func:`make3DConf`.
    """

    mol, best_conf, _ = make3DConf(inmol, **kwargs)

    writer = Chem.SDWriter(out_fname)
    writer.write(mol, best_conf)
    writer.close()

def write3DConfs(inmol, out_fname, num_out_confs=10, **kwargs):
    """
    Generate a set of 3D conformers for a molecule and write them to a file.
    
    Parameters
    ----------
    inmol : :class:`~rdkit.Chem.rdchem.Mol`
        The molecule to generate conformers for.
    out_fname : str
        The name of the file to write the conformers to.
    num_out_confs : int
        The number of conformers to write to the file.
    kwargs
        Additional keyword arguments to pass to :func:`make3DConf`.
    """

    mol, _, energies = make3DConf(inmol, **kwargs)


    sorted_confs = sorted(list(range(mol.GetNumConformers())),key=lambda x: energies[x])
    writer = Chem.SDWriter(out_fname.split('.')[0]+'.sdf')
    for conf in range(min(num_out_confs,mol.GetNumConformers())):
        writer.write(mol, sorted_confs[conf])
    writer.close()

def ligprocess(input_file, output_file, **kwargs):
    """
    Generate 3D conformers for a molecule given by `input_file` and write them to a file.

    Parameters
    ----------
    input_file : str
        The name of the file to read the molecule from. Can contain a SMILES string.
    output_file : str
        The name of the file to write the conformers to.
    kwargs
        Additional keyword arguments to pass to :func:`make3DConf`.
    """

    input_info = open(input_file).readlines()
    if len(input_info) == 1:
        mol = Chem.MolFromSmiles(input_info[0].strip())
        mol.SetProp('_Name', os.path.basename(input_file).replace('.smi', ''))

        write3DConfs(mol, output_file, **kwargs)
    else:
        ## need to make the multiplexed write out several conformations per ligand
        raise NotImplementedError
        # writer = Chem.SDWriter(output_file)
        # for line in input_info:
        #     smile, name = line.strip().split(' ')
        #     print(name)
        #     mol = Chem.MolFromSmiles(smile)
        #     mol.SetProp('_Name', name)

        #     mol, best_conf, _ = make3DConf(mol, confgen=confgen, ff=ff, num_confs=num_confs, maxIters=maxIters)
       
        #     writer.write(mol, best_conf)

        # writer.close()

def ligprep(smiles, **kwargs):
    """
    Generate 3D conformers for a molecule in `smiles` and write to a file with the same basename as `smiles`

    Parameters
    ----------
    smiles : str
        The name of the file to read the molecule from. Can contain a SMILES string.
    kwargs
        Additional keyword arguments to pass to :func:`make3DConf`.
    """

    sdf_file = smiles.replace('.smi', '.sdf')
    ligprocess(smiles, sdf_file, **kwargs)

def ligprep_mp(smiles, args):
    """
    Multiprocessing wrapper for :func:`ligprep`.

    Parameters
    ----------
    smiles : str
        The name of the file to read the molecule from. Can contain a SMILES string.
    args : :class:`argparse.Namespace`
        The command line arguments.

    See Also
    --------
    ligprep
    """

    ligprep(smiles, **args._asdict())

def ligsplit(big_sdf, root, multiplex=False, name_prop='BindingDB MonomerID',
        confgen='etkdg_v2', ff='UFF', processes=1, num_confs=50, 
        maxIters=1000, num_out_confs=10):
    """
    Split a large SDF file into individual SDF files for each ligand.

    Parameters
    ----------
    big_sdf : str
        The name of the file to read the ligands from.
    root : str
        The root directory to write the ligands to.
    multiplex : bool
        Whether to multiplex the ligands into a single file.
    name_prop : str
        The name of the property to use as the ligand name.
    confgen : str
        The conformer generation method to use.
    ff : str
        The forcefield to use.
    processes : int
        The number of processes to use.
    num_confs : int
        The number of conformers to generate.
    maxIters : int
        The maximum number of iterations to use to minimize the conformers.
    num_out_confs : int
        The number of conformers to write to the file.
    """

    if os.path.splitext(big_sdf)[-1] == ".gz":
        big_sdf_data = gzip.open(big_sdf)
    else:
        big_sdf_data = open(big_sdf, 'rb')
    ligands = Chem.ForwardSDMolSupplier(big_sdf_data)
    unfinished = []
    for count, ligand in enumerate(ligands):
        het_id = ligand.GetProp('Ligand HET ID in PDB')
        name = None
        if len(het_id):  # need to check if a crystal ligand
            pdbs = sorted(ligand.GetProp('PDB ID(s) for Ligand-Target Complex').strip().split(','))
            for i, pdb in enumerate(pdbs):
                if os.path.isfile(f"structures/ligands/{pdb}_lig.sdf"):
                    name = pdb
                    break
        if name is None:
            name = ligand.GetProp(name_prop)
        _sdf = f"{root}/{name}.sdf"
        if not os.path.exists(_sdf):
            unfinished.append((ligand, _sdf, confgen, ff, num_confs, maxIters))

    if not multiplex:
        from open_combind.utils import mp
        print(f"Creating {len(unfinished)} ligands in {root}")
        mp(write3DConfs, unfinished, processes)
    else:
        output_file = f"{big_sdf.split('.',1)[0]}-3d_coords.sdf"
        if not os.path.exists(output_file):
            print(f"Creating {output_file} with {len(unfinished)} ligands")
            writer = Chem.SDWriter(output_file)
            for lig, _, confgen, ff in unfinished:
                mol, best_conf = make3DConf(lig, confgen=confgen, ff=ff, num_confs=num_confs, maxIters=maxIters)
           
                writer.write(mol, best_conf)

            writer.close()

def check_filetype(fname):
    base_fname = os.path.basename(fname)
    ext = base_fname.split('.')[-1]
    return ext

def process_both(inname,ext,outname, **kwargs):
    if ext in ['csv','smi']:
        ligprocess(inname,outname, **kwargs)
    elif ext == 'sdf':
        mol = next(Chem.ForwardSDMolSupplier(inname))
        mol.SetProp('_Name', os.path.basename(inname).replace('.sdf', ''))
        write3DConfs(mol, outname, **kwargs)

if __name__ == '__main__':
    def argument_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_file', '-I', required=True, help='Input file')
        parser.add_argument('--output_file', '-O', required=True, help='Output file')
        parser.add_argument('--seed', '-S', default=-1, type=int, help='Random seed')
        parser.add_argument('--num_out_confs', default=10, type=int, help='Number of output conformations')
        parser.add_argument('--ff', default='UFF', help='Force field for conformer minimization')
        parser.add_argument('--confgen', default='etkdg_v2', help='Conformer generation method')
        parser.add_argument('--maxIters', default=1000, type=int, help='Maximum number of minimization iterations')
        parser.add_argument('--num_confs', default=50, type=int, help='Number of conformations to generate with the embedding method')
        return parser.parse_args()

    args = argument_parser()
    extension = check_filetype(args.input_file)
    process_both(args.input_file, extension, args.output_file, num_out_confs=args.num_out_confs,
            seed=args.seed, ff=args.ff, confgen=args.confgen, maxIters=args.maxIters,
            num_confs=args.num_confs)
