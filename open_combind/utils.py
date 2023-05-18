from multiprocessing import Pool
import os
import numpy as np
from rdkit import Chem

def np_load(fname, halt=True, delete=False):
    """
    Load a numpy file, handling corrupt files.

    Parameters
    ----------
    fname : str
        Path to file to load.
    halt : bool, default=True
        Whether to halt execution if the file is corrupt.
    delete : bool, default=False
        Whether to delete the file if it is corrupt.

    Returns
    -------
    data : :class:`~numpy.ndarray`
        Data in the file.
    """
    fname = os.path.abspath(fname)
    try:
        return np.load(fname)
    except ValueError as e:
        m = 'Cannot load file containing pickled data when allow_pickle=False'
        if m in str(e):
            print('{} is corrupt. Regenerate and try again.'.format(fname))
            if delete:
                os.remove(fname)
        else:
            print("Can't open {}".format(fname))
            print(str(e))

        if halt:
            exit()

def pv_path(root, name):
    """
    Get the path to a pose viewer file.

    Parameters
    ----------
    root : str
        Root directory.
    name : str
        Name of the protein.
    
    Returns
    -------
    path : str
        Path to the pose viewer file.
    """

    if '_native' in name:
        name = name.replace('_native', '')
        return '{}/{}/{}_native_pv.maegz'.format(root, name, name)
    return '{}/{}/{}_pv.maegz'.format(root, name, name)

def get_pose(pv, pose):
    """
    Get a pose from a pose viewer file.

    Parameters
    ----------
    pv : str
        Path to the pose viewer file.
    pose : int
        Pose number.
    
    Returns
    -------
    pose : :class:`~rdkit.Chem.rdchem.Mol`
        The selected pose.
    """
    if os.path.splitext(pv)[-1] == ".gz":
        import gzip
        pv = gzip.open(pv)
    else:
        pv = open(pv)
    sts = Chem.ForwardSDMolSupplier(pv)
    for i,st in enumerate(sts):
        if i == pose:
            break
    return st

def basename(path):
    """
    Get the basename of a file without the extension.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    x : str
        Basename of the file.
    """

    x = os.path.basename(path)
    x = os.path.splitext(x)[0]
    return x

def mp(function, unfinished, processes, maxtasksperchild=None):
    """
    Run a function in parallel.

    Parameters
    ----------
    function : function
        Function to run.
    unfinished : list
        List of arguments to pass to the function. Usually a list of iterables.
    processes : int
        Number of processes to use. If -1, will use all available cpus.
    maxtasksperchild : int, default=None
        Number of tasks per child process.
    
    Returns
    -------
    x : list
        List of results from the function.
    """
    if processes == -1:  # Will use all available cpus if processes is -1
        processes = None
    if unfinished:
        with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
            x = pool.starmap(function, unfinished)
        return x

def mkdir(path):
    """
    Make a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        Path to the directory.
    """

    if not os.path.exists(path):
        os.mkdir(path)

def count_poses(pv):
    """
    Count the number of molecules in a Mol/SDF file.

    Parameters
    ----------
    pv : str
        Path to the pose viewer file.

    Returns
    -------
    num_poses : int
        Number of poses in the file.
    """
    if os.path.splitext(pv)[-1] == ".gz":
        import gzip
        pv = gzip.open(pv)
    else:
        pv = open(pv)
    num_poses = [1 for i in Chem.ForwardSDMolSupplier(pv)]
    return sum(num_poses)
