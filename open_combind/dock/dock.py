import os
import subprocess
from glob import glob

GNINA = ' -l {lig} -o {out} --exhaustiveness {exh} --num_modes 200 > {log} \n'  #: default options for GNINA

def docking_failed(gnina_log):
    if not os.path.exists(gnina_log):
        return False
    with open(gnina_log) as fp:
        logtxt = fp.read()
    # Need to compile list of Gnina failure logs
        phrases = []
    return any(phrase in logtxt for phrase in phrases)

def check_dock_line(infile):
    if not infile.endswith('\n'):
        infile += '\n'

    assert '{lig}' in infile, "need to have {lig} in your docking line to specify ligand"
    assert '{out}' in infile, "need to have {out} in your docking line to specify outfile"
    assert '{log}' in infile, "need to have {log} in your docking line to specify logfile"
    if '{exh}' not in infile:
        print('Warning: your docking line does not contain {exh}\n\
                Docking will use either your specified exhaustiveness \
                (if specified) or the default GNINA exchaustiveness of 8')

    return infile

def dock(template, ligands, root, name, enhanced, infile=None, slurm=False, now=False):
    """
    Generate GNINA docking file that utilizes the receptor and autobox as defined in `template` to dock against each of the ligands in `ligands` using the format string `infile`

    Parameters
    ----------
    template : str
        String from ``.template`` file containing the gnina docking command in the format::

            gnina -r <path_to_receptor> --autobox_ligand <path_to_autobox>
    ligands : iterable of str
        Path to each ligand that should be docked
    root : str
        Root of paths to ligands
    name : iterable of str
        Name of the output poses for each ligand docked. Should be the same size as `root`
    enhanced : bool, default=True
        Use enhanced sampling (use exhaustiveness==16)
    infile : str
        Input format string of the form::

             -l {lig} -o {out} <OTHER_GNINA_KWARGS> > {log}
    slurm : bool, default=False
        After generating docking file, create a tarball of the necessary files and update the paths to relative to the
        tarball
    now : bool, default=False
        After generating docking string, run docking immediately

    """
    outfile = "docked/{inlig}-docked.sdf.gz"
    if infile is None:
        infile = GNINA
    else:
        infile = check_dock_line(infile)
    exh = 8
    if enhanced:
        exh = 16
    dock_template = open(template).readlines()[0].strip('\n')
    recname = os.path.splitext(os.path.split(dock_template.split('-r')[-1].strip().split(' ')[0])[1])[0]
    # aboxname = os.path.splitext(os.path.split(dock_template.split('--autobox_ligand')[-1].strip().split(' ')[0])[1])[0]
    dock_line = dock_template + infile
    if slurm:
        dock_line = dock_line.replace('>', '--cpu 1 >')

    gnina_in = '{}_docking_file.txt'.format(recname)
    with open(gnina_in, 'w') as fp:
        for lig, _r, n in zip(ligands, root, name):
            out = outfile.format(inlig=n)
            gnina_log = f"{recname}_{n}.log"

            if os.path.exists(outfile):
                return

            if enhanced and docking_failed(gnina_log):
                return

            if not os.path.exists(_r):
                os.system('mkdir {}'.format(_r))
            fp.write(dock_line.format(lig=lig,out=out,exh=exh,log=gnina_log))

    if now:
        run_gnina_docking(gnina_in)
    elif slurm:
        receptor = dock_template.split('-r')[-1].strip().split(' ')[0]
        abox = dock_template.split('--autobox_ligand')[-1].strip().split(' ')[0]
        setup_slurm(gnina_in,ligands,receptor,abox)


def setup_slurm(gnina_in,ligands,receptor,abox):
    """
     Creates a tarball of the `receptor`, `abox` and all of the `ligands`. Then runs :command:`sed` on `gnina_in` to remove the path to the current working directory.

     Parameters
     ----------
     gnina_in : str
        Path to the GNINA docking commands file relative to the current working directory
     ligands : list of str
       Paths of the ligands 
     receptor : str
        Path to the receptor file
     abox : str
        Path to the autobox_ligands
     """
    import tarfile

    native_ligs = glob('structures/ligands/*.sdf')
    tarfiles = (receptor,abox,*ligands,*native_ligs)
    new_tar = gnina_in.replace('.txt','.tar.gz')
    tar = tarfile.open(new_tar, "w:gz")

    for fname in tarfiles:
        tar.add(os.path.relpath(fname))
    tar.close()

    cwd = os.getcwd() + '/'
    os.system(f'sed -i s,{cwd},,g {gnina_in}')

def run_gnina_docking(gnina_dock_file):
    """
    Iterates over the lines in `gnina_dock_file` and runs each line in serial.

    Generates a tqdm progress bar to show what percent of the lines have been run
    
    Parameters
    ----------
    gnina_dock_file : str
        Path to GNINA docking commands file
    """
    from tqdm import tqdm
    import mmap
    def get_num_lines(file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines
    print("Running GNINA docking")
    with open(gnina_dock_file) as gnina_cmds:
        for gnina_cmd in tqdm(gnina_cmds,total=get_num_lines(gnina_dock_file)):
            gnina_run, logfile = gnina_cmd.split('>')
            logfile = logfile.strip()
            with open(logfile, 'w') as log:
                subprocess.run(gnina_run.strip().split(), check=True, stderr=subprocess.STDOUT, stdout=log)

