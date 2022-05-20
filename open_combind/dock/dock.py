import os
import subprocess
from glob import glob

GNINA = ' -l {lig} -o {out} --exhaustiveness {exh} --num_modes 200 > {log} \n'

def docking_failed(gnina_log):
    if not os.path.exists(gnina_log):
        return False
    with open(gnina_log) as fp:
        logtxt = fp.read()
    # phrases = ['** NO ACCEPTABLE LIGAND POSES WERE FOUND **',
    #            'NO VALID POSES AFTER MINIMIZATION: SKIPPING.',
    #            'No Ligand Poses were written to external file',
    #            'GLIDE WARNING: Skipping refinement, etc. because rough-score step failed.']
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

def dock(template, ligands, root, name, enhanced, infile=None, reference=None, slurm=False):
    outfile = "{inlig}-docked.sdf.gz"
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

    if slurm:
        receptor = dock_template.split('-r')[-1].strip().split(' ')[0]
        abox = dock_template.split('--autobox_ligand')[-1].strip().split(' ')[0]
        setup_slurm(gnina_in,ligands,receptor,abox)


def setup_slurm(gnina_in,ligands,receptor,abox):
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


# def filter_native(native, pv, out, thresh):
#     with StructureReader(native) as sts:
#         native = list(sts)
#         assert len(native) == 1, len(native)
#         native = native[0]

#     near_native = []
#     with StructureReader(pv) as reader:
#         receptor = next(reader)
#         for st in reader:
#             conf_rmsd = ConformerRmsd(native, st)
#             if conf_rmsd.calculate() < thresh:
#                 near_native += [st]

#     print('Found {} near-native poses'.format(len(near_native)))
#     if not near_native:
#         print('Resorting to native pose.')
#         native.property['r_i_docking_score'] = -10.0
#         near_native = [native]

#     with StructureWriter(out) as writer:
#         writer.append(receptor)
#         for st in near_native:
#             writer.append(st)
