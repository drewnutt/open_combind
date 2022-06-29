import os
import gzip
from rdkit.Chem import AllChem as Chem

def construct_set_conformers(mol, num_confs, confgen):
    if confgen == 'etkdg_v1':
        ps = Chem.ETKDG()
    elif confgen == 'etkdg_v2':
        ps = Chem.ETKDGv2()
    else:
        print('confgen not etkdg v1 or 2 so using etdg')
        ps = Chem.ETDG()
    cids = Chem.EmbedMultipleConfs(mol, num_confs, ps )
    if len(cids) == 0:
        ps.useRandomCoords = True
        cids = Chem.EmbedMultipleConfs(mol, num_confs, ps )
    return cids

def make3DConf(inmol, confgen='etkdg_v2', ff='UFF', num_confs=10, maxIters=500):
    mol = Chem.Mol(inmol)
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    if num_confs > 0:
        cids = construct_set_conformers(mol, num_confs, confgen)
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

    return mol, best_conf, cenergy[best_conf]


def write3DConf(inmol, out_fname, confgen='etkdg_v2', ff='UFF', num_confs=10, maxIters=500):
    mol, best_conf, _ = make3DConf(inmol, num_confs=num_confs,
                                confgen=confgen, ff=ff, maxIters=maxIters)

    writer = Chem.SDWriter(out_fname)
    writer.write(mol, best_conf)
    writer.close()

def ligprocess(input_file, output_file, confgen='etkdg_v2', ff='UFF', num_confs=10,maxIters=500):
    input_info = open(input_file).readlines()
    if len(input_info) == 1:
        mol = Chem.MolFromSmiles(input_info[0].strip())
        mol.SetProp('_Name', os.path.basename(input_file).replace('.smi', ''))

        write3DConf(mol, output_file, confgen=confgen, ff=ff, num_confs=num_confs, maxIters=maxIters)
    else:
        writer = Chem.SDWriter(output_file)
        for line in input_info:
            smile, name = line.strip().split(' ')
            print(name)
            mol = Chem.MolFromSmiles(smile)
            mol.SetProp('_Name', name)

            mol, best_conf = make3DConf(mol, confgen=confgen, ff=ff, num_confs=num_confs, maxIters=maxIters)
       
            writer.write(mol, best_conf)

        writer.close()

def ligprep(smiles,num_confs=10,confgen='etkdg_v2'):
    sdf_file = smiles.replace('.smi', '.sdf')
    ligprocess(smiles, sdf_file, num_confs=num_confs, confgen=confgen)

def ligsplit(big_sdf, root, multiplex=False, name_prop='BindingDB MonomerID',
        confgen='etkdg_v2', ff='UFF', processes=1,num_confs=10,maxIters=500):
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
        from utils import mp
        print(f"Creating {len(unfinished)} ligands in {root}")
        mp(write3DConf, unfinished, processes)
    else:
        output_file = f"{big_sdf.split('.',1)[0]}-3d_coords.sdf"
        if not os.path.exists(output_file):
            print(f"Creating {output_file} with {len(unfinished)} ligands")
            writer = Chem.SDWriter(output_file)
            for lig, _, confgen, ff in unfinished:
                mol, best_conf = make3DConf(lig, confgen=confgen, ff=ff, num_confs=num_confs, maxIters=maxIters)
           
                writer.write(mol, best_conf)

            writer.close()
        # write3DConf(ligand, f"{root}/{name}.sdf", confgen=confgen, ff=ff)

def check_filetype(fname):
    base_fname = os.path.basename(fname)
    ext = base_fname.split('.')[-1]
    return ext

def process_both(inname,ext,outname):
    if ext in ['csv','smi']:
        ligprocess(inname,outname)
    elif ext == 'sdf':
        mol = next(Chem.ForwardSDMolSupplier(inname))
        mol.SetProp('_Name', os.path.basename(input_file).replace('.sdf', ''))
        write3DConf(mol, outname)

if __name__ == '__main__':
    import sys
    input_file, output_file = sys.argv[1:]
    extension = check_filetype(input_file)
    process_both(input_file, extension, output_file)
