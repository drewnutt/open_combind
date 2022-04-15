# from schrodinger.structure import StructureReader, StructureWriter
import os
import gzip
import subprocess
from plumbum.cmd import obabel
from rdkit.Chem import AllChem as Chem

def ligprocess(input_file, output_file):
    _,_,out = obabel[input_file,'-O',output_file, '--gen3d'].run()
    if 'Error  in Do\n  3D coordinate generation failed' in out:
        mol = Chem.MolFromSmiles(open(input_file).readlines()[0].strip())
        mol.SetProp('_Name',os.path.basename(input_file).replace('.smi',''))
        Chem.EmbedMolecule(mol)
        assert mol.GetConformer().Is3D(), f"can't make {input_file} into 3d"
        writer = Chem.SDWriter(output_file)
        writer.write(mol)
        writer.close()

def ligprep(smiles):
    sdf_file = smiles.replace('.smi', '.sdf')
    ligprocess(smiles, sdf_file)

def ligsplit(big_sdf, root, name_prop='BindingDB MonomerID'):
    if os.path.splitext(big_sdf)[-1] == ".gz":
        big_sdf_data = gzip.open(big_sdf)
    else:
        big_sdf_data = open(big_sdf,'rb')
    sts = Chem.ForwardSDMolSupplier(big_sdf_data)
    for count, st in enumerate(sts):
        het_id = st.GetProp('Ligand HET ID in PDB')
        name = None
        if len(het_id): # Need to check if a crystal ligand
            pdbs = sorted(st.GetProp('PDB ID(s) for Ligand-Target Complex').strip().split(','))
            for i, pdb in enumerate(pdbs):
                if os.path.isfile(f"structures/ligands/{pdb}_lig.sdf"):
                    name = pdb
                    break
        if name is None:
            name = st.GetProp('BindingDB MonomerID')

        Chem.EmbedMolecule(st)
        assert st.GetConformer().Is3D(), f"can't make {name} into 3d"

        writer = Chem.SDWriter(f"{root}/{name}.sdf")
        writer.write(st)
        writer.close()

    print(f"Created {count+1} ligands in {root}")

if __name__ == '__main__':
    import sys
    input_file, output_file = sys.argv[1:]
    ligprocess(input_file, output_file)
