# from schrodinger.structure import StructureReader, StructureWriter
import os
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
    # cmd = 'ligprep -WAIT -epik -ismi {} -omae {}'.format(
    #     os.path.basename(smiles), os.path.basename(mae_noname_file))

    # subprocess.run(cmd, shell=True, cwd=os.path.dirname(smiles))
    # if not os.path.exists(mae_noname_file):
    #     print('ligprep failed on {}.'.format(smiles))
    #     print(cmd)
    #     return
    ligprocess(smiles, sdf_file)

if __name__ == '__main__':
    import sys
    input_file, output_file = sys.argv[1:]
    ligprocess(input_file, output_file)
