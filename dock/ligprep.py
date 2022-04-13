# from schrodinger.structure import StructureReader, StructureWriter
import os
import subprocess
from plumbum.cmd import obabel

def ligprocess(input_file, output_file):
    obabel[input_file,'-O',output_file]()
            # Remove explicit stereochemistry specifications. These cause
            # errors in downstream steps.
            # for k in st.property.keys():
            #     if 's_st_EZ_' in k or 'Chiral' in k:
            #         st.property.pop(k)

            # Give each atom a unique name, ligands generated from smiles
            # strings will not have any atom name by default.
            # names = set()
            # counts = {}
            # for atom in st.atom:
            #     if not atom.pdbname.strip():
            #         if atom.element not in counts: counts[atom.element] = 0
            #         counts[atom.element] += 1
            #         atom.pdbname = atom.element + str(counts[atom.element])
                    
            #         assert atom.pdbname not in names, atom.pdbname
            #         names.add(atom.pdbname)
            # writer.append(st)

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
