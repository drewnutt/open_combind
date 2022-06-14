from open_combind import structprep
from open_combind.dock.struct_align import struct_align
from open_combind.dock.struct_sort import struct_sort
from open_combind.dock.struct_process import struct_process
from open_combind.dock.grid import make_grid
from prody import parsePDB,calcRMSD

def test_struct_process():
    input_file=["1FKN","3UDH"]
    filtered_protein="open_combind/tests/structures/processed/{pdbid}/{pdbid}_prot.pdb"
    filtered_complex="open_combind/tests/structures/processed/{pdbid}/{pdbid}_complex.pdb"
    struct_process(input_file,
            protein_in="open_combind/tests/structures/raw/{pdbid}.pdb",
            ligand_info="open_combind/tests/structures/raw/{pdbid}.info",
            filtered_protein=filtered_protein,
            filtered_complex=filtered_complex,
            filtered_hetero="open_combind/tests/structures/processed/{pdbid}/{pdbid}_het.pdb",
            filtered_water="open_combind/tests/structures/processed/{pdbid}/{pdbid}_wat.pdb")

    truth_prot="open_combind/tests/structures/processed/{pdbid}_prot_truth.pdb"
    truth_complex="open_combind/tests/structures/processed/{pdbid}_complex_truth.pdb"
    for pdbid in input_file:
        prot_truth = parsePDB(truth_prot.format(pdbid=pdbid))
        prot_test = parsePDB(filtered_protein.format(pdbid=pdbid))
        assert prot_truth.numAtoms('protein') == prot_test.numAtoms('protein')
        assert prot_test.numAtoms('hetero') == 0
        assert prot_test.numAtoms('water') == 0
        assert calcRMSD(prot_test,prot_truth) == 0

        #eval complex (more important because used downstream)
        complex_truth = parsePDB(truth_complex.format(pdbid=pdbid))
        complex_test = parsePDB(filtered_complex.format(pdbid=pdbid))
        assert complex_truth.numAtoms('protein') == complex_test.numAtoms('protein')
        assert complex_truth.numAtoms('hetero') == complex_test.numAtoms('hetero')
        assert complex_test.numAtoms('water') == 0
        assert calcRMSD(complex_test,complex_truth) == 0



# def test_struct_align():
    
