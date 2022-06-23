from open_combind import structprep
from open_combind.dock.struct_align import struct_align
from open_combind.dock.struct_sort import struct_sort
from open_combind.dock.struct_process import struct_process
from open_combind.dock.grid import make_grid
from prody import parsePDB,calcRMSD
from rdkit.Chem import ForwardSDMolSupplier
from rdkit.Chem.rdMolAlign import CalcRMS

def test_struct_process():
    input_file=["1FKN","3UDH"]
    filtered_protein="open_combind/tests/structures/processed/{pdbid}/{pdbid}_prot.pdb"
    filtered_ligand="open_combind/tests/structures/processed/{pdbid}/{pdbid}_lig.sdf"
    filtered_complex="open_combind/tests/structures/processed/{pdbid}/{pdbid}_complex.pdb"
    struct_process(input_file,
            protein_in="open_combind/tests/structures/raw/{pdbid}.pdb",
            ligand_info="open_combind/tests/structures/raw/{pdbid}.info",
            filtered_protein=filtered_protein,
            filtered_complex=filtered_complex,
            filtered_ligand=filtered_ligand,
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



def test_struct_align():
    template = "1FKN"
    input_structs=["1FKN","3UDH"]
    filtered_protein="open_combind/tests/structures/processed/{pdbid}_complex.pdb"
    aligned_prot="open_combind/tests/structures/aligned/{pdbid}/{pdbid}_aligned.pdb"
    struct_align(template,input_structs, dist=15.0, retry=True,
            filtered_protein=filtered_protein,
            aligned_prot=aligned_prot,
            align_dir="open_combind/tests/structures/aligned")

    truth_lig="open_combind/tests/structures/aligned/{pdbid}_lig_truth.sdf"
    truth_complex="open_combind/tests/structures/aligned/{pdbid}_complex_truth.pdb"
    for pdbid in input_structs:
        prot_truth = parsePDB(truth_complex.format(pdbid=pdbid))
        prot_test = parsePDB(aligned_prot.format(pdbid=pdbid))
        assert calcRMSD(prot_test,prot_truth) == 0

    lig_test = next(ForwardSDMolSupplier("open_combind/tests/structures/aligned/{pdbid}/{pdbid}_lig.sdf".format(pdbid=input_structs[0])))
    lig_truth = next(ForwardSDMolSupplier(truth_lig.format(pdbid=input_structs[0])))
    assert CalcRMS(lig_test, lig_truth) == 0
     
    
def test_struct_sort():
    input_structs=["1FKN","3UDH"]
    struct_sort(input_structs)

    prot_truth = "open_combind/tests/structures/proteins/{pdbid}_prot_truth.pdb"
    prot_test = "open_combind/tests/structures/proteins/{pdbid}_prot.pdb"
    lig_truth = "open_combind/tests/structures/ligands/{sdfid}_lig_truth.sdf"
    lig_test = "open_combind/tests/structures/ligands/{sdfid}_lig.sdf"
    for pdbid in input_structs:
        prot_tu = parsePDB(prot_truth.format(pdbid=pdbid))
        prot_te = parsePDB(prot_test.format(pdbid=pdbid))
        assert prot_tu.numAtoms('protein') == prot_te.numAtoms('protein')
        assert prot_te.numAtoms('hetero') == 0
        assert prot_te.numAtoms('water') == 0
        assert calcRMSD(prot_te,prot_tu) == 0

        lig_tu = next(ForwardSDMolSupplier(lig_truth.format(pdbid=pdbid)))
        lig_te = next(ForwardSDMolSupplier(lig_test.format(pdbid=pdbid)))
        assert lig_tu.GetNumHeavyAtoms() == lig_te.GetNumHeavyAtoms()
        assert lig_tu.GetNumBonds() == lig_te.GetNumBonds()
        assert CalcRMS(lig_te, lig_tu) == 0
