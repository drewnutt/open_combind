import os
import pytest
from open_combind import structprep
from open_combind.dock.struct_align import struct_align
from open_combind.dock.struct_sort import struct_sort
from open_combind.dock.struct_process import struct_process
from open_combind.dock.grid import make_grid
from prody import parsePDB,calcRMSD
from rdkit.Chem import MolFromMolFile
from rdkit.Chem.rdMolAlign import CalcRMS

@pytest.fixture
def list_of_pdbs():
    return ["1FKN","3UDH"]

@pytest.fixture
def correct_process_structs(list_of_pdbs):
    truth_prot="open_combind/tests/structures/processed/{pdbid}_prot_truth.pdb"
    truth_complex="open_combind/tests/structures/processed/{pdbid}_complex_truth.pdb"
    complex_paths = []
    for pdbid in list_of_pdbs:
        complex_paths.append(truth_complex.format(pdbid=pdbid))

    prot_paths = []
    for pdbid in list_of_pdbs:
        prot_paths.append(truth_prot.format(pdbid=pdbid))

    return complex_paths, prot_paths

def test_struct_process(list_of_pdbs, correct_process_structs):
    filtered_protein="open_combind/tests/structures/processed/{pdbid}/{pdbid}_prot.pdb"
    filtered_ligand="open_combind/tests/structures/processed/{pdbid}/{pdbid}_lig.sdf"
    filtered_complex="open_combind/tests/structures/processed/{pdbid}/{pdbid}_complex.pdb"
    struct_process(list_of_pdbs,
            protein_in="open_combind/tests/structures/raw/{pdbid}.pdb",
            ligand_info="open_combind/tests/structures/raw/{pdbid}.info",
            filtered_protein=filtered_protein,
            filtered_complex=filtered_complex,
            filtered_ligand=filtered_ligand,
            filtered_hetero="open_combind/tests/structures/processed/{pdbid}/{pdbid}_het.pdb",
            filtered_water="open_combind/tests/structures/processed/{pdbid}/{pdbid}_wat.pdb")

    complex_paths, prot_paths = correct_process_structs
    for pdbid, truth_complex, truth_prot in zip(list_of_pdbs, complex_paths, prot_paths):
        prot_truth = parsePDB(truth_prot)
        prot_test = parsePDB(filtered_protein.format(pdbid=pdbid))
        assert prot_truth.numAtoms('protein') == prot_test.numAtoms('protein')
        assert prot_test.numAtoms('hetero') == 0
        assert prot_test.numAtoms('water') == 0
        assert calcRMSD(prot_test,prot_truth) == 0

        #eval complex (more important because used downstream)
        complex_truth = parsePDB(truth_complex)
        complex_test = parsePDB(filtered_complex.format(pdbid=pdbid))
        assert complex_truth.numAtoms('protein') == complex_test.numAtoms('protein')
        assert complex_truth.numAtoms('hetero') == complex_test.numAtoms('hetero')
        assert complex_test.numAtoms('water') == 0
        assert calcRMSD(complex_test,complex_truth) == 0


@pytest.fixture
def correct_align_structs(list_of_pdbs):
    truth_lig="open_combind/tests/structures/aligned/{pdbid}_lig_truth.sdf"
    truth_complex="open_combind/tests/structures/aligned/{pdbid}_complex_truth.pdb"
    complex_paths = []
    for pdbid in list_of_pdbs:
        complex_paths.append(truth_complex.format(pdbid=pdbid))

    lig_paths = []
    for pdbid in list_of_pdbs:
        lig_paths.append(truth_lig.format(pdbid=pdbid))

    return complex_paths, lig_paths

def test_struct_align(list_of_pdbs, correct_align_structs):
    template = sorted(list_of_pdbs)[0]
    filtered_protein="open_combind/tests/structures/processed/{pdbid}/{pdbid}_complex.pdb"
    aligned_prot="open_combind/tests/structures/aligned/{pdbid}/{pdbid}_aligned.pdb"
    struct_align(template, list_of_pdbs, dist=15.0, retry=True,
            filtered_protein=filtered_protein,
            aligned_prot=aligned_prot,
            align_dir="open_combind/tests/structures/aligned")

    complex_paths, lig_paths = correct_align_structs
    for pdbid, truth_complex in zip(list_of_pdbs, complex_paths):
        prot_truth = parsePDB(truth_complex)
        prot_test = parsePDB(aligned_prot.format(pdbid=pdbid))
        assert calcRMSD(prot_test,prot_truth) == 0

    print(os.listdir("open_combind/tests/structures/aligned/3UDH/"))
    for pdbid, truth_lig in zip(list_of_pdbs[1:], lig_paths[1:]):
        lig_test = MolFromMolFile("open_combind/tests/structures/aligned/{pdbid}/{pdbid}_lig.sdf".format(pdbid=pdbid))
        lig_truth = MolFromMolFile(truth_lig)
        assert CalcRMS(lig_test, lig_truth) == 0
     
@pytest.fixture
def correct_sort_structs(list_of_pdbs):
    truth_prot = "open_combind/tests/structures/proteins/{pdbid}_prot_truth.pdb"
    prot_paths = []
    for pdbid in list_of_pdbs:
        prot_paths.append(truth_prot.format(pdbid=pdbid))

    truth_lig = "open_combind/tests/structures/ligands/{pdbid}_lig_truth.sdf"
    lig_paths = []
    for pdbid in list_of_pdbs:
        lig_paths.append(truth_lig.format(pdbid=pdbid))

    return prot_paths, lig_paths
    
    
def test_struct_sort(list_of_pdbs, correct_sort_structs):
    struct_sort(list_of_pdbs, opt_path = "open_combind/tests/structures/aligned/{pdbid}/{pdbid}_aligned.pdb")
    prot_paths, lig_paths = correct_sort_structs

    prot_test = "open_combind/tests/structures/proteins/{pdbid}_prot.pdb"
    lig_test = "open_combind/tests/structures/ligands/{pdbid}_lig.sdf"
    for pdbid, true_prot, true_lig in zip(list_of_pdbs, prot_paths, lig_paths):
        prot_tu = parsePDB(true_prot).select('heavy')
        prot_te = parsePDB(prot_test.format(pdbid=pdbid)).select('heavy')
        assert prot_tu.numAtoms('protein') == prot_te.numAtoms('protein')
        assert prot_te.numAtoms('hetero') == 0
        assert prot_te.numAtoms('water') == 0
        assert calcRMSD(prot_te,prot_tu) == 0

        lig_tu = MolFromMolFile(true_lig)
        lig_te = MolFromMolFile(lig_test.format(pdbid=pdbid))
        assert lig_tu.GetNumHeavyAtoms() == lig_te.GetNumHeavyAtoms()
        assert lig_tu.GetNumBonds() == lig_te.GetNumBonds()
        assert CalcRMS(lig_te, lig_tu) == 0
