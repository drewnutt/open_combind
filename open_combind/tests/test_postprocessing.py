import pytest
from open_combind.dock.postprocessing import coalesce_poses, write_poses
from rdkit.Chem import ForwardSDMolSupplier

def test_coalesce_poses():
    sort_file = "open_combind/tests/test_sort.sdf"
    sorted_poses = coalesce_poses(sort_file)
    assert len(sorted_poses) == 40
    last_score = 1.1
    for pose in sorted_poses:
        assert pose.GetProp("CNNscore") <= last_score
        last_score = pose.GetProp("CNNscore")

    sort_files = [sort_file, "open_combind/tests/test_sort_1.sdf"]
    sorted_poses = coalesce_poses(sort_files)
    assert len(sorted_poses) == 80
    last_score = 1.1
    for pose in sorted_poses:
        assert pose.GetProp("CNNscore") <= last_score
        last_score = pose.GetProp("CNNscore")

def test_write_poses():
    sort_file = "open_combind/tests/test_sort.sdf"
    sorted_poses = coalesce_poses(sort_file)
    write_poses(sorted_poses, "open_combind/tests/test_write.sdf.gz")
    with fileinput.hook_compressed("open_combind/tests/test_write.sdf.gz") as f:
        supplier = ForwardSDMolSupplier(f)
        count = 0
        for mol in supplier:
            count += 1
        assert count == 40


    
