from open_combind import structprep
from open_combind.dock.struct_align import struct_align
from open_combind.dock.struct_sort import struct_sort
from open_combind.dock.struct_process import struct_process
from open_combind.dock.grid import make_grid

def test_struct_process():
    input_file=["open_combind/tests/structures/raw/1FKN.pdb"]
    struct_process(input_file)

# def test_struct_align():
    
