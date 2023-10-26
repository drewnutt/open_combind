"""Open-source docking pipeline leveraging pairwise statistics"""

# import subprocess
# subprocess.run(['source','setup.sh'])
# Add imports here
from .open_combind import structprep, ligprep, dock_ligands, featurize, pose_prediction
# from .features.features import Features
# from .dock.struct_process import struct_process
# from .dock.struct_align import struct_align
# from .dock.struct_sort import struct_sort
# from .score import *

from ._version import __version__
