"""Open-source docking pipeline leveraging pairwise statistics"""

# import subprocess
# subprocess.run(['source','setup.sh'])
# Add imports here
from .open_combind import structprep, ligprep, dock_ligands, featurize, pose_prediction, scores_to_csv
from .features.features import Features
# from .dock.struct_process import struct_process
# from .dock.struct_align import struct_align
# from .dock.struct_sort import struct_sort
# from .score import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

from . import _version
__version__ = _version.get_versions()['version']

