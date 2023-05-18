.. _intro/installation:

============
Installation
============

Open-Combind set-up
-------------------
Open-Combind is a fork of `Combind <https://github.com/drorlab/combind>`_ that uses open-source software for each component of the pipeline. Open-Combind is written in Python 3 and has been tested on Linux operating systems.

Open-Combind can be installed directly from the `GitHub Repository <https://github.com/drewnutt/open_combind>`_.

Pre-requisites
--------------
Open-Combind requires the following software to be installed:
- PDBFixer

PDBFixer can be installed using conda/mamba:

.. code-block:: bash

        conda install -c conda-forge pdbfixer


Installation from source
------------------------
Open-Combind can be cloned from the `Open-Combind GitHub Repository <https://github.com/drewnutt/open_combind>`_ or by using git:

.. code-block:: bash

        git clone git@github.com:drewnutt/open_combind.git

You can then install the repository into your python environment by navigating to the top level of the repository (the directory containing ``setup.py``) and using pip to install the package.

.. code-block:: bash
 
        python -m pip install .

Finally, to run the pose prediction pipeline we need `GNINA <https://github.com/gnina/gnina>`_.

It is strongly recommended to install GNINA from source as follows.

.. code-block:: bash

        git clone https://github.com/gnina/gnina.git
        cd gnina
        mkdir build
        cd build
        cmake ..
        make
        make install


A `pre-compiled binary <https://github.com/gnina/gnina/releases/>`_ is available if you are unable to install GNINA for any reason or if you do not care about the speed of docking.


