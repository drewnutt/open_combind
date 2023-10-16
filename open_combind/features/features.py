import os
import gzip
import numpy as np
import pandas as pd
from glob import glob
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit.Geometry.rdGeometry import Point3D
from open_combind.utils import basename, mp, mkdir, np_load
from scipy.special import logit

IFP = {'rd1':    {'version'           : 'rd1',
                   'level'             : 'residue',
                   'hbond_dist_opt'    : 2.5,
                   'hbond_dist_cut'    : 3.0,
                   'hbond_angle_opt'   : 60.0,
                   'hbond_angle_cut'   : 90.0,
                   'sb_dist_opt'       : 4.0,
                   'sb_dist_cut'       : 5.0,
                   'contact_scale_opt' : 1.25,
                   'contact_scale_cut' : 1.75,},
      }

# add an option to change which score you get from gnina
class Features:
    """
    Organize feature computation and loading.
    
    Parameters
    ----------
    root : str
        Path to root directory of poses/features
    ifp_version : str, default='rd1'
        Version of the interaction fingerprint to use for featurization
    mcss_param : str, default='strict'
        MCSS parameter set to use for MCSS RMSD featurization, either ``'strict'`` or ``'relaxed'``
    max_poses : str, default=10000  
        Maximum number of poses per ligand used to compute features
    pv_root
        Root of the directory containing the poses, if not specified then set to ``{root}/docking``
    ifp_features : :class:`list[str]<list>`, default=['hbond', 'saltbridge', 'contact']
        Interaction fingerprint features to calculate
    cnn_scores : bool, default=True
        Keep track of CNN produced scores from GNINA (i.e. CNNscore and CNNaffinity)
        
    Attributes
    ----------
    root : str
        Root directory where the features/poses should be looked for
    ifp_version : str
        Which version of the interaction fingerprint to use
    shape_version : str
        (not currently used) which version of the shape algorithm to use for featurization
    max_poses : int
        Maximum number of poses to handle for each ligand
    ifp_features : :class:`list[str]<list>`
        Which interaction features to track
    cnn_scores : bool
        Keep track of CNN produced scores from GNINA (i.e. CNNscore and CNNaffinity)
    raw : dict
        Contains the raw data of the computed (or loaded) features where the key is the feature name
    """
    def __init__(self, root, ifp_version='rd1', mcss_param='strict',
                 max_poses=10000, pv_root=None,
                 ifp_features=['hbond', 'saltbridge', 'contact'], cnn_scores=True,
                 template='',check_center_ligs=False):
        self.root = os.path.abspath(root)
        if pv_root is None:
            self.pv_root = self.root + '/docking'
        else:
            self.pv_root = pv_root

        self.ifp_version = ifp_version
        self.mcss_param = mcss_param
        self.max_poses = max_poses
        self.ifp_features = ifp_features
        self.cnn_scores = cnn_scores
        self.template = template
        self.check_center_ligs = check_center_ligs
        if len(self.template):
            with open(self.template) as tmpfile:
                template_line = tmpfile.readlines()[0]
            abox_ligand = template_line.split('--autobox_ligand')[-1].split()[0]
        self.center_ligand = Chem.MolFromMolFile(abox_ligand) if self.check_center_ligs else None

        self.raw = {}

    def get_molecules_from_files(self, pvs, native=False, center_ligand=None):
        """
        .. include :: <isotech.txt>
        Load molecules from docked files containing many poses of a ligand.
        
        If `center_ligand` is set, then the poses will be filtered to only include poses
        whose centroid are within 7.5 |angst| of the centroid of the `center_ligand`.

        Parameters
        ----------
        pvs : :class:`list[str]<list>`
            List of pose viewer files to load
        native : bool, default=False
            Whether the loaded file contains a native pose. Only one pose should be there.
        center_ligand : :class:`~rdkit.Chem.rdchem.Mol`
            Center ligand to use for filtering poses

        Returns
        -------
        molbundle_dict : :class:`dict[str, list[~rdkit.Chem.rdchem.Mol]]<dict>`
            Dictionary of pose viewer files to list of molecules
        """
        molbundle_dict = dict()
        center = Point3D(0,0,0)
        if center_ligand is not None:
            center = ComputeCentroid(center_ligand.GetConformer())
        for pv in pvs:
            # mol_bundle = Chem.FixedMolSizeMolBundle()
            mol_bundle = []
            pv_open = pv
            if pv.endswith('.gz'):
                pv_open = gzip.open(pv)
            mol_suppl = Chem.ForwardSDMolSupplier(pv_open)
            mol_count = 0
            for mol in mol_suppl:
                lig_centroid = ComputeCentroid(mol.GetConformer())
                displacement = lig_centroid.DirectionVector(center) * lig_centroid.Distance(center)
                # print(distance)
                if center_ligand is not None and (np.abs(displacement.x) > 7.5 or np.abs(displacement.y) > 7.5 or np.abs(displacement.z) > 7.5):
                    print(f"skipped for {pv}")
                    continue
                mol_bundle.append(mol)
                mol_count += 1
                if mol_count == self.max_poses:
                    break
            if (native is False) and (mol_count != self.max_poses):
                print(f"Did not get {self.max_poses} poses for {pv}, only {len(mol_bundle)} poses")
            print(mol_count,pv)
            molbundle_dict[pv] = mol_bundle
        return molbundle_dict

    def path(self, name, base=False, pv=None, pv2=None):
        """
        Get the path to a feature file

        Parameters
        ----------
        name : str
            Name of the feature
        base : bool, default=False
            Whether to return the path to the base feature file
        pv : str, default=None
            Path to the pose viewer file
        pv2 : str, default=None
            Path to the second pose viewer file
        
        Returns
        -------
        path : str
            Path to the feature file
        """

        if base:
            return '{}/{}'.format(self.root, name)

        # single features
        if name in ['rmsd', 'gscore', 'gaff', 'vaff', 'name','ifp']:
            if name == 'ifp':
                ext = '_ifp_{}.csv'.format(self.ifp_version)
            else:
                ext = '_{}.npy'.format(name)
            return pv.replace('.gz', '').replace('.sdf', ext)

        # pair features
        else:
            return f'{self.root}/{name}.npy'

    def load_features(self):
        """
        Load all of the features into self.raw
        """

        paths = glob(f'{self.root}/*.npy')
        for path in paths:
            name = path.split('/')[-1][:-4]
            self.raw[name] = np.load(path)

    def get_view(self, ligands, features):
        """
        Load the pairwise features `features` for the ligands in `ligands`

        Parameters
        ----------
        ligands : :class:`list[str]<list>`
            Ligands features should be loaded
        features : :class:`list[str]<list>`
            Features that should be loaded

        Returns
        -------
        dict
            Pairwise `features` for all of the ligand poses of the ligands in `ligands`
        """
        data = {}
        if self.cnn_scores:
            data['gscore'] = {}
            data['gaff'] = {}
        data['vaff'] = {}
        data['rmsd'] = {}
        for ligand in ligands:
            mask = self.raw['name1'] == ligand
            assert sum(mask)
            if self.cnn_scores:
                data['gscore'][ligand] = self.raw['gscore1'][mask]
                data['gaff'][ligand] = self.raw['gaff1'][mask]
            data['vaff'][ligand] = self.raw['vaff1'][mask]
            data['rmsd'][ligand] = self.raw['rmsd1'][mask]

        for feature in features:
            data[feature] = {}
            for i, ligand1 in enumerate(ligands):
                for ligand2 in ligands[i+1:]:
                    mask1 = self.raw['name1'] == ligand1
                    mask2 = self.raw['name1'] == ligand2
                    data[feature][(ligand1, ligand2)] = self.raw[feature][mask1, :][:, mask2]

        return data

    def load_single_features(self, pvs, ligands=None, center_ligand=None):
        """
        Load the single pose features (e.g. docking score, RMSD, IFP, etc.) of the poses

        Parameters
        ----------
        pvs : :class:`list[str]<list>`
            Poses that need features loaded
        ligands : 

		center_ligand: :class:`~rdkit.Chem.rdchem.Mol`, default=None
			Ligand to center the poses around

        Returns
        -------
        list
            List of rmsds to the native pose for each loaded ligand
        list
            List of docking scores for each loaded ligand
        list
            List of GNINA computed CNNaffinity scores for each loaded ligand
        list
            List of Vina Affinity scores for each loaded ligand
        list
            List of poses for each loaded ligand
        list
            List of names for each loaded ligand
        list
            List of IFPs for each loaded ligand
        """

        if center_ligand is not None:
            center = ComputeCentroid(center_ligand.GetConformer())
        rmsds, gscores, gaffs, vaffs, poses, names, ifps = [], [], [], [], [], [], []
        for pv in pvs:
            _rmsds = np.load(self.path('rmsd', pv=pv))
            if self.cnn_scores:
                _gscores = np.load(self.path('gscore', pv=pv))
                _gaffs = np.load(self.path('gaff', pv=pv))
            _vaffs = np.load(self.path('vaff', pv=pv))
            _names = np.load(self.path('name', pv=pv))

            _ifps = pd.read_csv(self.path('ifp', pv=pv))
            _ifps = [_ifps.loc[_ifps.pose==p] for p in range(max(_ifps.pose)+1)]

            #Need to check for if ligand is centered here.
            sts = Chem.ForwardSDMolSupplier(gzip.open(pv))
            _poses = []
            for st in sts:
                if center_ligand is not None:
                    lig_centroid = ComputeCentroid(st.GetConformer())
                    displacement = lig_centroid.DirectionVector(center) * lig_centroid.Distance(center)
                    # print(distance)
                    if np.abs(displacement.x) > 7.5 or np.abs(displacement.y) > 7.5 or np.abs(displacement.z) > 7.5:
                        print(f"skipped for {pv}")
                        continue
                _poses.append(st)
            print(len(poses))

            keep = []
            for i in range(len(_names)):
                if ((ligands == None or (_names[i] in ligands))
                    and sum(_names[:i] == _names[i]) < self.max_poses):
                    keep += [i]
            print(keep)
            rmsds += [_rmsds[keep]]
            if self.cnn_scores:
                gscores += [_gscores[keep]]
                gaffs += [_gaffs[keep]]
            vaffs += [_vaffs[keep]]
            names += [_names[keep]]
            for i in keep:
                try:
                    assert _poses[i]
                except:
                    print(i)
            poses += [_poses[i] for i in keep]
            ifps += [_ifps[i] for i in keep]

        rmsds = np.hstack(rmsds)
        names = np.hstack(names)
        vaffs = np.hstack(vaffs)
        if self.cnn_scores:
            gaffs = np.hstack(gaffs)
            gscores = np.hstack(gscores)
        else:
            gaffs = None
            gscores = None
        return rmsds, gscores, gaffs, vaffs, poses, names, ifps

    def compute_single_features(self, pvs, native_poses):
        """
        Compute all of the single pose features (e.g. GNINA scores, RMSD to native, etc.) for the provided poses

        Parameters
        ----------
        pvs : :class:`list[str]<list>`
            Path to poses to compute the single pose features
        native_poses : :class:`list[str]<list>`
            Path to pose of the native ligand structures, if they exist
        """
        # For single features, there is no need to keep sub-sets of ligands
        # separated,  so just merge them at the outset to simplify the rest of
        # the method.
        if type(pvs[0]) == list:
            pvs = [pv for _pvs in pvs for pv in _pvs]
        pvs = [os.path.abspath(pv) for pv in pvs]
        molbundles = self.get_molecules_from_files(pvs, center_ligand=self.center_ligand)
        native_sts = self.get_molecules_from_files(list(native_poses.values()), native=True)
        native_poses = {name: native_sts[pv][0] for name, pv in native_poses.items()}

        if self.cnn_scores:
            print('Extracting GNINA affinities.')
            for pv, bundle in molbundles.items():
                out = self.path('gaff', pv=pv)
                if not os.path.exists(out):
                    self.compute_gaff(bundle, out)

            print('Extracting GNINA pose score.')
            for pv, bundle in molbundles.items():
                out = self.path('gscore', pv=pv)
                if not os.path.exists(out):
                    self.compute_gscore(bundle, out)

        print('Extracting Vina minimizedAffinity.')
        for pv, bundle in molbundles.items():
            out = self.path('vaff', pv=pv)
            if not os.path.exists(out):
                self.compute_vaff(bundle, out)

        print('Extracting names.')
        for pv, bundle in molbundles.items():
            out = self.path('name', pv=pv)
            if not os.path.exists(out):
                self.compute_name(bundle, out)

        print('Computing RMSDs to native poses')
        for pv, bundle in molbundles.items():
            out = self.path('rmsd', pv=pv)
            if not os.path.exists(out):
                self.compute_rmsd(bundle, native_poses, out)

        print('Computing interaction fingerprints.')
        for pv in molbundles.keys():
            # print(pv)
            out = self.path('ifp', pv=pv)
            if not os.path.exists(out):
                self.compute_ifp(pv, out)

    def compute_pair_features(self, pvs, pvs2=None, ifp=True, shape=True, mcss=True, processes=1):
        """
        Computes the pairwise features for the poses in `pvs` and `pvs2`. If `pvs2` is not specified,
        then the pairwise features are computed between the poses in `pvs`

        Requires the `compute_single_features` method to have been run first as the IFPs are loaded from disk

        Parameters
        ----------
        pvs : :class:`list[str]<list>`
            Path to poses
        pvs2 : list[str]<list
            Path to second set of poses to pair with `pvs` for pairwise features
        ifp : bool, default=True
            Compute the pairwise interaction fingerprint feature
        shape : bool, default=True
            Compute the pairwise shape feature
        mcss : bool, default=True
            Compute the pairwise maximum common substructure RMSD feature
        processes : int, default=1
            Number of processes to use for computing the pairwise features, if -1 then use all available cores
        """
        mkdir(self.root)
        rmsds1, gscores1, gaffs1, vaffs1, poses1, names1, ifps1 = self.load_single_features(pvs, center_ligand=self.center_ligand)
        out = self.path('rmsd1')
        np.save(out, rmsds1)
        out = self.path('gscore1')
        np.save(out, gscores1)
        out = self.path('gaff1')
        np.save(out, gaffs1)
        out = self.path('vaff1')
        np.save(out, vaffs1)
        out = self.path('name1')
        np.save(out, names1)
        if pvs2 is None:
            (rmsds2, gscores2, poses2, names2, ifps2
            ) = rmsds1, gscores1, poses1, names1, ifps1
        else:
            rmsds2, gscores2, poses2, names2, ifps2 = self.load_single_features(pvs2)
            out = self.path('rmsd2')
            np.save(out, rmsds2)
            out = self.path('gscore2')
            np.save(out, gscores2)
            out = self.path('name2')
            np.save(out, names2)

        if ifp:
            print('Computing interaction similarities.')
            for feature in self.ifp_features:
                out = self.path(feature)
                if not os.path.exists(out):
                    self.compute_ifp_pair(ifps1, ifps2,  feature, out, processes=processes)

        if shape:
            print('Computing shape similarities.')
            out = self.path('shape')
            if not os.path.exists(out):
                self.compute_shape(poses1, poses2, out)

        if mcss:
            print('Computing mcss similarities.')
            out = self.path('mcss')
            if not os.path.exists(out):
                self.compute_mcss(poses1, poses2, out, processes=processes)

    # Methods to calculate features
    def compute_name(self, bundle, out):
        """
        Get the name for all of the ligand poses and save as `.npy`

        Parameters
        ----------
        bundle : :class:`list[Mol]<list>`
            :class:`~rdkit.Chem.rdchem.Mol` s to process
        out : str
            Path to `.npy` file to save all of the pose names
        """

        names = []
        docked_fname = os.path.basename(out).split('.')[0]
        name = docked_fname.replace('-docked_name','')
        for idx, st in enumerate(bundle):
            names += [name]
        np.save(out, names)

    def compute_gaff(self, bundle, out):
        """
        Retrieve the GNINA computed CNNaffinity for all of the poses
        
        Parameters
        ----------
        bundle : :class:`list[Mol]<list>`
            :class:`~rdkit.Chem.rdchem.Mol` s to process
        out : str
            Path to `.npy` file to save all of the pose CNNaffinities
        """
        
        gaffs = []
        for idx, st in enumerate(bundle):
            gaffs += [float(st.GetProp('CNNaffinity'))]
        np.save(out, gaffs)

    def compute_gscore(self, bundle, out):
        """
        Retrieve the GNINA computed CNNscore for all of the poses 
        
        Parameters
        ----------
        bundle : :class:`list[Mol]<list>`
            :class:`~rdkit.Chem.rdchem.Mol` s to process
        out : str
            Path to `.npy` file to save all of the pose CNNaffinities
        """
        
        gscores = []
        for idx, st in enumerate(bundle):
            gscores += [logit(float(st.GetProp('CNNscore')))]
        np.save(out, gscores)

    def compute_vaff(self, bundle, out):
        """
        Retrieve the Autodock Vina scores for all of the poses
        
        Parameters
        ----------
        bundle : :class:`list[Mol]<list>`
            :class:`~rdkit.Chem.rdchem.Mol` s to process
        out : str
            Path to `.npy` file to save all of the pose Vina scores
        
        """
        
        vaffs = []
        for idx, st in enumerate(bundle):
            vaffs += [float(st.GetProp('minimizedAffinity'))]
        np.save(out, vaffs)

    def compute_rmsd(self, bundle, native_poses, out):
        """
        Compute the root mean square deviation (RMSD) from the pose to its native pose, if available.
        
        Parameters
        ----------
        bundle : :class:`list[Mol]<list>`
            :class:`~rdkit.Chem.rdchem.Mol` s to process
        native_poses : :class:`dict[str,Mol]<dict>`
            Dictionary with keys as ligand names and values as :class:`~rdkit.Chem.rdchem.Mol` s of the native ligand poses
        out : str
            Path to `.npy` file to save all of the pose RMSDs
        """
        
        rmsds = []
        name = os.path.basename(out).split('-')[0]
        # print(name)
        if name in native_poses:
            native = native_poses[name]
            for idx, st in enumerate(bundle):
                rmsd = Chem.CalcRMS(native, st)
                rmsds += [rmsd]
        else:
            # print(name)
            rmsds = [-1] * len(bundle)

        np.save(out, rmsds)

    def compute_ifp(self, pv, out):
        """
        Compute the interaction fingerprint (IFP) of the ligand poses based on the ``ifp_version``
        
        Parameters
        ----------
        pv : :class:`list[str]<list>`
            Path to poses
        out : str
            Path to `.npy` file to save all of the pose IFPs
        
        """
        
        from open_combind.features.ifp import ifp
        settings = IFP[self.ifp_version]
        ifp(settings, pv, out, self.max_poses)

    def compute_ifp_pair(self, ifps1, ifps2, feature, out, processes=1):
        """
        Compute the pseudo-Tanimoto similarity between the given IFPs for the given features
        
        Parameters
        ----------
        ifps1 : :class:`list[DataFrame]<list>`
            List of pose IFPs to calculate pairwise with `ifps2` 
        ifps2 : :class:`list[DataFrame]<list>`
            List of pose IFPs to calculate pairwise with `ifps1` 
        feature : :class:`list[str]<list>`
            List of IFP features to calculate the pseudo-Tanimoto similarity between
        out : str
            Path to `.npy` file to save all of the pairwise IFPs
        processes : int, default=1
            Number of processes to use for computing the pairwise features, if -1 then use all available cores
        """
        
        if processes != 1:
            from open_combind.features.ifp_similarity import ifp_tanimoto_mp
            tanimotos = ifp_tanimoto_mp(ifps1, ifps2, feature, processes)
        else:
            from open_combind.features.ifp_similarity import ifp_tanimoto
            tanimotos = ifp_tanimoto(ifps1, ifps2, feature)
        np.save(out, tanimotos)

    def compute_shape(self, poses1, poses2, out, processes=1):
        """
        Compute the pseudo-Tanimoto similarity of the shape between the given poses
        
        Parameters
        ----------
        poses1 : :class:`list[Mol]<list>`
            :class:`~rdkit.Chem.rdchem.Mol` s of poses to calculate pairwise with `poses2` 
        poses2 : :class:`list[Mol]<list>`
            :class:`~rdkit.Chem.rdchem.Mol` s of poses to calculate pairwise with `poses1` 
        out : str
            Path to `.npy` file to save all of the pairwise shape similarities
        processes : int, default=1
            Number of processes to use for computing the pairwise features, if -1 then use all available cores
        """
        if processes != 1:
            from open_combind.features.shape import shape_mp
            sims = shape_mp(poses2, poses1,processes=processes).T
        else:
            from open_combind.features.shape import shape
            # More efficient to have longer pose list provided as second argument.
            # This only matters for screening.
            sims = shape(poses2, poses1).T
        np.save(out, sims)

    def compute_mcss(self, poses1, poses2, out, processes=1):
        """
        Compute the Maximum Common Substructure (MCSS) RMSD between the given poses
        
        Parameters
        ----------
        poses1 : :class:`list[Mol]<list>`
            :class:`~rdkit.Chem.rdchem.Mol` s of poses to calculate pairwise with `poses2` 
        poses2 : :class:`list[Mol]<list>`
            :class:`~rdkit.Chem.rdchem.Mol` s of poses to calculate pairwise with `poses1` 
        out : str
            Path to `.npy` file to save all of the MCSS RMSDs
        processes : int, default=1
            Number of processes to use for computing the pairwise features, if -1 then use all available cores
        """
        if processes != 1:
            from open_combind.features.mcss import mcss_mp
            rmsds = mcss_mp(poses1, poses2, param_string=self.mcss_param, processes=processes)
        else:
            from open_combind.features.mcss import mcss
            rmsds = mcss(poses1, poses2, param_string=self.mcss_param)
        np.save(out, rmsds)
