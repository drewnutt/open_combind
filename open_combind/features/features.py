import os
import gzip
import numpy as np
import pandas as pd
from glob import glob
from rdkit.Chem import AllChem as Chem
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
    """
    def __init__(self, root, ifp_version='rd1', shape_version='pharm_max',
                 max_poses=10000, pv_root=None,
                 ifp_features=['hbond', 'saltbridge', 'contact'], cnn_scores=True):
        self.root = os.path.abspath(root)
        if pv_root is None:
            self.pv_root = self.root + '/docking'

        self.ifp_version = ifp_version
        self.shape_version = shape_version
        self.max_poses = max_poses
        self.ifp_features = ifp_features
        self.cnn_scores = cnn_scores

        self.raw = {}

    def get_molecules_from_files(self,pvs,native=False):
        molbundle_dict = dict()
        for pv in pvs:
            # mol_bundle = Chem.FixedMolSizeMolBundle()
            mol_bundle = []
            pv_open = pv
            if pv.endswith('.gz'):
                pv_open = gzip.open(pv)
            mol_suppl = Chem.ForwardSDMolSupplier(pv_open)
            for idx, mol in enumerate(mol_suppl):
                mol_bundle.append(mol)
                if (idx + 1) == self.max_poses:
                    break
            if (native is False) and (len(mol_bundle) != self.max_poses):
                print(f"Did not get {self.max_poses} poses for {pv}, only {len(mol_bundle)} poses")
            molbundle_dict[pv] = mol_bundle
        return molbundle_dict

    def path(self, name, base=False, pv=None, pv2=None):
        if base:
            return '{}/{}'.format(self.root, name)

        if self.pv_root != self.root+'/docking':
            if pv is not None:
                pv = pv.replace(self.pv_root), self.root+'/single'
            if pv2 is not None:
                pv2 = pv2.replace(self.pv_root), self.root+'/single'

        # single features
        if name == 'rmsd':
            return pv.replace('.sdf.gz', '_rmsd.npy')
        elif name == 'gscore':
            return pv.replace('.sdf.gz', '_gscore.npy')
        elif name == 'gaff':
            return pv.replace('.sdf.gz', '_gaff.npy')
        elif name == 'vaff':
            return pv.replace('.sdf.gz', '_vaff.npy')
        elif name == 'name':
            return pv.replace('.sdf.gz', '_name.npy')
        elif name == 'ifp':
            suffix = '_ifp_{}.csv'.format(self.ifp_version)
            return pv.replace('.sdf.gz', suffix)

        # pair features
        elif name == 'shape':
            return f'{self.root}/shape.npy'
        elif name == 'mcss':
            return f'{self.root}/mcss.npy'
        else:
            return f'{self.root}/{name}.npy'

    def load_features(self):
        paths = glob(f'{self.root}/*.npy')
        for path in paths:
            name = path.split('/')[-1][:-4]
            self.raw[name] = np.load(path)

    def get_view(self, ligands, features):
        """
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

    def load_single_features(self, pvs, ligands=None):
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

            sts = Chem.ForwardSDMolSupplier(gzip.open(pv))
            _poses = [st for st in sts]

            keep = []
            for i in range(len(_names)):
                if ((ligands == None or (_names[i] in ligands))
                    and sum(_names[:i] == _names[i]) < self.max_poses):
                    keep += [i]
            rmsds += [_rmsds[keep]]
            if self.cnn_scores:
                gscores += [_gscores[keep]]
                gaffs += [_gaffs[keep]]
            vaffs += [_vaffs[keep]]
            names += [_names[keep]]
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
        # For single features, there is no need to keep sub-sets of ligands
        # separated,  so just merge them at the outset to simplify the rest of
        # the method.
        if type(pvs[0]) == list:
            pvs = [pv for _pvs in pvs for pv in _pvs]

        pvs = [os.path.abspath(pv) for pv in pvs]
        molbundles = self.get_molecules_from_files(pvs)
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
        mkdir(self.root)
        rmsds1, gscores1, gaffs1, vaffs1, poses1, names1, ifps1 = self.load_single_features(pvs)
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
        names = []
        docked_fname = os.path.basename(out).split('.')[0]
        name = docked_fname.replace('-docked_name','')
        for idx, st in enumerate(bundle):
            names += [name]
        np.save(out, names)

    def compute_gaff(self, bundle, out):
        gaffs = []
        for idx, st in enumerate(bundle):
            gaffs += [float(st.GetProp('CNNaffinity'))]
        np.save(out, gaffs)

    def compute_gscore(self, bundle, out):
        gscores = []
        for idx, st in enumerate(bundle):
            gscores += [logit(float(st.GetProp('CNNscore')))]
        np.save(out, gscores)

    def compute_vaff(self, bundle, out):
        vaffs = []
        for idx, st in enumerate(bundle):
            vaffs += [float(st.GetProp('minimizedAffinity'))]
        np.save(out, vaffs)

    def compute_rmsd(self, bundle, native_poses, out):
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
        from open_combind.features.ifp import ifp
        settings = IFP[self.ifp_version]
        ifp(settings, pv, out, self.max_poses)

    def compute_ifp_pair(self, ifps1, ifps2, feature, out, processes=1):
        if processes != 1:
            from open_combind.features.ifp_similarity import ifp_tanimoto_mp
            tanimotos = ifp_tanimoto_mp(ifps1, ifps2, feature, processes)
        else:
            from open_combind.features.ifp_similarity import ifp_tanimoto
            tanimotos = ifp_tanimoto(ifps1, ifps2, feature)
        np.save(out, tanimotos)

    def compute_shape(self, poses1, poses2, out, processes=1):
        if processes != 1:
            from open_combind.features.shape import shape_mp
            sims = shape_mp(poses2, poses1, version=self.shape_version,processes=processes).T
        else:
            from open_combind.features.shape import shape
            # More efficient to have longer pose list provided as second argument.
            # This only matters for screening.
            sims = shape(poses2, poses1, version=self.shape_version).T
        np.save(out, sims)

    def compute_mcss(self, poses1, poses2, out, processes=1):
        if processes != 1:
            from open_combind.features.mcss import mcss_mp
            rmsds = mcss_mp(poses1, poses2, processes)
        else:
            from open_combind.features.mcss import mcss
            rmsds = mcss(poses1, poses2)
        np.save(out, rmsds)
