import numpy as np
from scipy.special import logit

class PosePrediction:
    """
    Compute sets of poses that optimize the ComBind scoring function.

    Parameters
    ----------
    ligands: `list[str]<list>`
        names of ligands for which to predict poses.
    features: `list[str]<list>`
        names of features for computing similarity scores.
    data: dict
        Raw data.
    stats: `dict[str, ~open_combind.score.DensityEstimate]<dict>`
    alpha: float
        Factor to which to weight the GNINA CNNscores.
    max_poses: int
        largest number of poses for any ligand.
    single: :class:`~numpy.ndarray`
        1 x # ligands, array of ligand docking scores.
    pair: :class:`~numpy.ndarray`
        # ligands x # ligands x max_poses x max_poses, array of pairwise energy terms.
    """
    def __init__(self, ligands, features, data, stats, alpha, **kwargs):
        """
        Parameters
        ----------
        ligands: `list[str]<list>`
            names of ligands for which to predict poses.
        features: `list[str]<list>`
            names of features for computing similarity scores.
        data: dict
            Raw data.
        stats: `dict[str, ~open_combind.score.DensityEstimate]<dict>`
        alpha: float
            Factor to which to weight the GNINA CNNscores.
        """
        self.ligands = ligands
        self.features = features
        self.data = data
        self.stats = stats
        self.alpha = float(alpha)

        self.singlescore = 'gscore'
        if 'newscore' in kwargs and kwargs['newscore'] is not None:
            self.singlescore = kwargs['newscore']

        self.max_poses = self._get_max_poses()
        self.single = self._get_single()
        self.pair = self._get_pair()

    def _get_max_poses(self):
        """
        Get largest number of poses present for any ligand.

        Returns
        -------
        max_poses: int
           maximum number of poses for any ligand. 
        """
        return max(len(self.data[self.singlescore][ligand]) for ligand in self.ligands)

    def _get_single(self):
        """
        Transform docking scores into a # ligands x # poses array.

        * Scale docking scores by - self.alpha *

        Returns
        -------
        single: :class:`~numpy.ndarray`
            1 x # ligands, array of ligand docking scores.
        """
        single = [self.data[self.singlescore][ligand] for ligand in self.ligands]
        single = [self.pad(x, self.max_poses,C=np.nan) for x in single]
        single = np.vstack(single)
        single *= -self.alpha
        return single

    def _get_pair(self):
        """
        Transform pairwise similarities into pairwise energy terms stored
        in a # ligands x # ligands x max_poses x max_poses array.

        For each pairwise feature, compute the log ratio of feature
        likelihood in the native v. reference distribution.

        Sum over feature types to get a single energy term for each pose pair.

        Returns
        -------
        pair: :class:`~numpy.ndarray`
            # ligands x # ligands x max_poses x max_poses, array of pairwise energy terms.
        """
        pair = np.zeros((len(self.ligands), len(self.ligands),
                         self.max_poses, self.max_poses))
        for i, ligand1 in enumerate(self.ligands):
            for j, ligand2 in enumerate(self.ligands[i+1:]):
                j += i+1
                for feature in self.features:
                    stats = self.stats[feature]
                    raw = self.data[feature][(ligand1, ligand2)]

                    if raw[0, 0] == float('inf'):
                        # Features should either all be there or all be absent.
                        assert np.all(raw == float('inf'))
                        continue

                    energy = np.log(stats['native'](raw)) - np.log(stats['reference'](raw))
                    energy = self.pad(energy, self.max_poses, self.max_poses,C=-np.inf)
                    pair[i, j] += energy
                    pair[j, i] += energy.T
        return pair

    def max_posterior(self, max_iterations, restart):
        """
        Compute (probably) globally optimal pose set.

        Perform coordinant ascent from "restart" random initial configurations.

        Parameters
        ----------
        max_iterations: int 
            Maximum number of iterations to attempt before exiting.
        restart: int
            Number of times to run the optimization

        Returns
        -------
        poses: `dict[str, int]<dict>`
            {ligand_name: pose_number, }, where pose_number is the pose number selected by the objective
        """
        if len(self.ligands) == 1:
            return -float('inf'),{self.ligands[0]: 0}

        best_score, best_poses = -float('inf'), None
        for i in range(restart):
            if i == 0:
                poses = {lig: 0 for lig in self.ligands}
            else:
                poses = {lig: np.random.randint(self.max_poses)
                         for lig in self.ligands}

            poses = self.optimize_poses(poses, max_iterations)
            score = self.log_posterior(poses)
            if score > best_score:
                best_score = score
                best_poses = poses.copy()

            print(poses)
            print('run {}, score {}'.format(i, score))
        return best_poses, best_score

    def optimize_poses(self, poses, max_iterations):
        """
        Find (local) optimum by performing coordinate ascent starting from
        `poses`.

        Parameters
        ----------
        poses: `dict[str, int]<dict>`
            {ligand_name: current pose number, }
        max_iterations: int 
            Maximum number of iterations to attempt before exiting.

        Returns
        -------
        poses: `dict[str, int]<dict>`
            {ligand_name: pose_number, }, where pose_number is the pose number selected by the objective
        """
        for _ in range(max_iterations):
            update = False
            for query in np.random.permutation(list(poses.keys())):
                plp = self.partial_log_posterior(poses, query)
                best_pose = np.nanargmax(plp)
                if best_pose != poses[query]:
                    update = True
                    poses[query] = best_pose
            if not update:
                break
        else:
            print('Maximum iteractions reached.')
        return poses

    def partial_log_posterior(self, poses, query):
        """
        Returns the terms of the log posterior involving "query".

        Parameters
        ----------
        poses: `dict[str, int]<dict>`
            {ligand_name: current pose number, }, currently selected poses
        query: str
            Ligand name to compute partial log posterior for.

        Returns
        -------
        plp: :class:`~numpy.ndarray`
            1 x max_poses, array of partial log posterior terms.
        """
        iposes = {self.ligands.index(lig): pose
                  for lig, pose in poses.items()
                  if lig != query}
        iquery = self.ligands.index(query)

        plp = 0
        plp += self.single[iquery, :]
        for lig, pose in iposes.items():
            plp += self.pair[iquery, lig, :, pose] / len(iposes)
        return plp

    def log_posterior(self, poses):
        """
        Returns the log posterior for pose cluster.

        Parameters
        ----------
        poses: `dict[str, int]<dict>`
            {ligand_name: current pose number, }, currently selected poses

        Returns
        -------
        lp: float
            Log posterior for pose cluster.
        """
        iposes = [(self.ligands.index(lig), pose) for lig, pose in poses.items()]
        lp = 0
        for lig, pose in iposes:
            lp += self.single[lig, pose]

        for i, (lig1, pose1) in enumerate(iposes):
            for lig2, pose2 in iposes[i+1:]:
                lp += self.pair[lig1, lig2, pose1, pose2] / (len(poses)-1)
        return lp

    def pad(self, x, shape1, shape2=0, C=-np.inf):
        """
        Expand array to ("shape1",) if 1D or ("shape1", "shape2") if 2D
        and fill missing values with "C".

        Parameters
        ----------
        x: :class:`~numpy.ndarray`
            Array to pad.
        shape1: int
            First dimension of padded array.
        shape2: int, default=0
            Second dimension of padded array.
        C: float, default=-:class:`~numpy.inf`
            Value to fill missing values with.
        
        Returns
        -------
        y: :class:`~numpy.ndarray`
            Padded array.
        """
        if len(x.shape) == 1:
            y = np.zeros(shape1)+C
            y[:x.shape[0]] = x[:shape1]
        elif len(x.shape) == 2:
            y = np.zeros((shape1, shape2))+C
            y[:x.shape[0], :x.shape[1]] = x[:shape1, :shape2]
        else:
            assert False
        return y
