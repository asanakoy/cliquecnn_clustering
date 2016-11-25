# --
# Copyright (c) 2016 Miguel Bautista
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ++
#
# 24.11.2016 Fixed, refactored: Artsiom Sanakoyeu


import numpy as np
import matplotlib.pylab as pylab
from eval.image_getter import ImageGetterFromMat
import scipy.stats as stats
import copy


class BatchSampler(object):
    """
    This class encapsulates the data and logic to sample batches from a set of cliques. A batch is a list of cliques
    used to compute the gradient of the ConvNet.
    """
    def __init__(self,
                 batches,
                 cliques=None,
                 clique_sample_prob=None,
                 sim_matrix=None,
                 flipvals=None,
                 seq_names=None,
                 crops_dir=None,
                 relative_image_pathes=None,
                 clique_sim_matrix=None,
                 dataset=None,
                 category=None,
                 seed=None):
        if cliques is None:
            cliques = list()
        if clique_sample_prob is None:
            clique_sample_prob = np.empty(0)
        args = locals()
        for k, v in args.iteritems():
            self.__setattr__(k, v)
        self._check_params()

        self.random_state = np.random.RandomState(self.seed)
        print 'BatchSampler::random seed = {}'.format(self.seed)

        for batch in batches:
            for clique in batch:
                assert len(clique.samples) > 1, 'Adding a clique of a single sample'
                self.add_clique(clique)

    def _check_params(self):
        """
        Param checker
        :return:
        """
        assert len(self.cliques) == len(self.clique_sample_prob)
        assert self.sim_matrix is not None, "Similarity matrix is empty"

    def add_clique(self, clique):
        """
        Add clique to the list of cliques from which to sample a batch
        :param clique:
        :return:
        """
        self.cliques.append(clique)
        self.clique_sample_prob = np.append(self.clique_sample_prob, 1)

    def update_clique_sim_matrix(self):
        """
        Update the similarity matrix between cliques for fast heuristic sampling of cliques.
        :return:
        """
        print "Updating interclique similarities..."
        self.clique_sim_matrix = np.empty((len(self.cliques), len(self.cliques)))
        for idx_a, clique_a in enumerate(self.cliques):
            print idx_a
            for idx_b in xrange(idx_a, len(self.cliques)):
                clique_b = self.cliques[idx_b]
                # TODO: FIXME: may have bug? Recheck later
                self.clique_sim_matrix[idx_a, idx_b] = np.mean(self.sim_matrix[:, clique_a.samples.reshape(1, -1)[0]][clique_b.samples.reshape(1, -1)[0]].reshape(1, -1)[0])
        self.clique_sim_matrix = (self.clique_sim_matrix + self.clique_sim_matrix.T) / 2.0

    def set_clique_sample_prob(self, lossperclique):
        """
        Update clique sampling probabilities
        :param lossperclique: Loss for each clique
        :return:
        """
        self.clique_sample_prob = lossperclique

    def set_sim_matrix(self, updated_simMatrix, flipvals):
        """
        Update the similarity matrix over samples for transitive clique update
        :param updated_simMatrix:
        :return:
        """
        if flipvals is None:
            flipvals = self.flipvals
        assert flipvals is not None
        assert updated_simMatrix.shape[0] > 0, "Empty similarity matrix"
        assert len(flipvals) > 0, "Empty similarity matrix"
        assert flipvals.shape == updated_simMatrix.shape
        self.sim_matrix = updated_simMatrix
        self.flipvals = flipvals

    def sample_batch(self, batch_size=128, max_cliques_per_batch=8, mode='heuristic'):
        """
        This function samples a Batch from all the cliques of a dataset holding triplet constraints if heuristic mode is
        selected, otherwise it selects a random cliques.
        :param batch_size: SGD mini-batch size
        :param max_cliques_per_batch: Maximum number of cliques per batch
        :param mode: Sampling mode either random (random clique selection) or heuristic (cliques are selected so that
        they do not represent the same semantic class while having different labels)
        :return: a list of clique objects (a batch of cliques)
        """
        # # Update sim_matrix if its not the appropiate size
        # if len(self.cliques) != self.clique_sim_matrix.shape[0]:
        #     self.update_clique_sim_matrix()

        if mode == 'random':
            # Select random clique indices
            idxs = self.random_state.choice(len(self.cliques),
                                            size=int(max_cliques_per_batch),
                                            replace=False,
                                            p=self.clique_sample_prob / self.clique_sample_prob.sum())
        elif mode == 'heuristic':
            # Select heuristic clique indices
            idxs = self.compute_indices_with_heuristic(max_cliques_per_batch)
        assert len(idxs) == max_cliques_per_batch, 'Got num cliques({}) != max_cliques_per_batch'.format(len(idxs))
        # Balance the number of samples per clique in the batch (cliques may contained repeated samples, responsability
        # delegated to transformations to change them)
        if idxs.shape[0] > 1:
            batch = self.balance_samples_per_class(idxs, batch_size)
        else:
            batch = None
        return batch

    def balance_samples_per_class(self, idxs, batch_size):
        """
        This function balances the number of samples of each clique in a batch.
        :param idxs: Indices of cliques in a batch
        :param batch_size: SGD mini-batch size
        :return: list of copies of clique objects with number of samples balanced (a batch of cliques)
        """
        # The list of cliques to return
        batch = []
        samples_per_clique = int(np.floor(batch_size / len(idxs)))
        remainder = batch_size % len(idxs)
        assert remainder == 0, 'We haven\'t fixed the bug yet. So remainder > 0 is not allowed for now.'
        for itt, i in enumerate(idxs):
            # If on last iteration add the remainder of samples
            if itt == len(idxs):
                # FIXME: fix the bug here. This code is never called
                samples_per_clique += remainder

            # Copy the clique not to modify the original clique for future usage
            clique_aux = copy.deepcopy(self.cliques[i])

            # If size of clique bigger than the number of samples per clique choose at random, otherwise replicate first
            # to choose at random afterwards
            if len(clique_aux.samples) > samples_per_clique:
                rand_idxs = self.random_state.choice(len(clique_aux.samples),
                                                     size=samples_per_clique,
                                                     replace=False).astype(dtype=np.int32)
                clique_aux.samples = clique_aux.samples[rand_idxs]
                clique_aux.isflipped = clique_aux.isflipped[rand_idxs]
                clique_aux.imnames = [clique_aux.imnames[i] for i in rand_idxs]
            elif len(clique_aux.samples) < samples_per_clique:
                chosen_indices = range(len(clique_aux.samples))
                clique_aux.samples = clique_aux.samples
                factor = max(int(np.ceil(samples_per_clique / float(len(clique_aux.samples)))), 2)
                assert factor >= 2, "Factor is {} < 2".format(factor)
                clique_aux.samples = np.tile(clique_aux.samples, factor)
                clique_aux.isflipped = np.tile(clique_aux.isflipped, factor)
                clique_aux.imnames = np.tile(np.asarray(clique_aux.imnames), factor).tolist()

                rand_idxs = self.random_state.choice(range(len(chosen_indices), len(clique_aux.samples)),
                                                     size=samples_per_clique - len(chosen_indices),
                                                     replace=False).astype(dtype=np.int32)
                chosen_indices += rand_idxs.tolist()

                clique_aux.samples = clique_aux.samples[chosen_indices]
                clique_aux.isflipped = clique_aux.isflipped[chosen_indices]
                clique_aux.imnames = [clique_aux.imnames[i] for i in chosen_indices]
            else:
                # len(clique_aux.samples) == samples_per_clique
                pass
            # Append the clique to the batch
            assert len(clique_aux.samples) == len(clique_aux.isflipped) == len(clique_aux.imnames), 'Corrupted sizes in balancing'
            batch.append(clique_aux)

        return batch

    def compute_indices_with_heuristic(self, max_cliques_per_batch):
        """
        Heuristically select cliques which do not violate triplet constraints.
        :param max_cliques_per_batch: Maximum number of cliques per batch
        :return: Indices of cliques in a batch
        """
        if self.clique_sim_matrix is None:
            raise ValueError('clique_sim_matrix must be not None')
        # Sample first clique based on loss
        idxs = []
        seed_clique = self.random_state.choice(len(self.cliques),
                                               size=1, replace=False,
                                               p=self.clique_sample_prob / self.clique_sample_prob.sum())
        idxs.append(seed_clique)

        # Start search cliques from the least similar
        search_order = self.clique_sim_matrix[seed_clique].argsort()[0]
        for itt, clique_idx in enumerate(search_order):
            if len(idxs) == max_cliques_per_batch:
                break
            if self.triplet_checker(idxs, clique_idx):
                idxs.append(clique_idx)

        return np.asarray(idxs, dtype=np.int32)

    def triplet_checker(self, current, temptative):

        """
        Check that the min intraclique similarity is smaller than the max interclique sim.
        :param current: Current list of clique indices in the batch
        :param temptative: Prospective clique to append to the batch
        :return: Boolean indicator of triplet violation
        """
        # TODO: FIXME: single sample??
        for clique_idx in current:
            min_intraclique_sim = self.sim_matrix[:, self.cliques[clique_idx].samples.reshape(1, -1)[0]][self.cliques[clique_idx].samples.reshape(1, -1)[0]].min()
            max_interclique_sim = self.sim_matrix[:, self.cliques[clique_idx].samples.reshape(1, -1)[0]][self.cliques[temptative].samples.reshape(1, -1)[0]].max()

            if max_interclique_sim > min_intraclique_sim:
                return False
        return True

    def parse_to_list(self, batch):
        """
        Parse batch to list format for batch loader.
        :param batch: Batch to parse
        :return: List of image indices, flipping indicators and labels
        """
        x_idx = np.empty(0)
        flipvals = np.empty(0)
        labels = np.empty(0)
        for clique in batch:
            x_idx = np.append(x_idx, clique.samples)
            flipvals = np.append(flipvals, clique.isflipped)
            labels = np.append(labels, np.tile(clique.label, len(clique.samples)))
            assert x_idx.shape == flipvals.shape == labels.shape, "Corrupted size of clique while parsing"

        assert x_idx.shape == flipvals.shape == labels.shape, "Corrupted size of clique while parsing"
        return x_idx, flipvals, labels

    def transitive_clique_computation(self):
        """
        Filter cliques using transitivity constraints.

        :param clique: clique to be filtered
        :return: a filtered clique (which can vary in size)
        """

        points_to_sample_null = 500
        threshold_pval = 0.001
        # For each clique fit distribution t
        for idx_clique, clique in enumerate(self.cliques):
            print "Growing clique {}/{}".format(idx_clique, len(self.cliques))

            # Clear and reset Available indices with temporal windows
            clique.AvailableIndices = np.ones(self.sim_matrix.shape[0], dtype=np.bool)
            for sample in clique.samples:
                self.__update_available_indices(clique, sample, temporal_window=20)

            avg_sims_to_clique = self.sim_matrix[clique.samples, :].mean(axis=0)
            mask = np.ones(self.sim_matrix.shape[1], dtype=np.bool)
            mask[clique.samples] = False
            idxs_true_mask = np.where(mask)[0]
            avg_sims_to_clique = avg_sims_to_clique[mask]
            assert len(avg_sims_to_clique) + len(clique.samples) == self.sim_matrix.shape[1]

            random_sampling = self.random_state.choice(avg_sims_to_clique,
                                                       size=points_to_sample_null,
                                                       replace=False)
            clique_dist = self.fit_distr(random_sampling)
            cdf = lambda sample1d: stats.t.cdf(sample1d, *clique_dist['other_args'],
                                                                loc=clique_dist['loc'],
                                                                scale=clique_dist['scale'])

            pval_clique = 1.0 - cdf(avg_sims_to_clique)

            # Filter our samples in cliques if their avg_distance to clique is similar to what random points would have
            # idxs_to_remove = []
            # for idx_sample, sample_id in enumerate(clique.samples):
            #
            #     # Substract self similarity
            #     clique_samples_aux = np.setdiff1d(clique.samples, sample_id)
            #     avg_sim_aux = self.sim_matrix[sample_id, clique_samples_aux].mean()
            #
            #     # If this sim could come from random points then remove sample from clique
            #     if (1.0 - cdf(avg_sim_aux)) > threshold_pval*10:
            #         idxs_to_remove.append(idx_sample)
            # clique.remove_sample(np.asarray(idxs_to_remove, dtype=np.int32))

            idxs_points = np.where(pval_clique < threshold_pval)[0]
            # Double indexing points from true mask
            for idx in idxs_true_mask[idxs_points]:
                if not clique.AvailableIndices[idx]:
                    continue
                else:
                    f = self.calculate_flip(clique, idx)
                    clique.add_sample(idx, f, self.relative_image_pathes[idx])
                    self.__update_available_indices(clique, idx, temporal_window=20)

    def fit_distr(self, features):
        """
        Fit t distribution to a set of 1 dimensional features and return the params of the distribution
        :return:
        """
        # assert len(features.shape) == 1, "Features are not 1 dimensional"

        dist = getattr(stats, 't')
        param = dist.fit(features)
        other_args = param[:-2]
        loc = param[-2]
        scale = param[-1]
        dist_params = {'other_args': other_args, 'loc': loc, 'scale': scale}

        return dist_params

    def __update_available_indices(self, clique, sample, temporal_window=0):
        """
        Update the indicator vector of available samples to include in clique. Constraining so that a clique does not
        include two samples of the same sequence
        :param avail: current indicator vector
        :param sample:
        :return:
        """
        # If there is no sequence structure update is not done
        if self.seq_names is None:
            print 'BatchSampler::WARNING! seq_names is None'
            return

        if temporal_window:
            clique.availableIndices[sample - temporal_window: sample + temporal_window] = False
        else:
            # No temporal window, take the whole sequence
            ind_not_seq = np.asarray(self.seq_names) != np.asarray([self.seq_names[sample]] * self.sim_matrix.shape[0])
            clique.availableIndices = np.logical_and(np.asarray(clique.availableIndices), ind_not_seq)

    def calculate_flip(self, clique, new_sample):
        return int(np.mean(self.flipvals[clique.samples.reshape(-1, 1), new_sample]) >= 0.5)

    def visualize_list_indices(self, indices):

        key = 'y'
        counter = 0
        while key == 'y':

            idx = indices[counter % indices.shape[0]]
            im = pylab.imread(self.crops_dir + self.relative_image_pathes[idx][1:-1])
            pylab.imshow(im)
            pylab.show()
            counter += 1
            # key = raw_input('Continue?')

