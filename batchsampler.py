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
    def __init__(self, **kwargs):

        default_params = {
            # Sample cliques according to loss
            'cliques': [],
            'cliqueSampleProb': np.empty(0),
            'simMatrix': np.empty(0),
            'cliqueSimMatrix': np.empty(0),
            'dataset': None,
            'category': None,
        }

        default_params.update(kwargs)
        for k in default_params.keys():
            self.__setattr__(k, default_params[k])
        for batch in kwargs['batches']:
            for clique in batch:
                assert clique.samples.shape[0] > 1, 'Adding a clique of a single sample'
                self.addClique(clique)
        self._check_params()

    def _check_params(self):
        """
        Param checker
        :return:
        """
        assert self.simMatrix is not None, "Similarity matrix is empty"

    def addClique(self, clique):
        """
        Add clique to the list of cliques from which to sample a batch
        :param clique:
        :return:
        """
        self.cliques.append(clique)
        self.cliqueSampleProb = np.append(self.cliqueSampleProb, 1)

    def updateCliqueSimMatrix(self):
        """
        Update the similarity matrix between cliques for fast heuristic sampling of cliques.
        :return:
        """
        print "Updating interclique similarities..."
        self.cliqueSimMatrix = np.empty((len(self.cliques), len(self.cliques)))
        for idx_a, clique_a in enumerate(self.cliques):
            print idx_a
            for idx_b in range(idx_a, len(self.cliques)):
                clique_b = self.cliques[idx_b]
                self.cliqueSimMatrix[idx_a, idx_b] = np.mean(self.simMatrix[:, clique_a.samples.reshape(1, -1)[0]]
                                                             [clique_b.samples.reshape(1, -1)[0]].reshape(1, -1)[0])
        self.cliqueSimMatrix = (self.cliqueSimMatrix + self.cliqueSimMatrix.T) / 2.0

    def updateCliqueSampleProb(self, lossperclique):
        """
        Update clique sampling probabilities
        :param lossperclique: Loss for each clique
        :return:
        """

        self.cliqueSampleProb = lossperclique / lossperclique.sum()

    def updateSimMatrix(self, updated_simMatrix):
        """
        Update the similarity matrix over samples for transitive clique update
        :param updated_simMatrix:
        :return:
        """
        assert updated_simMatrix.shape[0] > 0, "Empty similarity matrix"
        self.simMatrix = updated_simMatrix

    def sampleBatch(self, batch_size=128, max_cliques_per_batch=8, mode='heuristic'):
        """
        This function samples a Batch from all the cliques of a dataset holding triplet constraints if heuristic mode is
        selected, otherwise it selects a random cliques.
        :param batch_size: SGD mini-batch size
        :param max_cliques_per_batch: Maximum number of cliques per batch
        :param mode: Sampling mode either random (random clique selection) or heuristic (cliques are selected so that
        they do not represent the same semantic class while having different labels)
        :return: a list of clique objects (a batch of cliques)
        """
        # # Update simMatrix if its not the appropiate size
        # if len(self.cliques) != self.cliqueSimMatrix.shape[0]:
        #     self.updateCliqueSimMatrix()

        if mode == 'random':
            # Select random clique indices
            idxs = np.random.choice(len(self.cliques), int(max_cliques_per_batch), replace=False,
                                    p=self.cliqueSampleProb)
        elif mode == 'heuristic':
            # Select heuristic clique indices
            idxs = self.computeIndicesWithHeuristic(max_cliques_per_batch)

        # Balance the number of samples per clique in the batch (cliques may contained repeated samples, responsability
        # delegated to transformations to change them)
        if idxs.shape[0] > 1:
            batch = self.balanceSamplesPerClass(idxs, batch_size)
        else:
            batch = None

        return batch

    def balanceSamplesPerClass(self, idxs, batch_size):
        """
        This function balances the number of samples of each clique in a batch.
        :param idxs: Indices of cliques in a batch
        :param batch_size: SGD mini-batch size
        :return: list of copies of clique objects with number of samples balanced (a batch of cliques)
        """
        # The list of cliques to return
        batch = []
        samples_per_clique = int(np.floor(batch_size/len(idxs)))
        remainder = batch_size % len(idxs)
        for itt, i in enumerate(idxs):
            # If on last iteration add the remainder of samples
            if itt == len(idxs):
                samples_per_clique += remainder

            # Copy the clique not to modify the original clique for future usage
            clique_aux = copy.deepcopy(self.cliques[i])

            # If size of clique bigger than the number of samples per clique choose at random, otherwise replicate first
            # to choose at random afterwards
            if clique_aux.samples[0] > samples_per_clique:
                rand_idxs = np.random.choice(clique_aux.samples.shape[0], samples_per_clique).astype(dtype=np.int32)
                clique_aux.samples = clique_aux.samples[rand_idxs]
                clique_aux.isflipped = clique_aux.isflipped[rand_idxs]
                clique_aux.imnames = [clique_aux.imnames[i] for i in rand_idxs]
            else:
                factor = np.max([np.ceil(samples_per_clique/clique_aux.samples.shape[0]), 1.0])
                assert factor >= 1.0, "Factor is {}".format(str(factor))
                clique_aux.samples = np.tile(clique_aux.samples, factor)
                clique_aux.isflipped = np.tile(clique_aux.isflipped, factor)
                clique_aux.imnames = np.tile(np.asarray(clique_aux.imnames), factor).tolist()

                rand_idxs = np.random.choice(clique_aux.samples.shape[0], samples_per_clique).astype(dtype=np.int32)

                clique_aux.samples = clique_aux.samples[rand_idxs]
                clique_aux.isflipped = clique_aux.isflipped[rand_idxs]
                clique_aux.imnames = [clique_aux.imnames[i] for i in rand_idxs]

            # Append the clique to the batch
            batch.append(clique_aux)

        return batch

    def computeIndicesWithHeuristic(self, max_cliques_per_batch):
        """
        Heuristically select cliques which do not violate triplet constraints.
        :param max_cliques_per_batch: Maximum number of cliques per batch
        :return: Indices of cliques in a batch
        """
        # Sample first clique based on loss
        idxs = []
        seed_clique = np.random.choice(len(self.cliques), 1, p=self.cliqueSampleProb)
        idxs.append(seed_clique)

        # Start search cliques from the least similar
        search_order = self.cliqueSimMatrix[seed_clique].argsort()[0]
        for itt, clique_idx in enumerate(search_order):
            if len(idxs) == max_cliques_per_batch:
                break
            if self.tripletChecker(idxs, clique_idx):
                idxs.append(clique_idx)

        return np.asarray(idxs, dtype=np.int32)

    def tripletChecker(self, current, temptative):

        """
        Check that the min intraclique similarity is smaller than the max interclique sim.
        :param current: Current list of clique indices in the batch
        :param temptative: Prospective clique to append to the batch
        :return: Boolean indicator of triplet violation
        """
        for clique_idx in current:
            min_intraclique_sim = self.simMatrix[:, self.cliques[clique_idx].samples.reshape(1, -1)[0]][self.cliques[clique_idx].samples.reshape(1, -1)[0]].min()
            max_interclique_sim = self.simMatrix[:, self.cliques[clique_idx].samples.reshape(1, -1)[0]][self.cliques[temptative].samples.reshape(1, -1)[0]].max()


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
        f_ind = np.empty(0)
        y = np.empty(0)
        for clique in batch:
            x_idx = np.append(x_idx, clique.samples)
            f_ind = np.append(f_ind, clique.isflipped)
            y = np.append(y, np.tile(clique.label, clique.samples.shape[0]))
        return x_idx, f_ind, y

    def transitiveCliqueComputation(self):
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
            clique.AvailableIndices = np.asarray([True] * self.simMatrix.shape[0])
            for sample in clique.samples:
                self.updateAvailableIndices(clique, sample, temporalWindow=20)

            avg_sims_to_clique = self.simMatrix[clique.samples, :].mean(axis=0)[0]
            mask = np.ones(self.simMatrix.shape[1], dtype=np.bool)
            mask[clique.samples] = False
            avg_sims_to_clique = avg_sims_to_clique[mask]
            assert len(avg_sims_to_clique) + len(clique.samples) == self.simMatrix.shape[1]

            random_sampling = np.random.choice(avg_sims_to_clique, points_to_sample_null,
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
            #     avg_sim_aux = self.simMatrix[sample_id, clique_samples_aux].mean()
            #
            #     # If this sim could come from random points then remove sample from clique
            #     if (1.0 - cdf(avg_sim_aux)) > threshold_pval*10:
            #         idxs_to_remove.append(idx_sample)
            # clique.removeSample(np.asarray(idxs_to_remove, dtype=np.int32))

            idxs_points = np.where(pval_clique < threshold_pval)[0]
            for idx in idxs_points:
                assert idx not in clique.samples
                if not clique.AvailableIndices[idx]:
                    continue
                else:
                    f = self.calculateFlip(clique, idx)
                    clique.addSample(idx, f, self.imagePath[idx])
                    self.updateAvailableIndices(clique, idx, temporalWindow=20)

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

    def updateAvailableIndices(self, clique, sample, temporalWindow=0):
        """
        Update the indicator vector of available samples to include in clique. Constraining so that a clique does not
        include two samples of the same sequence
        :param avail: current indicator vector
        :param sample:
        :return:
        """
        # If there is no sequence structure update is not done
        if self.seqNames is None:
            return

        if temporalWindow:
            clique.availableIndices[sample - temporalWindow: sample + temporalWindow] = False
        else:
            # No temporal window, take the whole sequence
            ind_not_seq = np.asarray(self.seqNames) != np.asarray([self.seqNames[sample]] * self.simMatrix.shape[0])
            clique.availableIndices = ~((~np.asarray(clique.availableIndices)) | (~np.asarray(ind_not_seq)))

    def calculateFlip(self, clique, new_sample):
        return int(np.mean(self.flipMatrix[clique.samples, new_sample]) >= 0.5)

    def visualizelistindices(self, indices):

        key = 'y'
        counter = 0
        while key == 'y':

            idx = indices[counter % indices.shape[0]]
            im = pylab.imread(self.pathToFolder + self.imagePath[idx][1:-1])
            pylab.imshow(im)
            pylab.show()
            counter += 1
            # key = raw_input('Continue?')

