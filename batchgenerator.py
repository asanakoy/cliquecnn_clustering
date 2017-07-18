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
from clique import Clique
from tqdm import tqdm
import time


class BatchGenerator(object):

    """
    This class encapsulates the logic and operations to generate batches of cliques to train a CNN.
    """

    def __init__(self, **kwargs):
        """
        """
        default_params = {
            'sim_matrix': None,
            'flipvals': None,
            'seq_names': None,
            'crops_dir': None,
            'relative_image_pathes': None,
            'num_cliques_per_initial_batch': None,
            'num_samples_per_clique': None,
            'anchors': None,
            'diff_prob': 1.0,  # set to 1.0 to disable this feature, while not correctly(?) implemented
            'seed': None
        }

        for key in kwargs:
            if key not in default_params:
                raise ValueError('Unexpected parameter: {}'.format(key))

        default_params.update(kwargs)
        for k in default_params.keys():
            self.__setattr__(k, default_params[k])
        self.random_state = np.random.RandomState(self.seed)
        print 'BatchGenerator::random seed = {}'.format(self.seed)
        self.sample_freq = np.ones(self.sim_matrix.shape[0]) * 0.00001
        self._curr_label = 0

        self._check_params()

    def _check_params(self):
        """
        Param checker
        :return:
        """
        assert self.sim_matrix is not None, "Similarity matrix is empty"
        assert self.num_samples_per_clique is not None

    def generate_batches(self, num_initial_batches=100):
        """
        This function loops over batches and calls the batch computation
        :return:
        """
        batches = []
        # Check anchors for datasets which do not have anchors to evaluate
        if self.anchors is not None:
            assert num_initial_batches >= len(self.anchors['anchor']), "Number of batches must be larger than the number of anchors"

        for batch_id in tqdm(range(num_initial_batches)):
            # If there are no anchors use random seeds
            if self.anchors is not None:
                # FIXME: < len(self.anchors['anchor']) ?
                if batch_id <= len(self.anchors):
                    seed_sample = int(self.anchors['anchor'][batch_id][0])
                else:
                    seed_sample = self.random_state.randint(self.sim_matrix.shape[0])
            else:
                seed_sample = self.random_state.randint(self.sim_matrix.shape[0])
            batches.append(self.compute_batch(seed_sample))
        return batches

    def compute_batch(self, seed):
        """
        Compute a batch of cliques performing temporal augmentation in each clique
        :param seed: Seed used to compute the first clique
        :return: a list of cliques (a batch)
        """
        nClique = 0
        batch = []
        constraints = 'nClique < self.num_cliques_per_initial_batch'
        while eval(constraints):
            clique = self.compute_clique(seed)
            assert len(clique.samples) == len(clique.isflipped), "Samples and flips have different size"
            self.clique_purification(clique)
            # Probably better do temporal augmentation after transitive growing
            self.temporal_augmentation(clique)

            # Set label of clique and add to batch if it has more than 1 sample
            if len(clique.samples) > 1:
                clique.label = self._curr_label
                self._curr_label += 1

                # Update frequency table
                self.update_freq_table(clique)
                batch.append(clique)

            seed = self.find_dissimilar_seed(batch)
            nClique += 1

        # batch = self.transitive_clique_computation(batch)
        return batch

    def compute_clique(self, seed):
        """
        This function stores the logic of clique computation. It adds the point p that maximizes the average
        similarity to the current clique and does not belong to an already used sequence
        :param seed:
        :return: clique
        """
        # Initialize cliques and update available indices
        clique = Clique(self.crops_dir, self.sim_matrix.shape[0])
        clique.add_sample(seed, self.flipvals[seed, seed], self.relative_image_pathes[seed])
        self.update_available_indices(clique, seed)

        idx_to_add = 0
        # Add constraint for checking the freq of samples in cliques
        frq = np.max([(self.sample_freq / self.sample_freq.sum()) - self.diff_prob,
                      np.zeros(self.sample_freq.shape)], axis=0)
        idx_avail = np.where(clique.available_indices)[0]

        while (len(clique.samples) < self.num_samples_per_clique) and (idx_to_add < idx_avail.shape[0]):
            idx_avail = np.where(clique.available_indices)[0]
            # FIXME: here we choose the point that has max similarity to some point in clique. We can get a line structure instead of a compact cluster.
            search_order = np.max(self.sim_matrix[clique.samples.reshape(-1, 1), idx_avail], axis=0).argsort()[::-1]
            # FIXME: BUG? idx_to_add must be always 0. Because on each step idx_avail will be changed.
            # FIXME: So now we don't get the best cliques
            p = idx_avail[search_order[idx_to_add]]
            if p in clique.samples:
                pass
            # Add constraint for freq samples if random from normal distribution is higher than the freq with which
            # sample p has been sampled then add it
            elif frq[p] < self.random_state.rand():
                f = self.calculate_flip(clique, p)
                clique.add_sample(p, f, self.relative_image_pathes[p])
                self.update_available_indices(clique, p)
            idx_to_add += 1
        return clique

    def update_available_indices(self, clique, sample, temporalWindow=0):
        """
        Update the indicator vector of available samples to include in clique. Constraining so that a clique does not
        include two samples of the same sequence
        :param avail: current indicator vector
        :param sample:
        :return:
        """
        # If there is no sequence structure update is not done
        if self.seq_names is None:
            return

        if temporalWindow:
            clique.available_indices[sample - temporalWindow: sample + temporalWindow] = False
        else:
            # No temporal window, take the whole sequence
            ind_not_seq = np.asarray(self.seq_names) != np.asarray([self.seq_names[sample]] * self.sim_matrix.shape[0])
            clique.available_indices = ~((~np.asarray(clique.available_indices)) | (~np.asarray(ind_not_seq)))

    def clique_purification(self, clique):
        """
        Purify a clique by replacing an assignment with a non-assigned  point that maximizes average similarity to clique
        :param clique:
        :return:
        """
        # For each point in the clique, compute its avg similarity to the clique (substracting similarity to itself)
        avgSelfSim = np.zeros(len(clique.samples))
        init_samples = np.array(clique.samples)

        # Add constraint for freq samples
        frq = np.max([(self.sample_freq / self.sample_freq.sum()) - self.diff_prob, np.zeros(self.sample_freq.shape)],
                     axis=0)

        for idx, sample in enumerate(clique.samples):
            aux_samples = init_samples
            aux_samples = np.delete(aux_samples, idx)
            assert len(self.sim_matrix[aux_samples.reshape(-1, 1), sample]) == len(aux_samples)
            avgSelfSim[idx] = np.max(self.sim_matrix[aux_samples.reshape(-1, 1), sample], axis=0)

        sorted_self_sim = avgSelfSim.argsort()
        topk = int(sorted_self_sim.shape[0] * 1)
        for idx in sorted_self_sim[:topk]:

            # Sort points with respect to the avg similarity to the current clique minus the temptative point to extract
            aux_samples = init_samples
            aux_samples = np.delete(aux_samples, idx)

            # FIXME: here we choose the point that has max similarity to some point in clique. We can get a line structure instead of a compact cluster.
            avgSim = np.max(self.sim_matrix[:, aux_samples], axis=1)
            nearest_samples = avgSim.reshape(1, -1).argsort()[0][::-1]
            for i in nearest_samples:

                # Filter points that are already assigned to the clique or are not available
                if i in clique.samples or not clique.available_indices[i]:
                    continue
                # If the non assigned point i has better avg similarity to the clique than "sample", replace assignment
                # and jump to next point
                # Add constraint for freq samples if random from normal distribution is higher than the freq with which
                # sample p has been sampled then add it
                if avgSim[i] > avgSelfSim[idx] and frq[i] < self.random_state.rand():
                    self.replace_assigment(clique, idx, i)
                    break

    def find_dissimilar_seed(self, batch):
        """
        Find the seed that minimizes the min-similarity (top 5% least similar) with the current batch
        :param batch: Current batch of cliques
        :return:
        """
        # TODO: Find seed which is at least more dissimilar that the most dissimilar points in a clique

        topPercent = 0.05
        allIdxs = np.empty(0, dtype=np.int32)
        for clique in batch:
            allIdxs = np.append(allIdxs, clique.samples)
        assert allIdxs.ndim == 1
        # TODO: recheck this min finding here. Here we minimize min distance to all the points in already existing cliques.
        # May be it's better to minimize the max?
        sortedIdxs = np.min(self.sim_matrix[allIdxs], axis=0).argsort()
        seed = self.random_state.choice(sortedIdxs[:int(np.ceil(self.sim_matrix.shape[0] * topPercent))])
        return seed

    def temporal_augmentation(self, clique):
        """
        Do temporal augmentation by adding to the clique the immediate previous and next frame for each sample. Check
        cases where frame is either the first or last frame of a sequence
        :param clique:
        :return:
        """
        # If there is no sequence structure temporal augmentation is not done
        if self.seq_names is None:
            return
        for frame_id in clique.samples:
            idxsSeq = np.where((np.asarray(self.seq_names) == np.tile(np.asarray(self.seq_names[frame_id]),
                                                                     self.sim_matrix.shape[0])) == True)[0]
            if frame_id == idxsSeq[0]:
                clique.add_sample(frame_id + 1, int(clique.isflipped[-1]), self.relative_image_pathes[frame_id + 1])
            else:
                clique.add_sample(frame_id - 1, int(clique.isflipped[-1]), self.relative_image_pathes[frame_id - 1])

    def calculate_flip(self, clique, new_sample):
        return int(np.mean(self.flipvals[clique.samples.reshape(-1, 1), new_sample]) >= 0.5)

    def replace_assigment(self, clique, cliqueidxtoremove, generalIdxtoAdd):

        # Remove old sample and update params
        if self.seq_names is not None:
            idxsSeq = np.where((np.asarray(self.seq_names) ==
                                np.asarray([self.seq_names[clique.samples[cliqueidxtoremove]]] * self.sim_matrix.shape[0])) == True)[0]
            clique.available_indices[idxsSeq] = True
        clique.remove_sample(cliqueidxtoremove)

        #  Add sample and update params
        f = self.calculate_flip(clique, generalIdxtoAdd)
        clique.add_sample(generalIdxtoAdd, f, self.relative_image_pathes[generalIdxtoAdd])
        self.update_available_indices(clique, generalIdxtoAdd)

    def update_freq_table(self, clique):
        """
        This function updates the frequency table with the newly computed clique
        :param clique:
        :return:
        """

        self.sample_freq[clique.samples] += 1

