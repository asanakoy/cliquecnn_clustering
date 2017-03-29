# --
# Copyright (c) 2016 Miguel Bautista, Artsiom Sanakoyeu
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
from clipclique import ClipClique
from utils import get_clips_similarity
from tqdm import tqdm
import time
from tqdm import tqdm


class ClipBatchGenerator(object):

    """
    This class encapsulates the logic and operations to generate batches of cliques to train a CNN.
    """

    def __init__(self,
                 clip_len=None,
                 sim_matrix=None,
                 seq_names=None,
                 crops_dir=None,
                 relative_image_pathes=None,
                 num_cliques_per_initial_batch=None,
                 num_samples_per_clique=None,
                 anchors=None,
                 seed=None
                 ):

        self.clip_len = clip_len
        self.sim_matrix = sim_matrix
        self.seq_names = np.asarray(seq_names)
        self.crops_dir = crops_dir
        self.relative_image_pathes = np.asarray(relative_image_pathes)
        self.num_cliques_per_initial_batch = num_cliques_per_initial_batch
        self.num_samples_per_clique = num_samples_per_clique
        self.anchors = anchors
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        print 'BatchGenerator::random seed = {}'.format(self.seed)
        self._curr_label = 0

        self._check_params()

    def _check_params(self):
        """
        Param checker
        :return:
        """
        assert self.sim_matrix is not None, "Similarity matrix is empty"
        assert self.sim_matrix.ndim == 3, 'Sim matrix must be full! Non-flipped and flipped concatenated by axis=2'
        assert self.num_samples_per_clique is not None
        assert self.clip_len is not None

    def generate_batches(self, num_initial_batches=100):
        """
        This function loops over batches and calls the batch computation
        :return:
        """
        batches = []
        anchor_id = 0
        progress_bar = tqdm(total=num_initial_batches)
        while len(batches) < num_initial_batches:
            # If there are no anchors use random seeds
            if self.anchors is not None and anchor_id < len(self.anchors):
                seed_sample = int(self.anchors['anchor'][anchor_id][0])
                anchor_id += 1
            else:
                seed_sample = self.random_state.randint(self.sim_matrix.shape[0])
            if seed_sample - self.clip_len + 1 < 0 or self.seq_names[seed_sample] != self.seq_names[seed_sample - self.clip_len + 1]:
                continue
            seed_clip_frames = range(seed_sample - self.clip_len + 1, seed_sample + 1)

            batches.append(self.compute_batch(seed_clip_frames))
            progress_bar.update(len(batches))
        progress_bar.close()
        return batches

    def compute_batch(self, seed_clip_frames):
        """
        Compute a batch of cliques performing temporal augmentation in each clique
        :param seed: Seed used to compute the first clique
        :return: a list of cliques (a batch)
        """
        if self.seq_names[seed_clip_frames[0]] != self.seq_names[seed_clip_frames[-1]]:
            raise ValueError('Clip must be from the same sequence!')

        i_clique = 0
        batch = []
        while i_clique < self.num_cliques_per_initial_batch:
            clique = self.compute_clique(seed_clip_frames)
            assert len(clique.samples) == len(clique.isflipped), "Samples and flips have different size"
            # self.clique_purification(clique)

            # Set label of clique and add to batch if it has more than 1 sample
            if len(clique.samples) > 1:
                clique.label = self._curr_label
                self._curr_label += 1
                batch.append(clique)

                seed_clip_frames = self.find_dissimilar_clip(batch)
            i_clique += 1

        # batch = self.transitive_clique_computation(batch)
        return batch

    def compute_clique(self, seed_clip_frames):
        """
        This function stores the logic of clique computation. It adds the point p that maximizes the average
        similarity to the current clique and does not belong to an already used sequence
        :param seed:
        :return: clique
        """
        # Initialize cliques and update available indices
        clique = ClipClique(self.clip_len, self.crops_dir, self.sim_matrix.shape[0])
        seq_image_pathes = self.relative_image_pathes[seed_clip_frames]
        clique.add_seq(seed_clip_frames, False, seq_image_pathes)
        clique.update_available_indices(seed_clip_frames, self.seq_names)

        while len(clique.samples) < self.num_samples_per_clique:
            idx_avail = np.where(clique.available_indices)[0]

            clips = []
            for i in idx_avail:
                if self.seq_names[i] == self.seq_names[i - self.clip_len + 1]:
                    clip_frames = range(i - self.clip_len + 1, i + 1)
                    sims = [np.min(get_clips_similarity(self.sim_matrix, clique.samples, clip_frames, flipval=0)),
                            np.min(get_clips_similarity(self.sim_matrix, clique.samples, clip_frames, flipval=1))]
                    flipval = bool(np.argmax(sims))
                    sim = np.max(sims)
                    clips.append((sim, clip_frames, flipval))
            if len(clips) == 0:
                break
            clips.sort(reverse=True)

            clip_frames = clips[0][1]
            flipval = clips[0][2]
            clique.add_seq(clip_frames, flipval, self.relative_image_pathes[clip_frames])
            clique.update_available_indices(clip_frames, self.seq_names)
        return clique

    def find_dissimilar_clip(self, batch):
        """
        Find the seed that minimizes the min-similarity (top 5% least similar) with the current batch
        :param batch: Current batch of cliques
        :return:
        """
        top_percent = 0.05
        num_clips_to_choose_from = int(np.ceil(self.sim_matrix.shape[0] * top_percent))
        all_clips = list()
        for clique in batch:
            all_clips.extend(clique.samples)
        all_clips = np.vstack(all_clips)
        assert all_clips.ndim == 2 and all_clips.shape[1] == batch[0].samples.shape[1]

        avalilable_clips = []
        for i in xrange(self.clip_len - 1, len(self.seq_names)):
            if self.seq_names[i] == self.seq_names[i - self.clip_len + 1]:
                clip_frames = range(i - self.clip_len + 1, i + 1)
                max_sim = np.max(np.mean(self.sim_matrix[all_clips, clip_frames, 0], axis=1))
                avalilable_clips.append((max_sim, clip_frames))

        sorted_idxs = sorted(range(len(avalilable_clips)), key=avalilable_clips.__getitem__)[:num_clips_to_choose_from]
        chosen_seq_idx = self.random_state.choice(sorted_idxs)
        return avalilable_clips[chosen_seq_idx][1]

'''
    def clique_purification(self, clique):
        """
        Purify a clique by replacing an assignment with a non-assigned  point that maximizes average similarity to clique
        :param clique:
        :return:
        """
        # TODO: remake it for clips
        # For each point in the clique, compute its avg similarity to the clique (substracting similarity to itself)
        avgSelfSim = np.zeros(len(clique.samples))
        init_samples = np.array(clique.samples)

        for idx, sample in enumerate(clique.samples):
            aux_samples = init_samples
            aux_samples = np.delete(aux_samples, idx)
            assert len(self.sim_matrix[aux_samples.reshape(-1, 1), sample]) == len(aux_samples)
            avgSelfSim[idx] = np.max(self.sim_matrix[aux_samples.reshape(-1, 1), sample],
                                     axis=0)

        sorted_self_sim = avgSelfSim.argsort()
        topk = int(sorted_self_sim.shape[0] * 1)
        for idx in sorted_self_sim[:topk]:

            # Sort points with respect to the avg similarity to the current clique minus the temptative point to extract
            aux_samples = init_samples
            aux_samples = np.delete(aux_samples, idx)
            # FIXME:
            avgSim = np.max(self.sim_matrix[:, aux_samples], axis=1)
            nearest_samples = avgSim.reshape(1, -1).argsort()[0][::-1]
            for i in nearest_samples:

                # Filter points that are already assigned to the clique or are not available
                if i in clique.samples or not clique.available_indices[i]:
                    continue
                # If the non assigned point i has better avg similarity to the clique than "sample", replace assignment
                # and jump to next point
                if avgSim[i] > avgSelfSim[idx]:
                    self.replace_assigment(clique, idx, i)
                    break

    def replace_assigment(self, clique, cliqueidxtoremove, generalIdxtoAdd):

        # Remove old sample and update params
        if self.seq_names is not None:
            idxsSeq = np.where((np.asarray(self.seq_names) ==
                                np.asarray([self.seq_names[clique.samples[cliqueidxtoremove]]] * self.sim_matrix.shape[0])) == True)[0]
            clique.available_indices[idxsSeq] = True
        clique.remove_seq(cliqueidxtoremove)

        #  Add sample and update params
        f = self.calculate_flip(clique, generalIdxtoAdd)
        clique.add_sample(generalIdxtoAdd, f, self.relative_image_pathes[generalIdxtoAdd])
        self.update_available_indices(clique, generalIdxtoAdd)
'''