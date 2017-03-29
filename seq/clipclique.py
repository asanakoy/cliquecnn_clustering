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
import matplotlib.pyplot as plt
from PIL import Image


class ClipClique(object):

    def __init__(self, clip_len, crops_dir, nsamples):
        """
        Class for encapsulating and visualizing cliques
        """
        self.clip_len = clip_len
        self.samples = np.empty((0, clip_len), dtype=np.int32)
        self.isflipped = np.empty(0, dtype=np.bool)
        self.available_indices = np.ones(nsamples, dtype=np.bool)
        self.imnames = []
        self.label = None
        self.crops_dir = crops_dir

    def add_seq(self, seq_samples, flipval, image_names):
        """
        Add a sample to a clique
        Args:
          seq_samples: sequence to add
          flipval: Flip indicator
          image_names: names of the images
        """
        assert not np.isscalar(seq_samples), 'sample {} is a scalar'.format(seq_samples)
        assert np.isscalar(flipval), 'flipval {} is not a scalar'.format(flipval)
        for seq in self.samples:
            if np.array_equal(seq, seq_samples):
                raise ValueError('Seq {} is already in clique {}'.format(seq_samples, self.samples))
        self.samples = np.vstack([self.samples, seq_samples])
        self.isflipped = np.append(self.isflipped, flipval)
        self.imnames.append(image_names)

    def remove_seq(self, idx):
        """
        Remove a sequence from a clique given an index
        """
        self.samples = np.delete(self.samples, idx, axis=0)
        self.isflipped = np.delete(self.isflipped, idx)
        del self.imnames[idx]

    def visualize(self):
        """
        Visualize clique
        """
        plt.figure(figsize=(20, 20))
        plt.cla()
        imnames = list()
        i = 1
        for seq, flipval in self.samples, self.isflipped:
            for sample_id in seq:
                image_name = self.imnames[sample_id][1:-1]
                imnames.append(image_name)
                im = Image.open(self.crops_dir + image_name).convert('RGB')
                im = np.asarray(im.resize((300, 300), Image.ANTIALIAS))
                if flipval:
                    im = np.fliplr(im)

                plt.subplot(self.samples.shape[0], self.samples.shape[1], i)
                plt.imshow(im)
                i += 1
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    def update_available_indices(self, seq_samples, seq_names):
        """
        Update the indicator vector of available samples to include in clique.
        Constraining so that a clique does not include two samples of the same sequence.
        """
        if seq_names is None:
            raise ValueError('seq_names must be not None')

        sample = seq_samples[0]
        # No temporal window, take the whole sequence
        ind_not_seq = np.asarray(seq_names) != seq_names[sample]
        self.available_indices = np.logical_and(self.available_indices, ind_not_seq)