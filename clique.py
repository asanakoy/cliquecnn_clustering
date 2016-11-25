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
import matplotlib.pyplot as plt
from PIL import Image


class Clique(object):

    def __init__(self, crops_dir, nsamples):
        """
        Class for encapsulating and visualizing cliques
        """
        self.samples = np.empty(0, dtype=np.int32)
        self.isflipped = np.empty(0, dtype=np.bool)
        self.availableIndices = np.asarray([True] * nsamples, dtype=np.bool)
        self.weight = -1
        self.imnames = []
        self.label = None
        self.crops_dir = crops_dir

    def add_sample(self, s, f, imname):
        """
        Add a sample to a clique
        :param s: Sample to add
        :param f: Flip indicator
        :return:
        """
        assert np.isscalar(s), 'sample {} is not a scalar'.format(s)
        assert np.isscalar(f), 'flipval {} is not a scalar'.format(f)
        assert s not in self.samples, "Sample {} already in clique {}".format(s, self.samples)
        self.samples = np.append(self.samples, s)
        self.samples = self.samples.reshape(-1)
        self.isflipped = np.append(self.isflipped, f)
        self.imnames.append(imname)

    def remove_sample(self, idx):
        """
        Remove a sample from a clique given an index
        :param idx:
        :return:
        """
        self.samples = np.delete(self.samples, idx)
        self.isflipped = np.delete(self.isflipped, idx)
        self.imnames = np.delete(np.asarray(self.imnames), idx).tolist()

    def visualize(self):
        """
        Visualize clique
        :return:
        """
        fig = plt.figure()
        plt.cla()
        vol = np.empty((300, 300, 3, 0), dtype=np.uint8)
        imnames = list()
        for sample_id, sample in enumerate(self.samples):
            im = Image.open(self.crops_dir + self.imnames[sample_id][1:-1]).convert('RGB')
            imnames.append(self.imnames[sample_id][1:-1])

            im = im.resize((300, 300), Image.ANTIALIAS)
            im = np.asarray(im)
            # if self.isflipped[sample_id]:
            #     im = np.fliplr(im)
            im = np.expand_dims(im, axis=3)
            vol = np.append(vol, im, axis=3)
        self.implay(vol, imnames=imnames)

    def implay(self, volume, imnames=None, fps=2, ax=None, **kw):
        """Play a sequence of image in `volume` as a video.
        Parameters
        ----------
        volume: ndarray
            The video volume to be played. Its size can be either MxNxK (for
            single-channel image per frame) or MxNxCxK (for multi-channel image per
            frame).
        fps: int, optional
            The frame rate of the video.
        ax: axes, optional
            The axes in which the video to be played. If not specified, default to
            `plt.gca()`.
        **kw: key-value pairs
            Other parameters to be passed to `ax.imshow`, e.g. `cmap="gray"`,
            `vmin=0`, `vmax=1`, etc.
        """
        if not ax:
            ax = plt.gca()
            ax.cla()
        num_frames = volume.shape[-1]
        for i in xrange(num_frames):
            ax.cla()
            ax.imshow(volume[..., i], **kw)
            if imnames is not None:
                ax.set_title(imnames[i])
            plt.pause(1. / fps)
            pass
        plt.close()

