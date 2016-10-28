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

import h5py
from batchgenerator import BatchGenerator
from batchsampler import BatchSampler
import numpy as np
from trainhelper import trainhelper


def runClustering(**params):
    """
    Run clustering assignment procedure and return arrays for BatchLoader in a dict
    :param kwargs_generator: arguments for generator
    :param kwargs_sampler: arguments for sampler
    :return: Dict of arrays for BatchLoader
    """

    generator = BatchGenerator(**params)
    init_batches = generator.generateBatches(init_nbatches=100)
    params['batches'] = init_batches
    sampler = BatchSampler(**params)
    sampler.updateCliqueSampleProb(np.ones(len(sampler.cliques)))

    for i in range(params['sampled_nbatches']):
        print "Sampling batch {}".format(i)
        batch = sampler.sampleBatch(batch_size=128, max_cliques_per_batch=8, mode='heuristic')

    sampler.updateSimMatrix()
    sampler.transitiveCliqueUpdate()

dataset = 'OlympicSports'
category = 'long_jump'
pathtosim = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/compute_similarities/sim_matrices/hog-lda/simMatrix_long_jump.mat'
pathtosim_avg = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/similarities_lda/d_long_jump.mat'
data = h5py.File(pathtosim, 'r')
data2 = h5py.File(pathtosim_avg, 'r')
pathtoimg = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/image_data/imagePaths_long_jump.txt'
pathtocrops = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/crops/long_jump'
pathtoanchors = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/labels_HIWIs/processed_labels/anchors_long_jump.mat'
anchors = h5py.File(pathtoanchors, 'r')
with open(pathtoimg) as f:
    imnames = f.readlines()
seqnames = [n[2:25] for n in imnames]


params = {
    'simMatrix': (data2['d'][()] + data2['d'][()]) / 2.0,
    'flipMatrix': data['flipval'][()],
    'seqNames': seqnames,
    'imagePath': imnames,
    'pathToFolder': pathtocrops,
    'init_nCliques': 10,
    'nSamples': 8,
    'anchors': anchors,
    'dataset': dataset,
    'category': category,
    'sampled_nbatches': 1000,
    'clustering_round': 0
}
trainhelper.runClustering(**params)
