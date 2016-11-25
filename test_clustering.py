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
import scipy.io as sio
from trainhelper import trainhelper
from tqdm import tqdm

def runClustering(seed=None, **params):
    """
    Run clustering assignment procedure and return arrays for BatchLoader in a dict
    :param kwargs_generator: arguments for generator
    :param kwargs_sampler: arguments for sampler
    :return: Dict of arrays for BatchLoader
    """

    generator = BatchGenerator(sim_matrix=params['sim_matrix'],
                               flipvals=params['flipvals'],
                               seq_names=params['seq_names'],
                               crops_dir=params['crops_dir'],
                               relative_image_pathes=params['relative_image_pathes'],
                               num_cliques_per_initial_batch=params['num_cliques_per_initial_batch'],
                               num_samples_per_clique=params['num_samples_per_clique'],
                               anchors=params['anchors'],
                               seed=seed)
    init_batches = generator.generate_batches(num_initial_batches=100)
    sampler = BatchSampler(batches=init_batches,
                           sim_matrix=sim_matrix,
                           flipvals=flipvals,
                           seq_names=params['seq_names'],
                           crops_dir=params['crops_dir'],
                           relative_image_pathes=params['relative_image_pathes'],
                           seed=seed)
    for i in tqdm(range(10)):
        batch = sampler.sample_batch(batch_size=128,
                                     max_cliques_per_batch=8,
                                     mode='random')
    for i in range(30):
        sampler.cliques[i].visualize()


dataset = 'Caltech101'
category = 'Caltech101'

pathtosim_avg = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/Caltech101/sim/simMatrix_INIT.npy'

# data2 = h5py.File(pathtosim_avg, 'r')
sim_matrix = np.load(pathtosim_avg)
# data2 = sio.loadmat(pathtosim_avg)
# sim_matrix = (data2['sim_matrix'][()] + data2['sim_matrix'][()].T) / 2.0


flipvals = np.zeros(sim_matrix.shape)

pathtoimg = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/Caltech101/image_paths.txt'
pathtocrops = '/export/home/mbautist/Desktop/workspace/cnn_similarities/datasets/Caltech101/'


pathtoanchors = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/datasets/OlympicSports/labels_HIWIs/processed_labels/anchors_long_jump.mat'
anchors = h5py.File(pathtoanchors, 'r')
with open(pathtoimg) as f:
    imnames = f.readlines()
seqnames = [n[2:25] for n in imnames]


params = {
    'sim_matrix': sim_matrix,
    'flipvals': flipvals,
    'seq_names': None,
    'relative_image_pathes': imnames,
    'crops_dir': pathtocrops,
    'num_cliques_per_initial_batch': 5,
    'num_samples_per_clique': 5,
    'anchors': None,
    'dataset': dataset,
    'category': category,
    'num_batches_to_sample': 1000,
    'clustering_round': 0,
    'diff_prob': 0.1,

}
runClustering(**params)
