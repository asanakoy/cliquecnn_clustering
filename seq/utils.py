import numpy as np


def get_clips_similarity(full_sim_matrix, clique_clips, frames_clip2, flipval):
    sims = full_sim_matrix[clique_clips, frames_clip2, flipval]
    assert sims.shape == clique_clips.shape
    return np.mean(sims, axis=1)
