import numpy as np
import os
import csv
from scipy.spatial.distance import cdist
from pathlib import Path

def save_experiment(data, mds_emb, v_emb, idx_lms, idx_sources, folder):
    print("Save experiment")
    filepath = os.path.join('Results', folder)
    Path(filepath).mkdir(parents=True, exist_ok=True)

    np.savez(os.path.join(filepath, 'embedding_freyfaces.npz'), ff_data=data, mds_emb=mds_emb, v_emb=v_emb)

    with open(os.path.join(filepath, 'idx_lm'), 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows(idx_lms)

    with open(os.path.join(filepath, 'idx_sources'), 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows(idx_sources)

def select_landmarks_standard(data, n, nlm):
    """Randomly select nlm landmarks from data,
    n is the size of the dataset
    """
    idx = np.random.choice(np.arange(0, n), size=nlm)
    return data[idx]

def select_landmarks_kmeanspp(data, n, nlm):
    """Select landmarks with probability proportional
    to distance to existing landmarks"""
    lms = []
    lms_idx = []

    # First choose a random landmark
    idx = np.random.choice(np.arange(0, n), size=1)
    lms.append(data[idx])
    lms_idx.append(idx)

    for i in range(nlm - 1):
        distances = np.zeros(n)
        for lm in lms:
            distances = np.add(distances, cdist(lm, data))[0]

        mindist = (1-i/(nlm*2))*np.max(distances)
        idx = np.where(distances < mindist)
        distances[idx] = 0  # All points that are too close to existing lm are excluded (probability for selection to 0)
        P = distances/np.sum(distances)
        idx = np.random.choice(np.arange(0, n), size=1, p=P)
        lms.append(data[idx])
        lms_idx.append(idx)
    return np.array(lms).reshape(nlm, -1), lms_idx

def select_landmarks_max_spread(data, n, nlm):
    """Select landmarks nlm from data and make certain they are maximaly spread"""
    lms = []
    lms_idx = []

    # First choose a random landmark
    idx = np.random.choice(np.arange(0, n), size=1)
    lms.append(data[idx])
    lms_idx.append(idx)

    # Choose the next landmark that is furthest away
    for _ in range(nlm-1):
        distances = np.zeros(n)
        for lm in lms:
            distances = np.add(distances, cdist(lm, data))
        idx = [np.argmax(distances)]
        lms.append(data[idx])
        lms_idx.append(idx)
    return np.array(lms).reshape(nlm, -1), lms_idx