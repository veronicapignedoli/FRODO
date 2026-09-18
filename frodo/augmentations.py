"""3D augmentations only: stateless flip + 90-degree rotation, applied
identically to the QSM and FLAIR patch of a sample. No model or data imports."""
import random

import numpy as np


class RandomFlipRotate3D:
    """Random independent axis flips (p=0.5 each) plus a random 90-degree
    rotation in a random plane, applied with the same random state to both
    modalities. Used only during Stage 1/2 training, never at evaluation."""

    def __call__(self, qsm_patch, flair_patch):
        qsm, flair = qsm_patch, flair_patch

        for axis in range(3):
            if random.random() < 0.5:
                qsm = np.flip(qsm, axis=axis)
                flair = np.flip(flair, axis=axis)

        k = random.randint(0, 3)
        if k > 0:
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            qsm = np.rot90(qsm, k=k, axes=axes)
            flair = np.rot90(flair, k=k, axes=axes)

        return np.ascontiguousarray(qsm), np.ascontiguousarray(flair)
