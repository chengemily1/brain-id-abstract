"""Lanczos interpolation, generalized to use torch for the (sparse) resampling matmul.

Trimmed to just what scripts/finetuning/ needs: lanczosfun (used directly to
build a lanczos-window sinc matrix in brain_finetune.py) and lanczosinterp2D
(used to downsample extracted-feature sequences to TR time in
brain_finetune_dump_features.py/brain_refit_linear.py). lanczosinterp2D takes
and returns plain numpy arrays, but does the actual resampling matmul as a
torch sparse @ dense multiply internally, since the sinc matrix is often only
1-3% dense - constructing and multiplying it as a dense numpy array would be
far slower.
"""
from typing import cast

import numpy as np
import numpy.typing as npt
import torch

import multiprocessing
import os

num_cores = multiprocessing.cpu_count()
num_threads = min(num_cores, int(os.environ.get('OMP_NUM_THREADS', num_cores)), int(os.environ.get('MKL_NUM_THREADS', num_cores)))
torch.set_num_threads(num_threads)

def lanczosfun(cutoff: float, t: npt.NDArray, window: int=3):
    """Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = t * cutoff
    val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0
    return val# / (val.sum() + 1e-10)

def lanczosinterp2D(data: npt.NDArray, oldtime: npt.NDArray, newtime: npt.NDArray,
                    window: int=3, cutoff_mult: float=1.0, rectify: bool=False) -> npt.NDArray[np.floating]:
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i). A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.

    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the interpolation filter.

    [window] lobes of the sinc function will be used. [window] should be an integer.
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult # this is a scalar, not an np.array

    ## Build up sinc matrix ##
    sincmat = lanczosfun(cast(float, cutoff), (newtime[:, None] - oldtime), window)

    if rectify:
        raise NotImplementedError()
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)),
                            np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        ## Construct new signal by multiplying the sinc matrix by the data ##
        # Use a sparse matrix. The density of `sincmat` is often between
        # 0.01 and 0.03, which is quite sparse, especially for temporally
        # high-frequency data. Constructing it as a dense matrix is fine
        # because it's pretty small.
        sincmat = torch.from_numpy(sincmat).to_sparse() # absolutely CLUTCH function right here
        newdata = sincmat @ torch.from_numpy(data).to(sincmat)
        newdata = newdata.cpu().numpy()

    return newdata
