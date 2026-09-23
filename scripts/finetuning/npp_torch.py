# vim: set tabstop=4

"""zscore/mcorr, generalized to work on both numpy arrays and torch tensors.

Trimmed to just what scripts/finetuning/ needs: mcorr is called directly on
torch.Tensor model outputs as a differentiable training loss (in
brain_finetune.py's spatial_loss/temporal_loss), so it must stay vectorized
and autodiff-compatible rather than using the original numpy-only,
per-column Python loop.
"""
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch

ArrayType = TypeVar('ArrayType', npt.NDArray, torch.Tensor)

## Z-score -- z-score each column
def zscore(v: ArrayType) -> ArrayType:
	s: ArrayType = v.std(0)
	m: ArrayType = v - v.mean(0)

	if isinstance(s, torch.Tensor): nonzero_std = ~torch.isclose(s, torch.zeros_like(s))
	else: nonzero_std = ~np.isclose(s, np.zeros_like(s))
	m[:, nonzero_std] /= s[nonzero_std]
	return m

zscore.__doc__ = """Z-scores (standardizes) each column of [v]."""
zs = zscore

## Matrix corr -- find correlation between each column of c1 and the corresponding column of c2
mcorr = lambda c1,c2: (zs(c1)*zs(c2)).mean(0)
mcorr.__doc__ = """Matrix correlation. Find the correlation between each column of [c1] and the corresponding column of [c2]."""
