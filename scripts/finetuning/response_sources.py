import os
from pathlib import Path
from typing import Optional

import joblib as jbl
import numpy as np
import numpy.typing as npt

def load_responses(subject: str, DATA: Optional[Path]=None) -> tuple[dict[str, npt.NDArray], dict[str, npt.NDArray]]:
    """Load fMRI responses for a given subject or baseline feature.

    Returns:
        Rresps: Responses for the subject (dict of story -> (TR, nvox) array)
        Presps: Responses for the subject's partner (dict of story -> (TR, nvox) array)
    """
    if DATA is None:
        DATA = Path(os.environ.get('DATA', './DATA')) # read-only data directory (fMRI responses, masks, TR timing)

    Pstories = ['wheretheressmoke', 'fromboyhoodtofatherhood', 'onapproachtopluto']

    # Single dict of story -> (TR, nvox) array covering all stories, same
    # format/naming as scripts/encoding_models/main.py's response loading.
    responses_path = f"UTS0{subject}_responses.jbl"
    resps = jbl.load(DATA / responses_path, mmap_mode='r') # mmap for lazy load
    Rresps = {story: resps[story] for story in resps.keys() if (story not in Pstories)}
    Presps = {story: resps[story] for story in Pstories}

    return Rresps, Presps
