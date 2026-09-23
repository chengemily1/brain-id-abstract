#!/usr/bin/env python3
"""
Choose the best epoch based on validation performance, and re-run cross-validated
linear regression (ridge) with that epoch.

Based on and calls brain_refit_linear.py .
"""

import argparse
from pathlib import Path
import re
import subprocess
from typing import Dict

import numpy as np
import numpy.typing as npt
import joblib
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, help='Add this suffix to SAVE_DIR', required=True)
    parser.add_argument("--extract_features", action='store_true', help='Re-extract features, even if already present.')
    parser.add_argument("--no_extract_features", action='store_true', help="*Never* re-extract features. Useful if we know we're running on a machine with no GPUs.")
    args = parser.parse_args()

    SAVE_DIR: Path = args.save_path
    run_config = torch.load(SAVE_DIR / "finetuning-run-config.pyt", weights_only=False)
    num_epochs = run_config.get('num_epochs', 20) # assume value if not provided

    # Find the right file to load the checkpoint from
    # First parse all checkpoints (not actually necessary, but w/e)
    epoch_checkpoints: Dict[int, Path] = {}
    for f in SAVE_DIR.glob("model_epoch_*.pyt"):
        r = re.match(r"model_epoch_([0-9]+).pyt", f.name)
        if r is None: continue

        epoch_checkpoints[int(r.group(1))] = f
    epoch_checkpoints = {k: v for k, v in sorted(epoch_checkpoints.items(), key=lambda x: x[0])}

    # Set up model for feature extraction
    torch.set_grad_enabled(False)

    # Load fine-tuning config
    Rstories: list[str] = run_config['Rstories']
    Pstories: list[str] = run_config['Pstories']
    vox_mask = run_config['vox_mask'].numpy()

    # Only use the normal validation stories that were actually held-out during
    # training. (Sometimes "fromboyhoodtofatherhood" is used as a training
    # story, so exclude it from val_stories in that scenario.)
    val_stories = ['fromboyhoodtofatherhood', 'onapproachtopluto']
    val_stories = [val_story for val_story in val_stories if (val_story in Pstories)]
    print('Validation stories:', val_stories)
    test_stories = ['wheretheressmoke']
    assert all((test_story not in Rstories) for test_story in test_stories), \
        f"some test stories found in the train set! ({set(Pstories).intersection(test_stories)})"
    print('Test stories:', test_stories)

    # Find the epoch with the highest validation performance
    epoch_val_corrs: list[npt.NDArray[np.floating]] = [] # voxelwise val perf across epochs
    for epoch in epoch_checkpoints.keys():
        ridge_prefix = 'ridge' + ('_no-cv' if epoch > 0 else '') # epoch 0 is always bootstrapped
        story_corrs = joblib.load(SAVE_DIR / f"{ridge_prefix}_epoch_{epoch}_story_corrs.joblib")
        assert all(val_story in story_corrs for val_story in val_stories), \
            f"val stories missing from asdf"
        this_val_corrs = np.nanmean(np.stack([story_corrs[val_story][vox_mask] for val_story in val_stories],
                                              axis=0), axis=0) # average over val stories. shape: (vox_mask_count,)

        epoch_val_corrs.append(this_val_corrs)

    assert len(epoch_val_corrs) == num_epochs+1, \
        f"Expected {num_epochs+1} epochs, but found {len(epoch_val_corrs)}. Double check argmax logic below before removing the assert."
    epoch_val_corrs = np.stack(epoch_val_corrs, axis=0) # size: (num_epochs, vox_mask_count)
    best_epoch = list(epoch_checkpoints.keys())[np.argmax(np.nanmean(epoch_val_corrs, axis=1))] # int
    print('Val corrs:', np.nanmean(epoch_val_corrs, axis=1))
    print('Best epoch:', best_epoch)
    save_outputs = {'best_epoch': best_epoch, 'val_corrs': epoch_val_corrs}
    joblib.dump(save_outputs,
                SAVE_DIR / "ridge_best_epoch.joblib")

    # Run full cross-validation/bootstrapping for the best epoch.
    if best_epoch == 0:
        print('Best epoch is 0, which already has done bootstrapping, so performing no further actions.')
        exit(0)

    # best_epoch > 0, so need to perform bootstrapping
    refit_cli_args = ['python3', './brain_refit_linear.py', '--save_path', str(args.save_path), '--epoch', str(best_epoch)]
    print('Refitting ridge parameters for epoch', best_epoch)
    # Don't bother with checking if both arguments are present. Let the other
    # script handle that.
    if args.extract_features:
        refit_cli_args += ['--extract_features']
    if args.no_extract_features:
        refit_cli_args += ['--no_extract_features']

    print('Running:')
    print('+', subprocess.list2cmdline(refit_cli_args))
    subprocess.run(refit_cli_args, check=True)
