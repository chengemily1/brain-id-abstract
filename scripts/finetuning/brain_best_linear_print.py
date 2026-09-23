#!/usr/bin/env python3
"""
Prints performance of pretrained and finetuned models using the outputs of
brain_best_linear.py .

Useful for quick inspection of results, not plotting or aggregating.
"""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import joblib
import torch
import xarray as xr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, help='Add this suffix to SAVE_DIR', required=True)
    args = parser.parse_args()

    SAVE_DIR: Path = args.save_path
    run_config = torch.load(SAVE_DIR / "finetuning-run-config.pyt", weights_only=False)
    num_epochs = run_config.get('num_epochs', 20) # assume value if not provided

    # Load fine-tuning config
    Rstories: list[str] = run_config['Rstories']
    Pstories: list[str] = run_config['Pstories']
    vox_mask = run_config['vox_mask'].numpy()
    subjects: list[str] = [run_config['subject']]

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

    best_epoch: int = joblib.load(
        SAVE_DIR / f"ridge_best_epoch.joblib")['best_epoch']
    print('Best epoch:', best_epoch)
    
    perfs = xr.DataArray(
        dims=['subject', 'training', 'story_set'],
        coords={
            'subject': subjects,
            'training': ['pretrained', 'finetuned'],
            'story_set': ['val', 'test'],
        },
        name='encperf',
    )
    
    for subject in subjects:
        # Load pretrained model results
        pretrained_story_corrs = joblib.load(
            SAVE_DIR / f"ridge_epoch_0_story_corrs.joblib")
        finetuned_story_corrs = joblib.load(
            SAVE_DIR / f"ridge_epoch_{best_epoch}_story_corrs.joblib")

        # NOTE: voxel subsetting via category_key isn't supported yet (only
        # category_key == 'all' works; see get_voxel_mask in brain_finetune.py).
        pretrained_story_corrs: dict[str, float] = {story: np.nanmean(corrs) for story, corrs in pretrained_story_corrs.items()}
        finetuned_story_corrs: dict[str, float] = {story: np.nanmean(corrs) for story, corrs in finetuned_story_corrs.items()}
        
        for training_name, training_perfs in [('pretrained', pretrained_story_corrs),
                                              ('finetuned', finetuned_story_corrs)]:
            val_perfs = np.nanmean(np.stack([training_perfs[val_story] for val_story in val_stories],
                                            axis=0))
            test_perfs = np.nanmean(np.stack([training_perfs[test_story] for test_story in test_stories],
                                                axis=0))
            perfs.loc[dict(subject=subject, training=training_name, story_set='val')] = val_perfs
            perfs.loc[dict(subject=subject, training=training_name, story_set='test')] = test_perfs
    
    perf_change = perfs.sel(training='finetuned') - perfs.sel(training='pretrained')
    perf_change = perf_change.rename('change')
    print("\n=== Performance Summary ===")
    print("Average performance across subjects:")
    print(perfs.mean(dim='subject').to_dataframe().unstack('training'))
    print("\nAverage performance change due to finetuning:")
    print(perf_change.mean(dim='subject').to_dataframe())
    print("\nPerformance change due to finetuning, by subject:")
    print(perf_change.to_dataframe().unstack('subject'))
