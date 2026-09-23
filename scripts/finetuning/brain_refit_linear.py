#!/usr/bin/env python3
"""
Re-fit ridge-based linear encoding models on fine-tuned models.
"""

import argparse
import os
from pathlib import Path
import re
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
import joblib
from peft import get_peft_model, LoraConfig, TaskType
import torch
from tqdm import tqdm
from transformers import AutoModel

from brain_finetune import get_lanczos_mat_fir
from response_sources import load_responses
from npp_torch import mcorr, zscore
from ridge_utils.util import make_delayed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, help='Add this suffix to SAVE_DIR', required=True)
    parser.add_argument("--epoch", type=int, help='Use features extracted from the checkpoint from this epoch', required=True)
    parser.add_argument("--extract_features", action='store_true', help='Re-extract features, even if already present.')
    parser.add_argument("--no_extract_features", action='store_true', help="*Never* re-extract features. Useful if we know we're running on a machine with no GPUs.")
    parser.add_argument("--reuse_valphas", action='store_true', help='Re-use bootstrapped alphas from epoch 0. (Does nothing if epoch==0.)')
    parser.add_argument("--layer", type=int, help='Extract features from specific layer. The layer will be included in all output filenames.')
    parser.add_argument("--Pstory_trim", default=0, type=int, help='Trim this many TRs from the beginning of each held-out story before computing correlation.')
    parser.add_argument("--id_hparams", action='store_true', help='Use ridge hyperparameters from the intrinsic dimensionality paper.')
    args = parser.parse_args()

    no_extract_features: bool = args.no_extract_features

    DATA = Path(os.environ.get('DATA', './DATA')) # read-only data directory (fMRI responses, masks, TR timing)

    SAVE_DIR: Path = args.save_path
    run_config = torch.load(SAVE_DIR / "finetuning-run-config.pyt", weights_only=False)

    # Find the right file to load the checkpoint from
    # First parse all checkpoints (not actually necessary, but w/e)
    # Running this regardless of --no_extract_features may catch instances when
    # we run on a non-existent epoch.
    epoch_checkpoints: Dict[int, Path] = {}
    for f in SAVE_DIR.glob("model_epoch_*.pyt"):
        r = re.match(r"model_epoch_([0-9]+).pyt", f.name)
        if r is None: continue

        epoch_checkpoints[int(r.group(1))] = f
    epoch_checkpoints = {k: v for k, v in sorted(epoch_checkpoints.items(), key=lambda x: x[0])}

    if args.epoch is not None:
        epoch = args.epoch
    else:
        # Run the last epoch, for now
        epoch = max(epoch_checkpoints.keys())

    if no_extract_features:
        model = None
        device = None
    else:
        # Only load model & checkpoint if we might be extracting features.

        # If features are already extracted, we can run this script on non-GPU
        # nodes.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load original WavLM model
        model = AutoModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True).to(device)

        # Instantiate LoRA parameters
        if run_config['lora_rank'] > 0:
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                    target_modules= ['k_proj', 'v_proj', 'q_proj'],
                                    inference_mode=False, r=run_config['lora_rank'], lora_alpha=4, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)

        # Load the model checkpoint. (Only updates the parameters specified in the dict)
        model.load_state_dict(torch.load(epoch_checkpoints[epoch], weights_only=True, map_location=device), strict=False)

        # Set up model for feature extraction
        torch.set_grad_enabled(False)
        model.eval()

    subject: str = run_config['subject']

    ridge_save_dir = SAVE_DIR
    if args.layer is not None:
        ridge_save_dir = ridge_save_dir / "layers" / f"layer.{args.layer}"

    ridge_save_dir.mkdir(parents=True, exist_ok=True)


    # Load responses
    Rstories: list[str] = run_config['Rstories']
    Pstories: list[str] = run_config['Pstories']
    allstories = Rstories + Pstories
    Rresps, Presps = load_responses(subject, DATA=DATA)
    tr_times = joblib.load(DATA / "trfiles_huge.jbl")
    # Make sure no train-test leakage. Use the original response files as a
    # master copy of which stories we can train or test on.
    assert len(set(Rresps.keys()) & set(run_config['Pstories'])) == 0, "detected likely train-test overlap!"
    assert len(set(Presps.keys()) & set(run_config['Rstories'])) == 0, "detected likely train-test overlap!"
    Rresps = {story: Rresps[story] for story in Rstories}
    Presps = {story: Presps[story] for story in Pstories}

    # Extract features and preprocess
    layer_num = run_config['layer_num']
    delays = run_config['delays']
    downsample_fmri: Optional[float] = run_config.get('downsample_fmri', None)

    if args.layer is not None:
        # All multi-layer features are saved together
        features_path = SAVE_DIR / f"features_epoch_{epoch}_all_layers.joblib"
    else:
        features_path = SAVE_DIR / f"features_epoch_{epoch}.joblib"

    features_exist = features_path.exists() and (not args.extract_features)
    if features_exist:
        print('Existing features found at:', features_path)
        if args.layer is not None:
            all_features = joblib.load(features_path) # mmap will NOT work b/c "too many open files" error
            all_features = all_features[f"layer.{args.layer}"]
        else:
            all_features = joblib.load(features_path, mmap_mode='r') # mmap for lazy load
            all_features = all_features[f"layer.{layer_num}"] # needed for newer feature dump format

    else:
        assert not args.no_extract_features, f"Features not found at {str(features_path)}, but we were told not to extract any!"
        all_features = {} # store features as we get them

    # Load features and responses
    delRstim = []
    delPstim = []
    Rresp = []
    Presp = []
    for story in tqdm(allstories, desc='stories'):
        if features_exist:
            # Load already extracted features, dumped from
            # `brain_finetune_dump_features.py`
            story_features = all_features[story]
        else:
            # Live feature extraction from waveform is not implemented (it needs
            # target_token handling to match brain_finetune_dump_features.py).
            # Run that script first to dump features.
            raise NotImplementedError("live feature extraction is not implemented; run brain_finetune_dump_features.py first")

        # Do NOT mask the responses here. Only do that when selecting the model for
        # early stopping (i.e. brain_best_linear.py)
        resp = Rresps[story] if story in Rstories else Presps[story]

        if downsample_fmri is not None:
            story_tr_times = tr_times[story].tr_times[10:-5]
            # From brain_finetune.py
            # Currently, features are dumped in normal TR time. If we're using
            # downsampled responses, we need to downsample the features too.
            downsampled_tr_times = story_tr_times[::int(downsample_fmri)]
            response_downsample_lanczos_mat = torch.from_numpy(
                get_lanczos_mat_fir(story_tr_times,
                                    downsampled_tr_times)).float()
            response_downsample_lanczos_mat[response_downsample_lanczos_mat < 1e-10] = 0
            response_downsample_lanczos_mat = response_downsample_lanczos_mat.to_sparse_csr()

            # Technically not the same as during training (2 levels of
            # downsampling here vs. 1 in training), but it avoids re-extracting features.
            story_features = (response_downsample_lanczos_mat @ torch.from_numpy(story_features).float()).numpy()
            resp = (response_downsample_lanczos_mat @ torch.from_numpy(resp).float()).numpy()

        if story in Rstories:
            delRstim.append(make_delayed(zscore(story_features), delays))
            Rresp.append(zscore(resp))
        elif story in Pstories:
            delPstim.append(make_delayed(zscore(story_features), delays))
            Presp.append(zscore(resp))
        else:
            raise ValueError(f"Story {story} not found in either Rstories or Pstories!")

    delRstim = np.concatenate(delRstim, axis=0).astype(np.float32)
    delPstim = np.concatenate(delPstim, axis=0).astype(np.float32)
    Rresp = np.concatenate(Rresp, axis=0).astype(np.float32)
    Presp = np.concatenate(Presp, axis=0).astype(np.float32)
    print('delRstim.shape:', delRstim.shape, 'delPstim.shape:', delPstim.shape)
    print('Rresp.shape:', Rresp.shape, 'Presp.shape:', Presp.shape)


    # Set up ridge parameters & run regresion
    if args.id_hparams:
        # From: https://github.com/chengemily1/encoding-models/blob/main/main.py#L160
        alphas = np.logspace(1, 4, 15)
        ridge_config = {'nboots': 3, 'nchunks': 0.25, 'chunklen': 20, 'alphas': alphas, 'use_corr': True, 'single_alpha': False}
    else:
        alphas = np.logspace(1, 4, 10)
        ridge_config = {'nboots': 3, 'nchunks': 0.25, 'chunklen': 40, 'alphas': alphas, 'use_corr': True, 'single_alpha': False}
    nchunks = int(Rresp.shape[0] * ridge_config['nchunks'] / ridge_config['chunklen'])

    ridge_epoch0_outs_path = ridge_save_dir / 'ridge_epoch_0_ridge-outs.joblib'
    valphas_found = args.reuse_valphas and (epoch != 0) and (0 in epoch_checkpoints) and ridge_epoch0_outs_path.exists()

    ridge_prefix = 'ridge' + ('_no-cv' if valphas_found else '')
    ridge_config_path = ridge_save_dir / f"{ridge_prefix}_config.pyt"
    print('Saving ridge config to:', ridge_config_path)
    torch.save(ridge_config, ridge_config_path)
    ridge_outs_path = ridge_save_dir / f"{ridge_prefix}_epoch_{epoch}_ridge-outs.joblib"

    from ridge_utils.ridge import bootstrap_ridge, ridge

    if valphas_found:
        print('Re-using valphas from epoch 0...')
        valphas = joblib.load(ridge_epoch0_outs_path)['valphas']
        wt = ridge(delRstim, Rresp, alpha=valphas)
        pred = delPstim @ wt
        corrs = mcorr(pred, Presp)

        joblib.dump({'corrs': corrs, 'valphas': valphas},
                    ridge_outs_path)
        del pred
    else:
        print('Running cross-validation to select valphas...')
        wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(delRstim, Rresp, delPstim, Presp,
                                                               **{**ridge_config, 'nchunks': nchunks}) # type: ignore

        joblib.dump({'corrs': corrs, 'valphas': valphas, 'bscorrs': bscorrs, 'valinds': valinds},
                    ridge_outs_path)


    # Save performance separately for each held-out story
    story_corrs = {}
    Pstory_trim: int = args.Pstory_trim
    for Pstory in Pstories:
        stim = all_features[Pstory]
        resp = Presps[Pstory]

        pred = make_delayed(zscore(stim), delays) @ wt

        corrs = mcorr(pred[Pstory_trim:], resp[Pstory_trim:])
        story_corrs[Pstory] = corrs

    joblib.dump(story_corrs, ridge_save_dir / f"{ridge_prefix}_epoch_{epoch}_story_corrs.joblib")
