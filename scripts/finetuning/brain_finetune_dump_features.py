#!/usr/bin/env python3
"""
Save downsampled features from fine-tuned models before fitting linearized
encoding models.
"""

import argparse
import collections
import os
from pathlib import Path
import re
import time
from typing import cast, Dict, Iterable, List, Optional

import numpy as np
import numpy.typing as npt
import joblib
from peft import get_peft_model, LoraConfig, TaskType
import peft
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from transformers import AutoModel

from brain_finetune import TARGET_SAMPLE_RATE
from extract_features_hf import slide_window, batch_snippets, extract_features_hf
from interpdata_torch import lanczosinterp2D
from npp_torch import mcorr, zscore
from ridge_utils.ridge import bootstrap_ridge
from torch_sparse_utils import slice_sparse_tensor
from ridge_utils.util import make_delayed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, help='Add this suffix to SAVE_DIR', required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--batchsz", type=int, default=256, help='Batch size during feature extraction. Can be different (larger) than during fine-tuning.')
    parser.add_argument("--num_threads", type=int, help='Use this many threads for torch.set_num_threads. Useful when running multiple fine-tuning jobs on one machine.')
    parser.add_argument("--stories", type=str, nargs='+', help='Only dump features for these stories. (Ignores previously extracted features.)')
    parser.add_argument("--all_layers", action='store_true', help='If set, dump features for all layers.')
    args = parser.parse_args()

    device = torch.device('cuda')

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    DATA = Path(os.environ.get('DATA', './DATA')) # read-only data directory (fMRI responses, masks, TR timing)

    SAVE_DIR: Path = args.save_path
    print("Loading from save_path:", SAVE_DIR)
    run_config = torch.load(SAVE_DIR / "finetuning-run-config.pyt", weights_only=False)
    target_layer_num: int = run_config['layer_num']

    # Load original WavLM model
    model = AutoModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True).to(device)

    # Instantiate LoRA parameters
    if run_config['lora_rank'] > 0:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                target_modules= ['k_proj', 'v_proj', 'q_proj'],
                                inference_mode=False, r=run_config['lora_rank'], lora_alpha=4, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)

    # Find the right file to load the checkpoint from
    # First parse all checkpoints (not actually necessary, but w/e)
    epoch_checkpoints: Dict[int, Path] = {}
    for f in SAVE_DIR.glob("model_epoch_*.pyt"):
        r = re.match(r"model_epoch_([0-9]+).pyt", f.name)
        if r is None: continue

        epoch_checkpoints[int(r.group(1))] = f

    if args.epoch is not None:
        epoch = args.epoch

        # Load the model checkpoint. (Only updates the parameters specified in the dict)
        model.load_state_dict(torch.load(epoch_checkpoints[epoch], weights_only=True), strict=False)
        if isinstance(model, peft.peft_model.PeftModel):
            # Merge LoRA weights into the base model. Reduces some overhead.
            model = model.merge_and_unload()
    else:
        # Run the last epoch, for now
        epoch = max(epoch_checkpoints.keys())


    # Set up model for feature extraction
    torch.set_grad_enabled(False)
    model = nn.DataParallel(model) # must be done after LoRA
    model.eval()

    # Load story names
    features_out_path = SAVE_DIR / f"features_epoch_{epoch}"
    #features_out_path = SAVE_DIR / f"features_epoch_pretrained"
    if args.all_layers:
        features_out_path = features_out_path.with_name(features_out_path.stem + "_all_layers")
    features_out_path = features_out_path.with_suffix('.joblib')
    if args.stories is None:
        # Automatically determine stories from finetuning config
        Rstories: list[str] = run_config['Rstories']
        Pstories: list[str] = run_config['Pstories']
        allstories = Rstories + Pstories
        all_features: dict[str, dict[str, np.ndarray]] = {} # layer_name -> (story -> features)
    else:
        # Use stories given from CLI
        allstories = args.stories
        # Load previous features
        if features_out_path.exists():
            # Add previously extracted features
            print('Loading existing features from:', features_out_path)
            all_features = {f"layer.{target_layer_num}": joblib.load(str(features_out_path))}
            print('Found existing features for', len(all_features.keys()), 'stories')
        else:
            all_features = {}
    all_features = collections.defaultdict(dict, all_features)
    print('Extracting features for', len(allstories), 'stories')
    # trfiles_huge.jbl is the same file scripts/encoding_models/main.py and
    # brain_finetune.py use: a dict of story -> [TRFile], each with a
    # `.trtimes` list of TR onset times (seconds).
    tr_times = joblib.load(DATA / "trfiles_huge.jbl") # need these for downsampling

    # Extract features and preprocess
    sel_layers = [target_layer_num] if not args.all_layers else list(range(13))
    delays = run_config['delays']
    target_token = run_config.get('target_token', 0) # 0 was the default before 2026-01-05
    for story in tqdm(allstories, desc='stories'):
        story_wav, sr = torchaudio.load(f"processed_stimuli/{story}.wav")
        story_wav = story_wav.mean(0).to(device)
        assert sr == TARGET_SAMPLE_RATE, f"resample audio for story {story}!"
        # Interestingly, DataParallel actually makes this very slow!
        outs = extract_features_hf(
            model,
            model_config={'stride': 320, 'min_input_length': 400},
            wav=story_wav,
            chunksz_sec=run_config['chunk_sz_sec'], contextsz_sec=run_config['context_sz_sec'],
            num_sel_frames=1, frame_skip=5, batchsz=args.batchsz*len(model.device_ids),
            require_full_context=False, # maybe require_full=False?
            disable_tqdm=False,
            return_numpy=False, move_to_cpu=False, sel_layers=sel_layers,
            target_token=target_token)

        for layer_num in sel_layers:
            layer_name = f"layer.{layer_num}"
            story_features = outs['module_features'][layer_name].cpu().numpy()

            story_features = lanczosinterp2D(
                story_features, outs['times'][:,1].cpu().numpy(), np.array(tr_times[story][0].trtimes))
            story_features = story_features[10:-5, :].astype(np.float32)

            all_features[layer_name][story] = story_features # shape: (dim_size, time_trs)
            del story_features
        del outs

    # Use joblib to save so we can read with mmap
    print('Saving features to:', features_out_path)
    joblib.dump(all_features, features_out_path)
