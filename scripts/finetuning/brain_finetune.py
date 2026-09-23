#!/usr/bin/env python3
"""
Fine-tune a WavLM-style speech model against fMRI responses (braintuning).

Run from scripts/finetuning/. Its `interpdata_torch`/`npp_torch` imports are
this directory's own local, torch-compatible modules, not the numpy-only
originals under scripts/encoding_models/ridge_utils/, so those don't need
PYTHONPATH setup. However, loading trfiles_huge.jbl (for TR timing) does:
that file was pickled against `huth.stimulus_utils.TRFile`, a shim under
scripts/encoding_models/huth/ that itself imports scripts/encoding_models's
ridge_utils.stimulus_utils, so PYTHONPATH needs that directory too, e.g.:
    PYTHONPATH=../encoding_models python3 brain_finetune.py ...
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
from typing import Optional, cast, Iterable

import numpy as np
import numpy.typing as npt
import joblib as jbl
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from transformers import AutoModel
import wandb

from extract_features_hf import slide_window, batch_snippets
from interpdata_torch import lanczosfun
from npp_torch import mcorr
from torch_sparse_utils import slice_sparse_tensor
from response_sources import load_responses

def make_delayed_torch(stim: torch.Tensor, delays: Iterable[int], circpad: bool=False) -> torch.Tensor:
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    dstims = []
    for di,d in enumerate(delays):
        dstim = torch.zeros_like(stim)
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.clone()
        dstims.append(dstim)
    return torch.hstack(dstims)

class DelayFeatures(nn.Module):
    def __init__(self, delays: list[int]) -> None:
        super().__init__()
        assert np.allclose(np.diff(delays), 1) and (np.array(delays) > 0).all(), "current logic only works for non-zero, 1-spaced delays"
        self.delays = delays

    def forward(self, x):
        return torch.cat(
            [x.new_zeros((max(self.delays), x.shape[1])), x],
            dim=0).unfold(0, max(self.delays), 1).flip(2)[:-min(self.delays)].permute(0, 2, 1).flatten(-2)

class Permute(nn.Module):
    """Module that just calls x.permute(dims). Useful for nn.Sequential"""
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

def get_lanczos_mat_fir(oldtime: npt.NDArray, newtime: npt.NDArray, window: int = 3, cutoff_mult: float = 1.0, rectify = False):
    """get matrix for downsampling from TR times to word times.
    This is based off of `lanczosinterp2D` from `interpdata.py`.
    """
    cutoff = 1 / np.mean(np.diff(newtime)) * cutoff_mult
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cast(float, cutoff), newtime[ndi] - oldtime, window)
    return sincmat

# sample rate for WavLM, Hubert, and some others. NOT the sample rate for Whisper!
TARGET_SAMPLE_RATE = 16000

def output_block_responses(model, layer_num: int, story: str, tr_start: int, tr_end: int, target_token: int = 0, mean_pool: bool = False) -> torch.Tensor:
    """Output responses for a given block of TRs
    NOTE: this function relies on some global state, like:
    `story_wavs`, `story_wav_snippet_bounds`, `lanczos_mat`, `preprocess_features`, `encoding_out`"""
    sliced_lanczos, first_snippet_idx = slice_sparse_tensor(lanczos_mat, (tr_start, tr_end), trim_columns=True)
    last_snippet_idx = first_snippet_idx+sliced_lanczos.shape[1]

    # These are the waveform snippets that are relevant to this TR block.
    tr_block_wav_snippet_bounds = wav_snippet_bounds[first_snippet_idx:last_snippet_idx]
    batched_wav_snippet_bounds = batch_snippets(
        tr_block_wav_snippet_bounds, batchsz=featext_batch_size,
        require_full_context=False)

    # Iterate through each batch of snippets and accumulate hidden states.
    # This loop is a basically very stripped down version of `extract_features_hf`.
    forward_passes = []
    for snippet_starts, snippet_ends in batched_wav_snippet_bounds:
        wav_snippet_batch = []
        for snippet_start, snippet_end in zip(snippet_starts, snippet_ends):
            wav_snippet_batch.append(story_wav[snippet_start:snippet_end])
        wav_snippet_batch = torch.stack(wav_snippet_batch, dim=0).to(model.device_ids[0])
        if mean_pool:
            out = model(wav_snippet_batch)['hidden_states'][layer_num].mean(dim=1) # shape: (batch sz, hidden state)
        else:
            out = model(wav_snippet_batch)['hidden_states'][layer_num][:, target_token] # shape: (batch sz, hidden state)
        forward_passes.append(out)
        del out

    forward_passes = torch.cat(forward_passes, dim=0)
    if torch.isnan(forward_passes).any() :
        print("found NaN 1", story)

    forward_passes = sliced_lanczos @ forward_passes # downsample; shape (num_trs, hidden state)
    if torch.isnan(forward_passes).any() :
        print("found NaN 2", story)

    forward_passes = preprocess_features(forward_passes) # z-score and delay; shape (num_trs, hidden state)

    if torch.isnan(forward_passes).any() :
        print("found NaN 3", story)

    forward_passes = encoding_out(forward_passes) # shape: (num_trs, vox_mask_count)

    if torch.isnan(forward_passes).any() :
        print("found NaN 4", story)

    return forward_passes

def spatial_loss(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # assumes that the dimensions are (time, space)
    return -mcorr(pred.T, true.T).mean()

def temporal_loss(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # assumes that the dimensions are (time, space)
    return -mcorr(pred, true).mean()

def mse_loss(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # assumes that the dimensions are (time, space)
    return (true-pred).square().mean()

loss_fns = {'spatial': spatial_loss, 'temporal': temporal_loss, 'mse': mse_loss}

def get_voxel_mask(subject: str, category_key: str, num_voxels: int) -> torch.Tensor:
    assert category_key == 'all'
    vox_mask = torch.ones(num_voxels, dtype=torch.bool)
    return vox_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", help="subject number for fMRI responses, e.g. 3 for UTS03", required=True)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--bottle_lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--bottle_size", type=int, default=100)
    parser.add_argument("--category_key", type=str, default='all')
    parser.add_argument("--layer_num", type=int, default=9)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--loss", default='spatial', choices=list(loss_fns.keys()))
    parser.add_argument("--save_suffix", type=str, help='Add this suffix to SAVE_DIR')
    parser.add_argument("--num_threads", type=int, help='Use this many threads for torch.set_num_threads. Useful when running multiple fine-tuning jobs on one machine.')
    parser.add_argument("--reinit_layers", action='store_true', help='Re-initialize all layers *after* layer 9 (as per Pasad et al., 2021)')
    parser.add_argument("--story_config", type=Path, help='Load the list of Rstories and Pstories from this JSON file. Used for finding scaling laws.')
    parser.add_argument("--target_token", type=int, default=0, help='Token within each snippet to extract features from. 0 means the first token in each snippet. -1 means the final token.')
    parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument("--mean_pool", action='store_true', help='Use mean pooling across tokens instead of taking a single target token.')
    parser.add_argument("--downsample_fmri", type=int, help='Train on downsampled fMRI responses. E.g., 2 means every other TR.')
    parser.add_argument("--ndelays", type=int, default=4)
    args = parser.parse_args()
    run_config = vars(args)

    device = torch.device('cuda')

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    DATA = Path(os.environ.get('DATA', './DATA')) # read-only data directory (fMRI responses, masks, TR timing)
    SCRATCH = Path(os.environ.get('SCRATCH', './SCRATCH'))
    SCRATCH = SCRATCH / args.subject
    DATA.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    print("Using subject:", args.subject)
    print("category_key is", args.category_key)

    print('Loading responses...')
    Rresps, Presps = load_responses(args.subject, DATA)

    if args.story_config is not None:
        print('Loading list of stories from:', args.story_config)
        with open(args.story_config, 'r') as f:
            story_config = json.load(f)

        train_stories = story_config['Rstories']
        test_stories = story_config['Pstories']
        assert all(x in Rresps.keys() for x in train_stories)
        assert all(x in Presps.keys() for x in test_stories)
    else:
        train_stories = list(Rresps.keys())
        test_stories = list(Presps.keys())

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create separate validation set, and remove from training & test set
    heldout_stories = copy.deepcopy(test_stories) # this is val + test
    val_stories = ['fromboyhoodtofatherhood', 'onapproachtopluto']
    test_stories = [story for story in heldout_stories if (story not in val_stories)]
    assert not any([story in train_stories for story in val_stories]), "val stories should not be in train stories"

    all_stories = heldout_stories + train_stories
    train_stories = [story for story in all_stories if (story not in heldout_stories)]
    train_stories = sorted(train_stories) # enforce order across processes
    val_stories = sorted(val_stories)
    test_stories = sorted(test_stories)
    heldout_stories = sorted(heldout_stories)
    # Shuffle training stories
    random.shuffle(train_stories)
    all_stories = heldout_stories + train_stories
    run_config['Rstories'] = train_stories
    run_config['Pstories'] = heldout_stories
    run_config['val_stories'] = val_stories
    run_config['test_stories'] = test_stories
    print('Training on', len(train_stories), 'stories')
    print('Validation on', len(val_stories), 'stories')
    print('Testing on', len(test_stories), 'stories')

    # trfiles_huge.jbl is the same file scripts/encoding_models/main.py uses:
    # a dict of story -> [TRFile], each with a `.trtimes` list of TR onset
    # times (seconds), trimmed the same way (10 TRs off the start, 5 off the
    # end) as the fMRI response data in UTS0X_responses.jbl.
    tr_times = jbl.load(DATA / "trfiles_huge.jbl")
    all_stories = list(filter(lambda k: k in tr_times.keys(), all_stories))

    lora_lr = args.lora_lr
    bottle_lr = args.bottle_lr
    bottle_size: int = args.bottle_size # 25, 50
    layer_num: int = args.layer_num
    category_key: str = args.category_key
    target_token: int = args.target_token
    mean_pool: bool = args.mean_pool
    downsample_fmri: Optional[float] = args.downsample_fmri

    num_voxels: int = Presps['wheretheressmoke'].shape[1]
    print("num_voxels:", num_voxels)
    vox_mask = get_voxel_mask(subject=args.subject, category_key=category_key,
                              num_voxels=num_voxels)

    vox_mask_count = cast(int, vox_mask.sum().item()) # number of voxels in the mask

    print('category:', category_key)
    print('seed:', args.seed)

    SAVE_DIR = Path(SCRATCH / category_key)
    if downsample_fmri is not None:
        SAVE_DIR = SAVE_DIR / f"downsampled"
    if args.save_suffix:
        SAVE_DIR = SAVE_DIR / args.save_suffix
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print('Saving to SAVE_DIR:', SAVE_DIR)

    run_config["vox_mask"] = vox_mask
    run_config["training_data_amt"] = all_stories

    # Initialize wandb
    run_name = (f"downsample{downsample_fmri}" if downsample_fmri is not None else "") + f"{args.subject}_layer{layer_num}_bottle{bottle_size}_lora{args.lora_rank}_cat{category_key}" + (f"_{args.save_suffix}" if args.save_suffix else "")
    wandb_settings = wandb.Settings(project="brainwavlm",
                                    run_name=run_name)
    wandb_run = wandb.init(settings=wandb_settings, config=run_config)
    run_config['wandb_metadata'] = {'run_id': wandb_run.id, 'run_name': wandb_run.name, 'project': wandb_run.project}

    # Load original WavLM model
    model = AutoModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True).to(device)
    if args.reinit_layers:
        print('Re-initializing layers 10 through 12...')
        for layer_idx in range(9,11+1): # 0-indexed
            model.encoder.layers[layer_idx].apply(model._init_weights)

    # bottleneck layer
    ndelays = args.ndelays
    if bottle_size > 0:
        encoding_out = nn.Sequential(
            nn.Linear(768*ndelays, bottle_size), # NOTE: hardcoded for WavLM-base's 768-dim hidden state; update if using a different model
            nn.Linear(bottle_size, vox_mask_count)).to(device)
        nn.init.xavier_uniform_(encoding_out[0].weight)
        nn.init.xavier_uniform_(encoding_out[1].weight)
    else:
        encoding_out= nn.Linear(768*ndelays, vox_mask_count).to(device)
        nn.init.xavier_uniform_(encoding_out.weight)

    # Instantiate LoRA parameters
    if args.lora_rank > 0:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                 target_modules= ['k_proj', 'v_proj', 'q_proj'],
                                 inference_mode=False, r=args.lora_rank, lora_alpha=4, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
    model_pretrained_state = model.state_dict().copy() # pretrained weights. Later, only save weights that change
    model = nn.DataParallel(model) # must be done after LoRA

    # filter for trainable parameters (i.e. LoRA parameters only, if using LoRA)
    model_param_names = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [{"params": model_param_names, "lr": lora_lr},
         {"params": encoding_out.parameters(), "lr": bottle_lr}],
        lr=5e-4) # NOTE: LR is overridden by param groups

    loss_fn = loss_fns[args.loss]
    best_loss = 10
    losses = []
    encperfs = []

    ndelays = 4
    # trying to increase context length
    tr_block_size = 10 # we really want to maximize this number. Increases model's ability to learn across long timescales
    test_tr_block_size = 500 # ideally we can fit an entire test story in 1 block
    featext_batch_size = 900 # maximize this when possible. Purely for efficiency
    delays = list(range(1, ndelays+1))
    context_sz_sec = 3.75
    chunk_sz_sec = 0.25
    #"""
    if downsample_fmri is not None:
        # Scale training set to have the correct number of time points
        tr_block_size = int(tr_block_size / downsample_fmri)

    run_config.update({'ndelays': ndelays, 'delays': delays, 'tr_block_size': tr_block_size,
                       'test_tr_block_size': test_tr_block_size, 'featext_batch_size': featext_batch_size,
                       'context_sz_sec': context_sz_sec, 'chunk_sz_sec': chunk_sz_sec})

    # Hidden state preprocessing & normalization modules
    # `InstanceNorm1d` expects a tensor of (features, time), whereas usually we do (time, features)
    feature_norm = nn.InstanceNorm1d(num_features=768, track_running_stats=True) # NOTE: track_running_stats=True is unverified; may be worth re-checking against track_running_stats=False
    delay_features = DelayFeatures(delays=delays)
    preprocess_features = nn.Sequential(Permute([1, 0]), feature_norm, Permute([1, 0]),
                                        delay_features).to(device)

    # Compile/JIT all basic layers
    if torch.cuda.get_device_capability(device) >= (7, 0):
        preprocess_features.compile()
        encoding_out.compile()
    else:
        # GTX 1080 Ti's don't support Triton
        preprocess_features = torch.jit.script(preprocess_features)
        encoding_out = torch.jit.script(encoding_out)

    # Save all hyperparameters & configuration
    torch.save(run_config, SAVE_DIR / "finetuning-run-config.pyt")

    start_time = time.time()

    # Load all stimuli, slide windows for feature extraction, then create Lanczos matrices.
    story_wavs: dict[str, torch.Tensor] = {}
    story_wav_snippet_bounds: dict[str, torch.Tensor] = {} # in samples! shape: (num_snippets, 2)
    lanczos_mats: dict[str, torch.Tensor] = {} # shape: (num_trs, num_snippets)
    response_downsample_lanczos_mats: dict[str, torch.Tensor] = {} # shape: (num_downsampled_trs, num_trs)
    for story in tqdm(all_stories, desc='loading stimuli'):
        story_wav, sr = torchaudio.load(f"processed_stimuli/{story}.wav")
        story_wav = story_wav.mean(0) # average across channels
        story_wavs[story] = story_wav
        assert sr == TARGET_SAMPLE_RATE, f"resample audio for story {story}!"

        wav_snippet_bounds = slide_window(story_wav, window_size=int((context_sz_sec+chunk_sz_sec)*TARGET_SAMPLE_RATE),
                                          window_stride=int(chunk_sz_sec*TARGET_SAMPLE_RATE), min_length_samples=400,
                                          require_full_context=False)
        story_wav_snippet_bounds[story] = wav_snippet_bounds

        wav_snippet_bounds_sec = wav_snippet_bounds / TARGET_SAMPLE_RATE

        story_tr_times = tr_times[story][0].trtimes[10:-5]

        if downsample_fmri is not None:
            downsampled_tr_times = story_tr_times[::int(downsample_fmri)]
            response_downsample_lanczos_mat = torch.from_numpy(
                get_lanczos_mat_fir(story_tr_times,
                                    downsampled_tr_times)).float()
            response_downsample_lanczos_mat[response_downsample_lanczos_mat < 1e-10] = 0
            response_downsample_lanczos_mats[story] = response_downsample_lanczos_mat.to_sparse_csr()
            story_tr_times = downsampled_tr_times # store stimulus lanczos matrices in terms of *downsampled* TRs

        lanczos_mat = torch.from_numpy( # shape: (num_trs, num_snippets)
            get_lanczos_mat_fir(wav_snippet_bounds_sec[:,1].numpy(),
                                story_tr_times)).float()
        lanczos_mat[lanczos_mat < 1e-10] = 0 # these terms are insignificant, so it could save a few forward passes later
        lanczos_mats[story] = lanczos_mat.to_sparse_csr().to(model.device_ids[0])

        # Sanity check: every TR should have _some_ corresponding stimulus.
        # For some reason, some stimuli were truncated during preprocessing/resampling.
        if not lanczos_mat.to_dense().any(1).all():
            print("ERROR: missing stimulus for response in story", story)
            print("Number of snippets:", wav_snippet_bounds_sec.shape[0])
            print("Number of TRs:", len(story_tr_times))
            print("Max TR time:", story_tr_times[-1])
            print("Max snippet time:", wav_snippet_bounds_sec[-1, 1])
            exit(1)

    # After training on the given num. of epochs, run one validation pass
    steps = 0
    epoch_stories = val_stories + train_stories # list of stories for one epoch. val set first.
    epoch_len_stories = len(epoch_stories)
    for es, story in enumerate(tqdm((epoch_stories * run_config['num_epochs']) + val_stories, desc='stories')):
        epoch = es // epoch_len_stories
        if es % epoch_len_stories == 0:
            print(f"\nStarting epoch {epoch}...\n")

        loss = 0

        story_wav = story_wavs[story]
        wav_snippet_bounds = story_wav_snippet_bounds[story] # stimulus snippets for this audio clip
        lanczos_mat = lanczos_mats[story]
        story_len = lanczos_mat.shape[0]

        if story in val_stories:
            with torch.no_grad():
                model.eval()
                for tr_start in range(0, story_len+1, test_tr_block_size): # iterates over all tr_starts
                    tr_end = min(tr_start + tr_block_size, story_len)
                    if tr_end - tr_start < 2: continue

                    predicted_responses = output_block_responses(model, layer_num=layer_num, story=story, tr_start=tr_start, tr_end=tr_end, target_token=target_token, mean_pool=mean_pool) # (num_trs, num_voxels)

                    # Compare against ground truth
                    tr_block_responses = torch.from_numpy(Presps[story].astype(np.float32))
                    if downsample_fmri is not None:
                        tr_block_responses = response_downsample_lanczos_mats[story] @ tr_block_responses
                    tr_block_responses = tr_block_responses[tr_start:tr_end] # shape: (num_trs, num_voxels)
                    responses_nan_mask = torch.any(torch.isnan(tr_block_responses), dim=0)
                    # Compute loss over the set of voxels in vox_mask
                    loss = loss_fn(
                        predicted_responses[:,~(responses_nan_mask[vox_mask])],
                        tr_block_responses[:,vox_mask&~responses_nan_mask].to(predicted_responses.device))

                    # correlation across time, average across voxels
                    encperf = -temporal_loss(
                        predicted_responses[:,~(responses_nan_mask[vox_mask])],
                        tr_block_responses[:,vox_mask&~responses_nan_mask].to(predicted_responses.device)).item()
                    encperfs.append(encperf)
                    losses.append(loss.item())

                    del predicted_responses

                    ### Early Stopping
                    if best_loss > loss or story in val_stories:
                        # saving to scratch dir
                        print("Val loss:", story, loss.item())
                        print("Val encperf:", story, encperf)
                        if story == val_stories[-1]:
                            torch.save(losses, SAVE_DIR / f"losses_epoch_{epoch}.pyt")
                            torch.save(encperfs, SAVE_DIR / f"encperfs_epoch_{epoch}.pyt")

                            # Log val loss & average across stories
                            this_epoch_val_loss = np.mean(losses[-len(val_stories):])
                            this_epoch_val_encperf = np.mean(encperfs[-len(val_stories):])
                            wandb.log({f"val_loss": this_epoch_val_loss, "epoch": epoch}, step=steps)
                            wandb.log({f"val_encperf": this_epoch_val_encperf, "epoch": epoch}, step=steps)

                            # Only save parameters that are backpropped or have changed
                            model_state = {k: v for k, v in model.module.state_dict().items() if \
                                model.module.get_parameter(k).requires_grad or (not torch.allclose(v, model_pretrained_state[k]))}
                            torch.save(model_state, SAVE_DIR / f"model_epoch_{epoch}.pyt")
                            del model_state
                            if isinstance(preprocess_features[1], torch.nn.InstanceNorm1d):
                                torch.save(preprocess_features[1].state_dict(), SAVE_DIR / f"instancenorm_epoch_{epoch}.pyt")
                            torch.save(encoding_out.state_dict(), SAVE_DIR / f"encodingout_epoch_{epoch}.pyt")
                        best_loss = loss

                    del loss

        else:
            model.train()
            for tr_start in tqdm(range(0, story_len+1, tr_block_size), desc='train_tr_block', leave=False): # iterates over all tr_starts
                optimizer.zero_grad()
                tr_end = min(tr_start + tr_block_size, story_len)
                # Need 2 TRs to z-score across time
                if tr_end - tr_start < 2: continue

                predicted_responses = output_block_responses(model, layer_num=layer_num, story=story, tr_start=tr_start, tr_end=tr_end, target_token=target_token, mean_pool=mean_pool) # (num_trs, num_voxels)

                # Compare against ground truth
                with torch.no_grad():
                    tr_block_responses = torch.from_numpy(Rresps[story].astype(np.float32))
                    if downsample_fmri is not None:
                        tr_block_responses = response_downsample_lanczos_mats[story] @ tr_block_responses
                    tr_block_responses = tr_block_responses[tr_start:tr_end] # shape: (num_trs, num_voxels)
                responses_nan_mask = torch.any(torch.isnan(tr_block_responses), dim=0)
                # Compute loss over the set of voxels in vox_mask
                loss = loss_fn(
                    predicted_responses[:,~(responses_nan_mask[vox_mask])],
                    tr_block_responses[:,vox_mask&~responses_nan_mask].to(predicted_responses.device))

                loss.backward()
                if run_config['lora_rank'] > 0:
                    assert model.module.base_model.model.encoder.layers[0].attention.k_proj.lora_A.default.weight.grad is not None, \
                "gradients not found for LoRA. Make sure to apply `patches/transformers+4.33.2.patch`"

                if torch.isnan(loss).any() :
                    print("training loss is NaN", story)

                # NOTE: stepping at each TR block, not story
                optimizer.step()
                steps += 1

                with torch.no_grad():
                    # Log training loss
                    wandb.log({"train_loss": loss.item(), "epoch": epoch, "story": story}, step=steps)

                del predicted_responses, loss

    print("Finished training!")
    print("Saved to:", SAVE_DIR)
