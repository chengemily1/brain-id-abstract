#!/usr/bin/env python3

"""
Feature extraction for ASR models supported by Hugging Face.

The feature extraction method is different than for s3prl, since we don't set
our own hooks.
"""

import argparse
import collections
import copy # for freezing specific parts of networks during randomization
import itertools
import json
import os
import operator
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import torch
import torchaudio
from transformers import AutoModel, AutoModelForPreTraining, PreTrainedModel,\
                         AutoFeatureExtractor, WhisperModel

# Resample to this sample rate. 16 kHz is used by most models (but not all! like
# MERT, which is 24 kHz).
TARGET_SAMPLE_RATE = 16000

# window_size` and `window_stride` are in *samples*, not seconds
def slide_window(wav: torch.Tensor, window_size: int, window_stride: int,
                 min_length_samples: int,
                 require_full_context: bool = False) -> torch.Tensor:

    # `snippet_ends` has the last (exclusive) sample for each snippet
    snippet_ends = []
    if not require_full_context:
        # Add all snippets that are _less_ than the total input size
        # (context+chunk)
        snippet_ends.append(torch.arange(window_stride, window_size, window_stride))

    # Add all snippets that are exactly the length of the requested input
    # (`Tensor.unfold` is basically a sliding window).
    if wav.shape[0] >= window_size:
        # `unfold` fails if `wav.shape[0]` is less than the window size.
        snippet_ends.append(
            torch.arange(wav.shape[0]).unfold(0, window_size, window_stride)[:,-1]+1
        )

    snippet_ends = torch.cat(snippet_ends, dim=0) # shape: (num_snippets,)

    if snippet_ends.shape[0] == 0:
        raise ValueError(f"No snippets possible! Stimulus is probably too short ({wav.shape[0]} samples). Consider reducing context size or setting `require_full_context=True`")

    # 2-D array where `[i,0]` and `[i,1]` are the start and end, respectively,
    # of snippet `i` in samples. Shape: (num_snippets, 2)
    snippet_times = torch.stack([torch.maximum(torch.zeros_like(snippet_ends),
                                               snippet_ends-window_size),
                                 snippet_ends], dim=1)

    snippet_times = snippet_times[(snippet_times[:,1] - snippet_times[:,0]) >= min_length_samples]

    snippet_length_samples = snippet_times[:,1] - snippet_times[:,0] # shape: (num_snippets,)
    if require_full_context:
        assert all(snippet_length_samples == snippet_length_samples[0]), "uneven snippet lengths!"
        snippet_length_samples = snippet_length_samples[0]
        assert snippet_length_samples.ndim == 0

    # [:, 0] is the start, [:, 1] is the end (excl.) of each snippet.
    return snippet_times # shape: (num_snippets, 2).

# Batch the output of sliding_window into batches of the same length.
# Can return a `list` of tensors with shape (T, 2) if
# `require_full_context=False`, otherwise a single tensor with shape (B, T, 2)
def batch_snippets(snippet_times: torch.Tensor, batchsz: int,
                   require_full_context: bool=False) -> list[torch.Tensor]:
    # Set up the iterator over batches of snippets
    if require_full_context:
        # This case is simpler, so handle it explicitly
        snippet_batches = list(snippet_times.T.split(batchsz, dim=1))
    else:
        # First, group the snippets that are of different lengths.
        snippet_length_samples = snippet_times[:,1] - snippet_times[:,0] # shape: (num_snippets,)
        snippets_by_length = snippet_times.tensor_split(torch.where(snippet_length_samples.diff() != 0)[0]+1, dim=0)
        # Then, split any groups that are too big to fit into the given
        # batch size.
        snippet_batches: List[torch.Tensor] = [] # type: ignore
        for batch in snippets_by_length:
            # split, *then* transpose
            if batch.shape[0] > batchsz:
                snippet_batches += batch.T.split(batchsz,dim=1)
            else:
                snippet_batches += [batch.T]

    return snippet_batches

# NOTE: does not use a HF FeatureExtractor on the inputs by default (pass
# feature_extractor= explicitly if needed). This makes a difference for some
# models (e.g. Hubert) but not others (e.g. WavLM).
def extract_features_hf(model: PreTrainedModel, model_config: dict, wav: torch.Tensor,
                        chunksz_sec: float, contextsz_sec: float,
                        num_sel_frames = 1, frame_skip = 5, sel_layers: Optional[List[int]]=None,
                        batchsz: int = 1,
                        return_numpy: bool = True, move_to_cpu: bool = True,
                        disable_tqdm: bool = False, feature_extractor=None,
                        sampling_rate: int = TARGET_SAMPLE_RATE, require_full_context: bool = False,
                        stereo: bool = False, target_token: int = -1):
    assert (num_sel_frames == 1), f"'num_sel_frames` must be 1 to ensure causal feature extraction, but got {num_sel_frames}. "\
        "This option will be deprecated in the future."
    if stereo:
        raise NotImplementedError("stereo not implemented")
    else:
        assert wav.ndim == 1, f"input `wav` must be 1-D but got {wav.ndim}"
    if return_numpy: assert move_to_cpu, "'move_to_cpu' must be true if returning numpy arrays"
    target_sample_rate = feature_extractor.sampling_rate if (feature_extractor is not None) else TARGET_SAMPLE_RATE

    # Whisper needs special handling
    is_whisper_model = isinstance(model, WhisperModel)

    # Remove snippets that are not long enough. (Seems easier to filter
    # after generating the snippet bounds than handling it above in each case)
    if 'min_input_length' in model_config:
        # this is stored originally in **samples**!!!
        min_length_samples = model_config['min_input_length']
    elif 'win_ms' in model.config:
        min_length_samples = model.config['win_ms'] / 1000. * target_sample_rate
    else:
        raise ValueError('Model has no minimum input length')

    # Compute chunks & context sizes in terms of samples & context
    # This muset happen *before* resampling to the model's sampling rate, since
    # we use these variables to window the stimulus.
    chunksz_samples = int(chunksz_sec * sampling_rate)
    contextsz_samples = int(contextsz_sec * sampling_rate)

    snippet_times = slide_window(wav=wav, window_size=chunksz_samples+contextsz_samples, window_stride=chunksz_samples,
                                 min_length_samples=min_length_samples,
                                 require_full_context=require_full_context)
    snippet_times_sec = snippet_times / sampling_rate # snippet_times, but in sec.

    frame_len_sec = model_config['stride'] / target_sample_rate # length of an output frame (sec.); used if num_sel_frames>1

    """
    # Pre-allocate the sliced wav matrix, which we'll reuse for all batches. It
    # might be faster than stacking the snippets within a batch.
    # (But, turns out in practice it's worse.)
    batched_wav_in = torch.empty((batchsz, snippet_length_samples), dtype=wav.dtype, device=wav.device,
                                 requires_grad=wav.requires_grad)
    """

    snippet_iter = batch_snippets(snippet_times=snippet_times, batchsz=batchsz,
                                  require_full_context=require_full_context)
    if not disable_tqdm:
        snippet_iter = tqdm(snippet_iter, desc='snippet batches', leave=False)
    snippet_iter = enumerate(snippet_iter)

    module_features = collections.defaultdict(list)
    out_features = [] # the final output of the model
    times = [] # times are shared across all layers

    # Iterate with a sliding window. stride = chunk_sz
    for batch_idx, (snippet_starts, snippet_ends) in snippet_iter:
        if ((snippet_ends - snippet_starts) < (contextsz_samples + chunksz_samples)).any() and require_full_context:
            raise ValueError("This shouldn't happen with require_full_context")

        # If we don't have enough samples, skip this chunk.
        if (snippet_ends - snippet_starts < min_length_samples).any():
            print('If this is true for any, then you might be losing more snippets than just the offending (too short) snippet. Consider increasing the input (chunk or context) to the model.')
            assert False

        # Construct the input waveforms for the batch
        batched_wav_in_list = []
        for snippet_start, snippet_end in zip(snippet_starts, snippet_ends):
            batched_wav_in_list.append(wav[snippet_start:snippet_end])
        batched_wav_in = torch.stack(batched_wav_in_list, dim=0)

        # The final batch may be incomplete if batchsz doesn't evenly divide
        # the number of snippets.
        if (snippet_starts.shape[0] != batched_wav_in.shape[0]) and (snippet_starts.shape[0] != batchsz):
            batched_wav_in = batched_wav_in[:snippet_starts.shape[0]]

        # Take the last 1 or 2 activations, and time-wise put it at the
        # end of chunk.
        if target_token != -1:
            assert num_sel_frames == 1, "Only num_sel_frames=1 is supported when using target_token != -1"
            assert target_token >= 0, "target_token must be non-negative"
            output_inds = np.array([target_token])
        else:
            output_inds = np.array([-1 - frame_skip*i for i in reversed(range(num_sel_frames))])

        # Use a pre-processor if given (e.g. to normalize the waveform), and
        # then feed into the model.
        if feature_extractor is not None:
            # This step seems to be NOT differentiable, since the feature
            # extractor first converts the Tensor to a numpy array, then back
            # into a Tensor.
            # If you want to backprop through the stimulus, you might have to
            # re-implement the feature extraction in PyTorch (in particular, the
            # normalization)

            if stereo: raise NotImplementedError("Support handling multi-channel audio with feature extractor")
            # It looks like most feature extractors (e.g.
            # Wav2Vec2FeatureExtractor) accept mono audio (i.e. 1-dimensional),
            # but it's unclear if they support stereo as well.

            feature_extractor_kwargs = {}
            if is_whisper_model:
                # Because Whisper auto-pads all inputs to 30 sec., we'll use
                # the attention mask to figure out when the "last" relevant
                # input was.
                features_key = 'input_features'
                feature_extractor_kwargs['return_attention_mask'] = True
            else:
                features_key = 'input_values'

            preprocessed_snippets = feature_extractor(list(batched_wav_in.cpu().numpy()),
                                                      return_tensors='pt',
                                                      sampling_rate=sampling_rate,
                                                      **feature_extractor_kwargs)
            if is_whisper_model:
                chunk_features = model.encoder(preprocessed_snippets[features_key].to(model.device))

                # Now we need to figure out which output index to use, since 2
                # conv layers downsample the inputs before passing them into
                # the encoder's Transformer layers. We can redo the encoder's
                # 1-D conv's on the attention mask to find the final output that
                # was influenced by the snippet.
                contributing_outs = preprocessed_snippets.attention_mask # 1 if part of waveform, 0 otherwise. shape: (batchsz, 3000)
                # Taking [0] works because all snippets have the same length.
                # Add the dimension back for `conv1d` to work
                contributing_outs = contributing_outs[0].unsqueeze(0)

                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+model.encoder.conv1.kernel_size).to(contributing_outs),
                                                               stride=model.encoder.conv1.stride,
                                                               padding=model.encoder.conv1.padding,
                                                               dilation=model.encoder.conv1.dilation,
                                                               groups=model.encoder.conv1.groups)
                # shape: (batchsz, 1500)
                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+model.encoder.conv2.kernel_size).to(contributing_outs),
                                                               stride=model.encoder.conv2.stride,
                                                               padding=model.encoder.conv2.padding,
                                                               dilation=model.encoder.conv2.dilation,
                                                               groups=model.encoder.conv1.groups)

                final_output = contributing_outs[0].nonzero().squeeze(-1).max()
            else:
                # sampling rates must match if not using a pre-processor
                assert sampling_rate == target_sample_rate, f"sampling rate mismatch! {sampling_rate} != {target_sample_rate}"

                chunk_features = model(preprocessed_snippets[features_key].to(model.device))
        else:
            chunk_features = model(batched_wav_in)

        # Make sure we have enough outputs.
        # NOTE: this assumes hidden state dims are ordered consistently across
        # models (via chunk_features['last_hidden_state'] rather than a
        # per-model output convention); this may not hold for every model.
        if(chunk_features['last_hidden_state'].shape[1] < (num_sel_frames-1) * frame_skip - 1):
            # NOTE: skips the whole batch rather than partially using it, even
            # if at least one output frame would have been available.
            print("Skipping:", batch_idx, "only had", chunk_features['last_hidden_state'].shape[1],
                    "outputs, whereas", (num_sel_frames-1) * frame_skip - 1, "were needed.")
            continue

        assert len(output_inds) == 1, "Only one output per evaluation is "\
            "supported for Hugging Face (because they don't provide the downsampling rate)"

        if is_whisper_model:
            assert target_token == -1, "target_token is not supported with Whisper models"
            output_inds = [final_output]

        for out_idx, output_offset in enumerate(output_inds):
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1))

            output_representation = chunk_features['last_hidden_state'][:, output_offset, :] # shape: (batchsz, hidden_size)
            if move_to_cpu: output_representation = output_representation.cpu()
            if return_numpy: output_representation = output_representation.numpy()
            out_features.append(output_representation)

            # Collect features from individual layers
            # NOTE: outs['hidden_states'] might have an extra element at
            # the beginning for the feature extractor.
            # e.g. 25 "layers" --> CNN output + 24 transformer layers' output
            for layer_idx, layer_activations in enumerate(chunk_features['hidden_states']):
                # Only save layers that the user wants (if specified)
                if sel_layers:
                    if layer_idx not in sel_layers: continue

                layer_representation = layer_activations[:, output_offset, :] # shape: (batchsz, hidden_size)
                if move_to_cpu: layer_representation = layer_representation.cpu()
                if return_numpy: layer_representation = layer_representation.numpy()

                if is_whisper_model:
                    # Leave the option open for using decoder layers in the
                    # future
                    module_name = f"encoder.{layer_idx}"
                else:
                    module_name = f"layer.{layer_idx}"

                module_features[module_name].append(layer_representation)

    out_features = np.concatenate(out_features, axis=0) if return_numpy else torch.cat(out_features, dim=0) # shape: (timesteps, features)
    module_features = {name: (np.concatenate(features, axis=0) if return_numpy else torch.cat(features, dim=0))\
                       for name, features in module_features.items()}

    assert all(features.shape[0] == out_features.shape[0] for features in module_features.values()),\
        "Missing timesteps in the module activations!! (possible PyTorch bug)"
    times = torch.cat(times, dim=0) / target_sample_rate # convert samples --> seconds. shape: (timesteps,)
    if return_numpy: times = times.numpy()

    del chunk_features # NOTE: attempt at mitigating a memory leak
    return {'final_outputs': out_features, 'times': times,
            'module_features': module_features}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stimulus_dir', type=Path,
                        default='./processed_stimuli/',
                        help="Directory with preprocessed stimuli wav's.")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--use_featext', action='store_true')
    parser.add_argument('--batchsz', type=int, default=1,
                        help='Number of audio clips to evaluate at once. (Only uses one GPU.)')
    parser.add_argument('--chunksz', type=float, default=100,
                        help="Divide the stimulus waveform into chunks of this many *milliseconds*.")
    parser.add_argument('--contextsz', type=float, default=8000,
                        help="Use these many milliseconds as context for each chunk.")
    parser.add_argument('--layers', nargs='+', type=int, help="Only save the "
                        "features from these layers. Usually doesn't speed up execution "
                        "time, but reduces total disk usage. "
                        "NOTE: only works with numbered layers (currently).")
    parser.add_argument('--full_context', action='store_true',
                        help="Only extract the representation for a stimulus if it is as long as the feature extractor's specified context (context_sz)")
    parser.add_argument('--resample', action='store_true',
                        help='Resample the stimuli to the necessary sample rate '
                        'and convert stereo to mono if needed. If this flag is '
                        'not supplied, an assertion will fail if either '
                        'condition is not met.')
    parser.add_argument('--stride', type=float,
                        help='Extract features every <n> seconds. If using --custom_stimuli, consider changing this argument. Don\'t use this for extracting story features to train encoding models (use --chunksz instead). 0.5 is a good value.')
    parser.add_argument('--pad_silence', action='store_true',
                        help='Pad short clips (less than context_sz+chunk_sz) with silence at the beginning')

    # Arguments for choosing stories
    stimulus_sel_args = parser.add_argument_group('stimulus_sel', 'Stimulus selection')
    stimulus_sel_args.add_argument('--stories', '--stimuli', nargs='+', type=str,
                                   help="Only process the given stories.")
    stimulus_sel_args.add_argument('--recursive', action='store_true',
                                   help='Recursively find .wav and .flac in the stimulus_dir.')
    stimulus_sel_args.add_argument('--custom_stimuli', type=str,
                                    help='Use custom (non-story) stimuli, stored in '
                                    '"{stimulus_dir}/{custom_stimuli}". If this flag '
                                    'is not set, use story stimuli.')
    stimulus_sel_args.add_argument('--overwrite', action='store_true',
                                   help='Overwrite existing features (default behavior is to skip)')

    # LoRA-specific arguments
    lora_args = parser.add_argument_group('lora', 'LoRA-specific arguments')
    lora_args.add_argument('--lora_path', type=Path)
    lora_args.add_argument('--checkpoint', type=Path)
    lora_args.add_argument('--save_prefix', type=Path, default=Path(''), help='user-friendly name for this LoRA checkpoint')
    lora_args.add_argument('--target_token', type=int, default=0, help='which token to extract features for (only relevant for models trained with LoRA)')

    args = parser.parse_args()

    print('Saving features to local filesystem.')

    # Load the model
    model_name = args.model
    lora_path = args.lora_path
    checkpoint_path = args.checkpoint
    assert not (lora_path is not None and checkpoint_path is not None), "You can only specify one of --lora_path or --checkpoint, not both"
    with open('model_configs.json', 'r') as f:
        model_config = json.load(f)[model_name]
        model_hf_path = model_config['huggingface_hub']
    print('Loading model', model_name, 'from the Hugging Face Hub...')
    # NOTE: no check that the Hub checkpoint is actually the original pretrained
    # model rather than some fine-tuned variant. For some models (e.g.
    # wav2vec2) you can use AutoModelForPretraining to guard against this, but
    # not all of them (e.g. hubert-base-ls960).
    model = AutoModel.from_pretrained(model_hf_path, output_hidden_states=True, trust_remote_code=True).cuda()
    feature_extractor = None
    if args.use_featext:
        if checkpoint_path is not None:
            print('Custom checkpoint_path not supported for --use_featext. Using the original feature extractor.')
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path)

    # Set up the lora model
    if lora_path is not None:
        assert model_hf_path == 'microsoft/wavlm-base-plus'
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                    target_modules= ['k_proj', 'v_proj', 'q_proj'],
                                    inference_mode=False, r=4, lora_alpha=4, lora_dropout=0.1)

        model = get_peft_model(model, peft_config)
        model.load_state_dict(torch.load(lora_path, weights_only=True, map_location=model.device), strict=False)
        lora_name = lora_path.name
        model = model.merge_and_unload() # reduce overhead for inference
    elif checkpoint_path is not None:
        # NOTE: no check that this checkpoint has the same architecture as the base model
        model = AutoModel.from_pretrained(checkpoint_path, output_hidden_states=True, trust_remote_code=True).cuda()
        lora_name = checkpoint_path.name
    else:
        lora_name = 'pretrained'

    # Re-initialize the weights, if requested (using the a specific seed, if
    # specified)
    if ('random_weights' in model_config) and model_config['random_weights']:
        print("Re-initializing model weights...")
        if 'random_seed' in model_config:
            seed = model_config['random_seed']
            # Re-seed all RNGs because some models *might* use non-pytorch RNGs
            torch.manual_seed(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)
        else:
            print("User did not specify a random seed")

        freeze_extractor = model_config.get('freeze_extractor', False)
        if freeze_extractor:
            print("Randomizing weights but NOT for the feature extractor")
            ext_state_dict = copy.deepcopy(model.feature_extractor.state_dict())

        model.apply(model._init_weights)

        if freeze_extractor:
            model.feature_extractor.load_state_dict(ext_state_dict)
            del ext_state_dict # try to save some memory

    ## Stimulus selection
    # Using CLI arguments, find stimuli and their locations.
    stories = set()

    if args.stories is not None:
        stories.update(args.stories)

    stimulus_dir = args.stimulus_dir
    assert stimulus_dir.exists(), f"Stimulus dir {str(stimulus_dir)} does not exist"
    assert stimulus_dir.is_dir(), f"Stimulus dir {str(stimulus_dir)} is not a directory"

    stimulus_paths: Dict[str, Path] = {} # map of stimulus name --> file path. We also use this as the list of stimuli

    if args.custom_stimuli: # optionally use non-story stimuli
        custom_stimuli_dir = stimulus_dir / args.custom_stimuli
        assert custom_stimuli_dir.exists(), f"dir {str(custom_stimuli_dir)} does not exist"
        stimulus_dir = custom_stimuli_dir

    # We haven't selected any stories yet, so just select all stories in the
    # stimulus directory.
    if len(stories) == 0:
        # Look for all files ending in '.flac' and '.wav'. If there are two
        # files with the same basename (i.e. without the suffix), then prefer
        # the FLAC file.
        if args.recursive:
            stimulus_glob_wav_iter = stimulus_dir.rglob('*.wav')
            stimulus_glob_flac_iter = stimulus_dir.rglob('*.flac')
        else:
            stimulus_glob_wav_iter = stimulus_dir.glob('*.wav')
            stimulus_glob_flac_iter = stimulus_dir.glob('*.flac')

        for stimulus_path in itertools.chain(stimulus_glob_wav_iter, stimulus_glob_flac_iter):
            # Use 'relative_to' to preserve directory structure when using
            # --recursive
            stimulus_name = str(stimulus_path.relative_to(stimulus_dir).with_suffix(''))
            # If stimulus already exists, overwrite the path with the
            # most recent extension
            stimulus_paths[stimulus_name] = stimulus_path
    else:
        for story in stories:
            # Find the associated sound file for each stimulus.
            # First extension found is preferred.
            for ext in ['flac', 'wav']:
                stimulus_path = stimulus_dir / f"{story}.{ext}"
                if stimulus_path.exists() and stimulus_path.is_file():
                    stimulus_paths[story] = stimulus_path
                    break

        missing_stories = set(stories).difference(set(stimulus_paths.keys()))
        if len(missing_stories) > 0:
            raise RuntimeError(f"missing stimuli for stories: " + ' '.join(missing_stories))

    assert len(stimulus_paths) > 0, "no stimuli to process!"

    # Make sure that all preprocessed stimuli exist and are readable.
    for stimulus_name, stimulus_local_path in stimulus_paths.items():
        wav, sample_rate = torchaudio.load(stimulus_local_path)
        if not args.resample:
            assert wav.shape[0] == 1, f"stimulus '{stimulus_local_path}' is not mono-channel"

    # chunk size in seconds and samples, respectively
    chunksz_sec = args.chunksz / 1000.

    # context size in terms of chunks
    assert (args.contextsz % args.chunksz) == 0, "These must be divisible"
    contextsz_sec = args.contextsz / 1000.

    # Some obsolete vars we need for the save path
    num_sel_frames = 1
    frame_skip = 5

    model_save_path = f"features_cnk{chunksz_sec:0.1f}_ctx{contextsz_sec:0.1f}_pick{num_sel_frames}_skip{frame_skip}/{model_name}"
    # Add LoRA-related path: becomes e.g. "[...]/UTS01/model_epoch_11.pyt[...]/layer_2/..."
    model_save_path = os.path.join(model_save_path, args.save_prefix, lora_name)
    if args.stride:
        # If using a custom stride length (e.g. for snippets), store in a
        # separate directory.
        model_save_path = os.path.join(model_save_path, f"stride_{args.stride}")
    if args.custom_stimuli:
        # Save custom (non-story) stimuli in their own subdirectory
        model_save_path = os.path.join(model_save_path, 'custom_stimuli', args.custom_stimuli)
    print('Saving features to:', model_save_path)

    ## Feature extraction loop
    # Go through each stimulus and save resulting features
    torch.set_grad_enabled(False) # VERY important! (for memory)
    model.eval()
    # Sort stimuli alphabetically. Allows us, in theory, to resume partial/failed jobs
    stimulus_paths = collections.OrderedDict(sorted(stimulus_paths.items(), key=lambda x: x[0]))
    target_sample_rate = feature_extractor.sampling_rate if args.use_featext else TARGET_SAMPLE_RATE
    for stimulus_name, stimulus_local_path in tqdm(stimulus_paths.items(), desc='Processing stories'):
        wav, sample_rate = torchaudio.load(stimulus_local_path)
        if not args.resample:
            # Perform checks on the original waveform
            assert wav.shape[0] == 1, f"stimulus '{stimulus_local_path}' is not mono-channel"
            assert sample_rate == target_sample_rate
        else:
            # Resample & convert to mono as needed
            if wav.shape[0] != 1: wav = wav.mean(0, keepdims=True) # convert to mono
            if sample_rate != target_sample_rate: # resample to 16 kHz (or model's SR)
                wav = torchaudio.functional.resample(wav, sample_rate, target_sample_rate)
                sample_rate = target_sample_rate

        wav.squeeze_(0) # shape: (num_samples,)

        assert sample_rate == target_sample_rate, f"Expected sample rate {target_sample_rate} but got {sample_rate}"

        features_save_path = os.path.join(model_save_path, stimulus_name)
        times_save_path = f"{features_save_path}_times"
        if not args.overwrite:
            if os.path.exists(times_save_path + '.npz'):
                print(f"Skipping {stimulus_name}, timestamps found at {times_save_path}")
                continue

        # Call a separate function to do the actual feature extraction
        extract_features_kwargs = {
            'model': model, 'model_config': model_config,
            'wav': wav.to(model.device), 'sampling_rate': sample_rate,
            'chunksz_sec': chunksz_sec, 'contextsz_sec': contextsz_sec,
            'num_sel_frames': num_sel_frames, 'frame_skip': frame_skip,
            'sel_layers': args.layers, 'feature_extractor': feature_extractor,
            'require_full_context': args.full_context or args.pad_silence,
            'batchsz': args.batchsz, 'return_numpy': False,
            'target_token': args.target_token
        }

        if args.stride:
            # Set the context_sz so that the total span length (context+chunk)
            # is the same as in non-stride mode, and so that the chunk_sz is
            # the "new" stride length.
            extract_features_kwargs['contextsz_sec'] = chunksz_sec + contextsz_sec - args.stride
            extract_features_kwargs['chunksz_sec'] = args.stride

        if args.pad_silence:
            # Pad with `context_sz` sec. of silence, so that the first
            # (non-silence) output is at time `chunk_sz`
            wav = torch.cat([torch.zeros(int(extract_features_kwargs['contextsz_sec']*target_sample_rate)), wav], dim=0)
            extract_features_kwargs['wav'] = wav.to(model.device)

        extracted_features = extract_features_hf(**extract_features_kwargs)
        out_features, times, module_features = [extracted_features[k] for k in \
                                                ['final_outputs', 'times', 'module_features']]
        del extracted_features # free up some memory after we've selected the outputs we want; maybe unnecessary

        # Remove the 'silence' we added at the beginning
        if args.pad_silence:
            times = torch.clip(times - extract_features_kwargs['contextsz_sec'], 0, torch.inf)
            assert torch.all(times >= 0), "padding is smaller than the correction (subtraction)!"
            assert torch.all(times[:,1] > 0), f"insufficient padding for require_full_context ! (times[times[:,1]<=0,1])"

        os.makedirs(os.path.dirname(features_save_path), exist_ok=True)
        np.savez_compressed(features_save_path + '.npz', features=out_features.numpy())
        np.savez_compressed(times_save_path + '.npz', times=times.numpy())

        module_save_paths = {module: os.path.join(model_save_path, module, stimulus_name) for module in module_features.keys()}

        # This is the "save name" of the module (not its original name)
        for module_name, features in module_features.items():
            features_save_path = module_save_paths[module_name]
            os.makedirs(os.path.dirname(features_save_path), exist_ok=True)
            np.savez_compressed(features_save_path + '.npz', features=features.numpy())
