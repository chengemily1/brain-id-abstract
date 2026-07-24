import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import WavLMModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, AutoProcessor, AutoModel
import glob
import numpy as np
import joblib
from datasets import load_dataset
import pdb
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser(description='save reps')

# Data selection
parser.add_argument('--model', type=str, default="wrice/wavlm-base-plus-weight-norm-fix")#"wrice/wavlm-large-weight-norm-fix")
parser.add_argument('--dataset', type=str, default='wikitext', choices=['librispeech', 'surprisal_train', 'wikitext', 'bookcorpus', 'pile'])
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--finetuned_on', type=str, default=None, choices=['UTS02', 'UTS03', 'UTS01', 'llama'])
parser.add_argument('--debug', type=int, default=0)
args = parser.parse_args()

np.random.seed(32)
device = 'cuda:0' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'

# Load the LibriSpeech dataset (you can specify "clean" or "other" subsets as well)
librispeech = load_dataset("librispeech_asr", "clean", split="train.360")

if 'wavlm' in args.model:
    model = WavLMModel.from_pretrained(args.model)
    size = args.model[len('wrice/wavlm-'):-len('-weight-norm-fix')]
    args.model = f'microsoft/wavlm-{size}'

    # model = AutoModel.from_pretrained(args.model, output_hidden_states=True)
    
    if args.finetuned_on is not None:
        model.load_state_dict(torch.load(f'/home/echeng/encoding-models/checkpoints/{args.finetuned_on}/model_merged.pyt' , map_location=model.device))
        
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
elif 'whisper' in args.model:
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

model.config.output_hidden_states = True
model.to(device)
model.eval()

max_length = 30 * 16000  # 30 seconds * 16000 samples per second

# SPLIT EACH LIBRISPEECH CLIP INTO AUDIO CHUNKS. GOAL: 50k chunks
audio_chunks = []

# Take 50k articles
for i, clip in tqdm(enumerate(librispeech)):
    if args.debug and i == 10000: break
    if i == 500000: break 
    audio_ = clip['audio']['array']  # Get audio array from the dataset
    sampling_rate = clip['audio']['sampling_rate']

    # Resample audio to 16kHz if necessary
    if sampling_rate != 16000:
        audio_ = torchaudio.transforms.Resample(sampling_rate, 16000)(torch.tensor(audio_))

    audio_chunks.extend([audio_[i:i + max_length] for i in range(0, len(audio_), max_length)])

# SAMPLE 10k audio chunks
random.shuffle(audio_chunks)
audio_chunks = audio_chunks[10000 * args.random_seed : 10000 * args.random_seed + 10000]

# COLLECT THE EMBEDDINGS
print('check model layers')

n_layers = model.config.encoder_layers if 'whisper' in args.model else model.config.num_hidden_layers
embeddings_list = [[] for _ in range(n_layers)] # one tensor per layer

with torch.no_grad():
    for chunk in tqdm(audio_chunks):
        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features if 'whisper' in args.model else inputs.input_values  # Shape: (1, feature_size, 3000)
        attention_mask = inputs.attention_mask  # Shape: (1, 3000)

        if 'whisper' in args.model:
            attention_mask_downsampled = attention_mask[:, ::2]  # whisper idiosyncrasy
            last_token_index = int(torch.sum(attention_mask_downsampled))-1
        elif 'wavlm' in args.model:
            last_token_index = -1

        if 'whisper' in args.model:
            expected_seq_length = 3000 

            assert attention_mask.shape[1] == expected_seq_length, \
                f"Unexpected attention_mask length: {attention_mask.shape[1]}"
            model_ = model.model.encoder
        elif 'wavlm' in args.model:
            model_ = model

        # Pass through the encoder to get hidden states
        input_features = input_features.to(device)
        attention_mask = attention_mask.to(device)
        encoder_outputs = model_(
            input_features,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = encoder_outputs.hidden_states 
        # Process embeddings for each layer
        for i, layer in enumerate(hidden_states):
            if i == 0: continue
            embeddings_list[i-1].append(layer[:,last_token_index,:].cpu())

        del hidden_states
        del encoder_outputs

embeddings_list = [torch.cat(layer, dim=0).unsqueeze(0) for layer in embeddings_list]
embeddings = torch.cat(embeddings_list, dim=0).numpy() # N layers x N datapoints x D

if args.finetuned_on is not None:
    args.model = args.model + f'_finetuned_{args.finetuned_on}'

np.save(f'{args.model}_librispeech_rs_{args.random_seed}.npy', embeddings)