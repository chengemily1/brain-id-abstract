from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pickle
import numpy as np
from dadapy import data
import argparse
import json

# Emily: REPLACE THESE WITH THE CORRECT PATH IF NEEDED
LLM_FORMAT = '{}_step{}.pickle' # model, step
PYTHIA_CHECKPOINT_FORMAT = 'hidden_pile_sane_pythia_reps_step{}.pickle' # data, model, step
AUDIO_FORMAT = '{}_librispeech_rs_{}.npy'

parser = argparse.ArgumentParser(description='ID computation')

# Data selection
parser.add_argument('--model', type=str, default="EleutherAI/pythia-140m")
parser.add_argument('--dataset', type=str, default='wikitext', choices=['wikitext', 'bookcorpus', 'pile'])
parser.add_argument('--method', type=str, default='gride')
parser.add_argument('--mode', type=str, default='sane', choices=['sane', '5', 'shuffled', 'random', '128'])
parser.add_argument('--random_seed', type=int, default=32)
parser.add_argument('--step', type=int, default=143000)
args = parser.parse_args()

np.random.seed(args.random_seed)


if 'whisper' not in args.model and 'wavlm' not in args.model:
    filepath = LLM_FORMAT.format(args.model, args.step)
    with open(filepath, 'rb') as f:
        reps = pickle.load(f) # dict {layer_idx: list of reps}

    # SELECT THE SUBSET
    subset_idx = args.random_seed % 5

    reps = {k: np.array(reps[k])[subset_idx * 10000: (subset_idx + 1) * 10000,:] for k in reps}

elif 'whisper' in args.model or 'wavlm' in args.model:
    filepath = AUDIO_FORMAT.format(args.model, args.random_seed)
    subset_idx = args.random_seed
    reps = np.load(filepath)
    reps = {k: reps[k] for k in range(len(reps))}

# initialise the Data class
if args.method == 'pca':
    results = {'pca_id': [None for _ in reps],  'pr_id': [None for _ in reps], 'explained_var': [None for _ in reps], 'eigenspectrum': [None for _ in reps]}

    for layer, layer_reps in reps.items():
        if layer == 0:
            results['pca_id'][0] = None
            results['pr_id'][0] = None
            continue
        pca = PCA()
        pca.fit(layer_reps)
        explained_variances = pca.explained_variance_
        results['pca_id'][int(layer)] = int(np.sum(np.cumsum(pca.explained_variance_ratio_) < 0.99))
        results['pr_id'][int(layer)] = float(sum(explained_variances)**2 / sum(explained_variances ** 2))
        results['eigenspectrum'][int(layer)] = [float(v) for v in list(explained_variances)]
        results['explained_var'][int(layer)] = 0.99
elif args.method == 'gride':
    results = {layer: {'id': [],
                       'err': [],
                       'r': []
                       } for layer in reps}
    for layer, layer_reps in reps.items():
        _data = data.Data(layer_reps)
        _data.remove_identical_points()

        # estimate ID
        ids_scaling, ids_scaling_err, rs_scaling = _data.return_id_scaling_gride(range_max = 2**13)
        results[layer]['r'] = [float(n) for n in rs_scaling.tolist()]
        results[layer]['err'] = [float(e) for e in ids_scaling_err.tolist()]
        results[layer]['id'] = [float(i) for i in ids_scaling.tolist()]

# Save dictionary as JSON
save_path = f'{args.model}_step{args.step}_id_pile_rs{subset_idx}_{args.method}.json'

print(f'saved to {save_path}')

with open(save_path, 'w') as json_file:
    json.dump(results, json_file)
