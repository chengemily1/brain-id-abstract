#### Dependencies ####
import numpy as np
import logging
import sys
import time
import joblib
import torch
from cvxopt import matrix, solvers # Only necessary for the stacked model.
from pydiffmap import diffusion_map as dm
import time
import argparse
import os
from sklearn.metrics import r2_score
import pdb
from tqdm import tqdm
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("hi")

# Repository imports
from ridge_utils.ridge import bootstrap_ridge_with_y_projection
import ridge_utils.npp
from ridge_utils.util import make_delayed
from ridge_utils.dsutils import make_word_ds

from manifold_utils.projection import down_project, get_up_projection_map, get_up_projections_torch, squish, squish_test
from manifold_utils.algorithms import *
from manifold_utils.constants import *
from manifold_utils.feature_extraction import FeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--y_projection", type=str, default="pca", choices=['pca', 'dm', 'I']) # I is identity projection
    parser.add_argument('--which_layers', type=str, default='single', help='feature selection algo', choices=['single', 'all', 'ipca', 'every_other', 'idCorr'])
    parser.add_argument('--target_x_dim', type=int, default=1000)
    parser.add_argument("--n_layers", type=int, default=1, help="How many layers we want to include from the model")
    parser.add_argument("--seed_layer", type=int, default=9, help="the first layer to include (only layer is n_layers=1)")
    parser.add_argument("--n_evecs", type=float, default=1000)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--subject', type=int, choices=[2,3])
    parser.add_argument("--autoencoder_epochs", type=int, default=1000)
    parser.add_argument("--autoencoder_lr", type=float, default=1e-3)

    args = parser.parse_args()

    if args.which_layers == 'all' or args.which_layers == 'every_other':
        args.n_layers = 0

    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)

    # These files are located in the story_data folder of the Box
    resp_dict = joblib.load(f"UTS0{args.subject}_responses.jbl") # Located in story_responses folder

    with open('grids_cheap.txt', 'r') as f: # avoid oom
        grids = [title.strip() for title in f.readlines()]

    # We'll build an encoding model using this set of stories for this tutorial.
    test_stories = ["wheretheressmoke", 'fromboyhoodtofatherhood', 'onapproachtopluto']
    train_stories = [story for story in resp_dict.keys() if story in grids and story not in test_stories]


    if 'whisper' not in args.model and 'wavlm' not in args.model:
        grids = joblib.load("grids_huge.jbl") # Load TextGrids containing story annotations
        trfiles = joblib.load("trfiles_huge.jbl") # Load TRFiles containing TR information

        # Filter out the other stories for the tutorial
        for story in list(grids):
            if story not in (train_stories + test_stories):
                del grids[story]
                del trfiles[story]

        # Make datasequence for story
        wordseqs = make_word_ds(grids, trfiles)

        # We will extract features now
        feature_extractor = FeatureExtractor(wordseqs, args.model, train_stories, test_stories)

        # Convert back from dictionary to matrix
        print('getting features')
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # multiprocessing with tokenizer in feature_extraction
        feats = feature_extractor.get_features(args.which_layers, seed_layer=args.seed_layer) # N stories x L layers x d (previously N stories x d)
        # n_layers = feature_extractor.L_layers
        print('got the features')
    elif 'whisper' in args.model:
        features_path = f"YOUR_PATH/encoding-models/whisper-features/downsampled_featureseqs_whisper-large_layer{args.seed_layer}.jbl"
        feats = joblib.load(features_path)

    elif 'wavlm' in args.model:
        n_layers = 25

        # Load directly from file
        features_path = 'YOUR_PATH/encoding-models/wavlm-large_downsampled/layer.{}/{}.npz'

        feats = {
                story: np.load(features_path.format(args.seed_layer, story))['features'] for story in train_stories + test_stories
            }

    # Training data
    Rstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(feats[story][10:-5]
                                                       ) for story in train_stories]))

    # Test data
    Pstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(feats[story][trim_start:-trim_end]
                                                       ) for story in test_stories]))

    # Add FIR delays
    print('Adding FIR delays...')
    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)

    # Get response data
    Rresp = np.vstack([resp_dict[story] for story in train_stories]) # training Y
    Presp = np.vstack([resp_dict[story][40:] for story in test_stories]) # testing Y
    My_train = Rresp.astype(np.float32)
    My_test = Presp.astype(np.float32)

    # Get explanatory variables
    Mx_train = delRstim.astype(np.float32)
    Mx_test = delPstim.astype(np.float32)

    # Bootstrap parameters
    alphas = np.logspace(1, 4, 15) # Equally log-spaced ridge parameters between 10 and 10000.
    nboots = 3 # Number of cross-validation ridge regression runs. You can lower this number to increase speed.
    chunklen = 20
    nchunks = int(len(My_train) * 0.25 / chunklen)

    print('Computing projection maps on train data response')
    _, projection_map_y = down_project(My_train, project_type=args.y_projection, n_evecs=args.n_evecs)
    up_projection_map_y = get_up_projection_map(args, My_train, My_train, My_test, project_type=args.y_projection, projection_map_y=projection_map_y)
    print("Bootstrap ridge")

    # Use RJ's bootstrap ridge code modified to handle projection
    wt, corr, best_alpha, bootstrap_corrs, valinds = bootstrap_ridge_with_y_projection(
                                                        Mx_train, My_train, Mx_test, My_test,
                                                        alphas, nboots, chunklen, nchunks,
                                                        up_projection_map_y, projection_map_y,
                                                        y_projection=args.y_projection,
                                                    )
    if type(best_alpha) != list:
        best_alpha = best_alpha.tolist()
    if type(valinds) != list:
        valinds = valinds.tolist()
    if type(bootstrap_corrs) != list:
        bootstrap_corrs = bootstrap_corrs.squeeze() # 1 x nvox x 1
        bootstrap_corrs = bootstrap_corrs.tolist()
    if type(corr) != list:
        corr = corr.tolist()

    results = {
        'params': vars(args),
        'corr': list(corr), # nvox
        'bscorrs': list(bootstrap_corrs),
        'val_indices': list(valinds),
        'alphas': best_alpha # scalar
    }
    model_str = args.model.split('/')[-1]

    # Save
    save_dir = f'YOUR_PATH/encoding-models/results/{model_str}/UTS0{args.subject}'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/results_{args.which_layers}_n_layers_{args.n_layers}_seed_layer_{args.seed_layer}_x_rank_{args.target_x_dim}_y_rank_{args.n_evecs}_{args.y_projection}_ridge.json', 'w') as f:
        json.dump(results, f)
