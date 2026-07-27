# Abstraction Induces the Brain Alignment of Language and Speech Models

** Under construction 7/25/2026**

This repo contains code for the ICML 2026 paper "Abstraction Induces the Brain Alignment of Language and Speech Models" by Emily Cheng, Aditya Vaidya, and Richard Antonello. 

Please use this citation if you repurpose any code:
```
@inproceedings{
cheng2026abstraction,
title={Abstraction Induces the Brain Alignment of Language and Speech Models},
author={Emily Cheng and Aditya R. Vaidya and Richard Antonello},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=n5Ds4qbtjM}
}
```

## Step 1: Downloading the data
Data were derived from the following open-source datasets:

- The Pile sample https://huggingface.co/datasets/NeelNanda/pile-10k; license: bigscience-bloom-rail-1.0
- LibriSpeech ASR https://huggingface.co/datasets/openslr/librispeech asr; license: cc by 4.0
- Probing tasks (Conneau et al., 2018) https://github.com/facebookresearch/SentEval/tree/main/data/probing; license: bsd
- fMRI (LeBel et al., 2023) https://openneuro.org/datasets/ds003020/versions/2.0.0; license: cc0
    - We use subjects UTS01, UTS02, and UTS03 for the analysis.
- ECoG (Zada et al., 2025) https://openneuro.org/datasets/ds005574; license: cc0

## Step 2: Intrinsic Dimension computation
To compute the layerwise intrinsic dimension on the last-token representation, use the following:
1. Run `scripts/intrinsic_dimension/extract_final_representations.py` if doing LLMs, and `scripts/intrinsic_dimension/extract_final_reps_audio.py` if audio.
    - This will feed The Pile through the LLMs (LibriSpeech if audio), and save layerwise representations to disk.
    - These representations will be used in the next step.
    - Example usage: `python3 scripts/intrinsic_dimension/extract_final_representations.py [MODEL] 1 data/pile_subsample.txt [YOUR PATH] [CKPT_STEP]`
2. Run `scripts/intrinsic_dimension/intrinsic_dimension.py` to save the layerwise IDs to a json file.
    - Example usage: `python3 scripts/intrinsic_dimension/pca_id.py --model [MODEL] --random_seed [RS] --method [METHOD] --step [STEP]`
3. If computing the ID using the GRIDE estimator, an additional scale analysis is needed. To do so, for each layer plot the ID wrt scale and choose the ID where the ID vs. scale plot visually plateaus. Alternatively, find the critical point(s) of the ID vs. scale plot programmatically. This choice will make the ID estimate robust to scale. (Plotting code not included, but final scales are in `gride_scales.txt`).

For further reference on how to compute ID using GRIDE, see `https://github.com/chengemily1/id-llm-abstraction`.

## Step 3: Probing
(TODO Emily)

## Step 4: Encoding models
(TODO Emily+RJ)

## Step 5: Random Fourier Features ablation
This is a control analysis. Instead of an LLM, each word is mapped to a fixed random vector and pushed through a Gaussian random Fourier feature map. Sweeping the RFF output dimensionality varies the intrinsic dimension of the feature space without introducing any semantic abstraction, which lets us ask whether encoding performance tracks ID on its own.

All three stages live in `scripts/rff_ablation/rff_ablation.py`, which needs `pip install random-fourier-features-pytorch dadapy`.

The encoding models reuse the setup from the encoding model scaling laws repo, `https://github.com/HuthLab/encoding-model-scaling-laws`:
- Put its `ridge_utils/` on your `PYTHONPATH`.
- Download `grids_huge.jbl`, `trfiles_huge.jbl`, and the per-subject `UTS0*_responses.jbl` fMRI responses from the Box folder linked in its README, https://utexas.box.com/v/EncodingModelScalingLaws. These response files are already trimmed by 10 TRs at the start and 5 at the end, and the stimulus is sliced to match.

1. Run the `sweep` stage to fit voxelwise ridge encoding models across a sweep of RFF dimensionalities. It also saves the feature maps, so the next step measures the ID of exactly the maps that were fit.
    - Example usage: `python3 scripts/rff_ablation/rff_ablation.py sweep --subject UTS02`
    - Writes `UTS02_RFF_sweep_results_plus_featuremap.jbl`.
2. Run the `id` stage to compute the GRIDE intrinsic dimension of those feature maps on the Pile subsample.
    - Example usage: `python3 scripts/rff_ablation/rff_ablation.py id --subject UTS02`
    - Writes `id_gride_rff_pile_seed32_n10000_UTS02.json`.
3. Run the `plot` stage to plot encoding performance and intrinsic dimension against RFF dimensionality, averaged over subjects.
    - Example usage: `python3 scripts/rff_ablation/rff_ablation.py plot --subjects UTS02 UTS03`

Steps 1 and 2 are run once per subject; the paper uses UTS02 and UTS03.

## Step 6: Braintuning on WavLM
(Aditya)



