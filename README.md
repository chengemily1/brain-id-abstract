# Abstraction Induces the Brain Alignment of Language and Speech Models

** Under construction **

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

For further reference on how to compute ID using GRIDE, see `https://github.com/chengemily1/id-llm-abstraction`.

## Step 3: Probing

## Step 4: 


