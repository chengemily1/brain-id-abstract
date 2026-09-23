## Setup
Requires transformers==4.33.2 and peft==0.5.0 . Then requires applying the patches in `patches/` .

## Running
Order of scripts:
 * brain_finetune.py
 * brain_finetune_dump_features.py -- loop over epochs
 * brain_refit_linear.py -- loop over epochs
 * brain_best_linear.py
 * brain_best_linear_print.py
 * extract_features_hf.py -- for intrinsic dimensionality
