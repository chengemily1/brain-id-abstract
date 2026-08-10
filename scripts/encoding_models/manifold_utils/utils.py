import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import joblib 
import json
import pdb

def get_stats(preds, actuals):
    preds = preds.T
    actuals = actuals.T
    r2 = r2_score(actuals, preds)
    print('R2: ', r2)
    correlations = [pearsonr(preds[i], actuals[i])[0] for i in range(len(preds))]
    print('Mean Correlation: ', np.mean(correlations))
    print('Std Correlation: ', np.std(correlations))
    print('Min Correlation: ', np.min(correlations))
    print('Max Correlation: ', np.max(correlations))

    return r2, correlations 

def get_layer_order(model_name):
    with open('llm_audio_ids.json', 'r') as f:
        ids = json.load(f)
    
    model_str = model_name.split('/')[-1]
    if 'pythia' in model_str:
        model_str = 'pythia_step143000'
    elif 'wavlm' in model_str:
        model_str = 'wavlm-base-plus'
    elif 'whisper' in model_str:
        model_str = 'whisper-large'

    ids = ids[model_str]['id']

    layer_order = sorted(range(len(ids)), key=lambda i: ids[i], reverse=False)
    
    if 'whisper' in model_str:
        layer_order = [l for l in layer_order if l % 2 == 0]
        layer_order = [l //2 for l in layer_order]
        layer_order.append(16) # missed the last layer for some reason

    return layer_order