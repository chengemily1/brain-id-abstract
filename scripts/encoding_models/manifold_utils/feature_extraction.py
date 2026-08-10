from typing import Iterable
import torch
import time 
from tqdm import tqdm 
import pdb 
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM # Only necessary for feature extraction.

from ridge_utils.tokenization_helpers import generate_efficient_feat_dicts_opt, convert_to_feature_mats_opt, generate_efficient_feat_dicts_pythia, convert_to_feature_mats_pythia, generate_efficient_feat_dicts_llama, convert_to_feature_mats_llama
from manifold_utils.id_corr import pick_two

class FeatureExtractor:
    """
        This class takes as input a model and text inputs,
        then selects the relevant features.
    """
    def __init__(self, wordseqs, model_str: str, train_stories: Iterable[str], test_stories: Iterable[str], device='cuda'):
        # Model and tokenizer
        self.model_name = model_str
        self.tokenizer = AutoTokenizer.from_pretrained(model_str) # Same tokenizer for all sizes
        self.model = AutoModelForCausalLM.from_pretrained(model_str, device_map='auto')#.to(device)
        self.device = device

        # Data
        self.wordseqs = wordseqs
        if 'pythia' in self.model_name:
            self._pythia_specific_cleanup()
        self.train_stories = train_stories 
        self.test_stories = test_stories

        # Input text format and model 
        generate_efficient_feat_dicts = self._get_efficient_feat_dicts_generator()
        self.text_dict, self.text_dict2, self.text_dict3 = generate_efficient_feat_dicts(wordseqs, self.tokenizer, 256, 512)

        # Features stored in self.text_dict3
        self.text_dict3 = self._extract_features()

        # Memory management
        del self.model 
        print('Deleted model')

    def _pythia_specific_cleanup(self):
        for es, story in enumerate(self.wordseqs.keys()):
            data = []
            data_times = []
            for ei, i in enumerate(self.wordseqs[story].data):
                if i.strip() != "" and i[0] != '{':
                    data.append(i.strip())
                    data_times.append(self.wordseqs[story].data_times[ei])
            self.wordseqs[story].data = data
            self.wordseqs[story].data_times = data_times
        
    def _get_efficient_feat_dicts_generator(self):
        if 'opt' in self.model_name:
            return generate_efficient_feat_dicts_opt
        elif 'pythia' in self.model_name:
            return generate_efficient_feat_dicts_pythia
        elif 'Llama' in self.model_name:
            return generate_efficient_feat_dicts_llama
        else:
            raise ValueError(f"Model {self.model_name} not supported for feature extraction.")


    def _extract_features(self):
        start_time = time.time()
        print('Extracting features')
        for phrase in tqdm(self.text_dict2):
            if self.text_dict2[phrase]:
                inputs = {}
                inputs['input_ids'] = torch.tensor([self.text_dict[phrase]]).int().to(self.model.device)
                inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape).to(self.model.device)
                out = torch.cat(self.model(**inputs, output_hidden_states=True)[2], dim=0) # L layers x N_toks x hidden_dim
                out = out.cpu().detach().numpy()
                out = np.array(out)

                this_key = tuple(inputs['input_ids'][0].cpu().detach().numpy())
                acc_true = 0
                for ei, _ in enumerate(this_key):
                    if this_key[:ei+1] in self.text_dict3:
                        acc_true += 1
                        self.text_dict3[this_key[:ei+1]] = out[:, ei, :] # index into the correct token
        end_time = time.time()
        print("Feature extraction took", end_time - start_time, "seconds on", self.model.device)

        return self.text_dict3
    
    def _get_features_getter(self):
        if 'opt' in self.model_name:
            return convert_to_feature_mats_opt
        elif 'pythia' in self.model_name:
            return convert_to_feature_mats_pythia
        elif 'Llama' in self.model_name:
            return convert_to_feature_mats_llama
        else:
            raise ValueError(f"Model {self.model_name} not supported for feature extraction.")


    def get_features(self, selection_method: str, seed_layer = None):
        """
        Args:
            selection_method (str): selection_method in "single layer", "idCorr"

        Returns:
            dict {story_name (str): features np.array} : Each feature matrix is N x d
        """
        assert selection_method in ['single', 'all', 'idCorr', 'every_other', 'ipca'], "selection_method must be one of ['single', 'all', 'idCorr']"

        # result is {story_name: N x L layers x d dimensions}
        convert_to_feature_mats = self._get_features_getter()
        t = time.time()
        result = convert_to_feature_mats(self.wordseqs, self.tokenizer, 256, 512, self.text_dict3)
        print(f'Convert to feature mats took {time.time() - t} seconds')

        # memory management
        del self.tokenizer

        # Select features
        if selection_method == 'single':
            result_feature_selected = {story: result[story][:,seed_layer,:] for story in result}
            layer_idxs = [seed_layer]
        elif selection_method in ('all', 'ipca'):
            result_feature_selected = {}
            for story in result:
                L_layers = result[story].shape[1]
                result_feature_selected[story] = np.reshape(result[story], (result[story].shape[0], -1))
                self.L_layers = L_layers
            layer_idxs = list(range(L_layers))
        elif selection_method == 'every_other':
            result_feature_selected = {}
            for story in result:
                L_layers = result[story].shape[1]
                result_feature_selected[story] = np.reshape(result[story], (result[story].shape[0], -1))
            layer_idxs = list(range(0,L_layers,2))
        elif selection_method == 'idCorr':
            train_stories_concat = np.concatenate([result[story] for story in self.train_stories], axis=0)
            print('get the shape of all_stores_concat, should still be L x D at the end')
            layer_idxs = pick_two(train_stories_concat, seed_layer)
            print('print the layer_idxs')
            dim = train_stories_concat.shape[-1]

            # {story: N x (l * D)}
            result_feature_selected = {story: np.reshape(result[story][:, layer_idxs, :], (result[story].shape[0], len(layer_idxs) * dim)) for story in result}
        
        self.layer_idxs = layer_idxs
        assert np.all([len(result_feature_selected[story].shape) == 2 for story in result_feature_selected]), "Feature selection failed, not 2D"
        
        # Memory management
        del result

        # Across train and test
        return result_feature_selected