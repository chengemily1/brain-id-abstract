import numpy as np
import logging
import sys
import time
import joblib
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM
from transformers import AutoTokenizer
import torch
import pickle
import sys
import argparse
from tqdm import tqdm 
import copy
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m") # choices: facebook/opt-13b EleutherAI/pythia-6.9b-deduped
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument('--task', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    return args


def train_linear_probe(X, y, X_val, y_val, num_classes=None, lr=1e-2, weight_decay=0.0, 
                       epochs=100, batch_size=64, device=None, verbose=True):
    """
    Train a linear probe on representations X to predict class indices y using cross-entropy.
    Returns the model with the highest validation accuracy.

    Args:
        X (torch.Tensor): Training representations, shape (N_train, d)
        y (torch.Tensor): Training labels, shape (N_train,)
        X_val (torch.Tensor): Validation representations, shape (N_val, d)
        y_val (torch.Tensor): Validation labels, shape (N_val,)
        num_classes (int, optional): Number of categories. If None, inferred from y.
        lr (float): Learning rate
        weight_decay (float): L2 regularization
        epochs (int): Number of training epochs
        batch_size (int): Mini-batch size
        device (str or torch.device): "cpu" or "cuda", default auto-detect
        verbose (bool): Print validation loss and accuracy during training

    Returns:
        best_model (nn.Module): Linear probe with highest validation accuracy
        val_history (dict): 'loss' and 'accuracy' lists per epoch on validation set
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = X.to(device), y.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    N_train, d = X.shape
    num_classes = num_classes or int(y.max().item() + 1)

    # Linear probe
    model = nn.Linear(d, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    val_history = {'loss': [], 'accuracy': []}

    best_acc = -1.0
    best_model_state = None

    # Training loader
    train_dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # --- Compute validation loss & accuracy ---
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        val_history['loss'].append(val_loss)
        val_history['accuracy'].append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Load best model
    best_model = nn.Linear(d, num_classes).to(device)
    best_model.load_state_dict(best_model_state)

    return best_model, val_history



if __name__ == '__main__':
    args = parse_args()
    print(args)

    model_name = args.model
    task = args.task
    batch_size = args.batch_size

    checkpoint = args.checkpoint

    if not checkpoint.startswith('step'): checkpoint = None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    print(model_name)
    print('checkpoint', checkpoint)
    # Load model
    if 'pythia' in model_name:
        model = GPTNeoXForCausalLM.from_pretrained(
              model_name,
              revision=checkpoint,
              device_map='auto'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        
    model.eval()

    # Load tokenizer
    if ("facebook/opt" in model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

    
    print('Model and tokenizer loaded on ', model.device)
    # Load and preprocess data
    df = pd.read_csv(f"/home/echeng/encoding-models/conneau/{task}.txt", sep="\t", header=None, names=['subset', 'label', 'sentence'])
    
    unique_labels = list(df.label.unique())
    df['label'] = df['label'].map(lambda x: unique_labels.index(x))

    # Sentence length issues
    if args.task == 'sentence_length':
        df = df[df['label']<5]

    train_df = df[df['subset']=='tr'][['label', 'sentence']]
    test_df = df[df['subset']=='te'][['label', 'sentence']]
    val_df = df[df['subset']=='va'][['label', 'sentence']]

    print('Data preprocessed')
    

    def model_pass(raw_inputs):
        inputs = tokenizer(raw_inputs, padding=True, return_tensors="pt").to(device)
    
        last_true_token_indices = []
        for att_mask in inputs.attention_mask:
            att_mask = att_mask.tolist()
            if 0 in att_mask:
                idx = att_mask.index(0) - 1
            else:
                idx = len(att_mask) - 1
            idx = max(0, min(idx, len(att_mask) - 1))
            last_true_token_indices.append(idx)
    
        with torch.no_grad():
            hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    
        per_layer_activations = []
        for raw_activation in hidden_states:
            last_token_activations = []
            seq_len = raw_activation.shape[1]
            for i in range(len(last_true_token_indices)):
                idx = min(last_true_token_indices[i], seq_len - 1)
                last_token_activation = raw_activation[i][idx].cpu().numpy()
                last_token_activations.append(last_token_activation)
            per_layer_activations.append(last_token_activations)
    
        return per_layer_activations


    def extract_reps(inputs):
        cases_count = len(inputs)
    
        first_index = 0
        current_batch_size = batch_size
        states = dict()

        for first_index in tqdm(range(0, cases_count, current_batch_size), desc="Processing batches"):
            try:
                curr_output = model_pass(inputs[first_index:first_index+current_batch_size])
            except Exception as e:
                print(e)
                return states
                
            for i in range(len(curr_output)):
                if not i in states:
                    states[i] = []
                states[i] = states[i] + curr_output[i]
            first_index=first_index+current_batch_size
            
        # in case cases_count is not a multiple of batch_size
        if first_index<cases_count:
            curr_output = model_pass(inputs[first_index:cases_count])
            for i in range(len(curr_output)):
                states[i] = states[i] + curr_output[i]
    
        return states

    # EXTRACT REPS
    print('Extracting reps')
    inputs = list(train_df['sentence'])
    states = extract_reps(inputs)
    print('train done')
    
    val_inputs = list(val_df['sentence'])
    val_states = extract_reps(val_inputs)
    print('val done')
    
    test_inputs = list(test_df['sentence'])
    test_states = extract_reps(test_inputs)
    print('test done')
    
    # Memory
    del model
    del tokenizer

    # Train probe for each layer
    test_accs = []
    for layer in tqdm(range(1, len(states)), desc='Training probe'):
        # pdb.set_trace()
        X_train, y_train = torch.Tensor(states[layer]).to(device), torch.LongTensor(list(train_df['label'])).to(device)
        X_val, y_val = torch.Tensor(val_states[layer]).to(device), torch.LongTensor(list(val_df['label'])).to(device)
        X_test, y_test = torch.Tensor(test_states[layer]).to(device), torch.LongTensor(list(test_df['label'])).to(device)

        
        model, val_history = train_linear_probe(X_train, y_train, X_val, y_val,
                                        epochs=args.epochs, lr=5e-3)

        criterion = nn.CrossEntropyLoss()
        test_logits = model(X_test)
        test_loss = criterion(test_logits, y_test).item()
        test_preds = test_logits.argmax(dim=1)
        test_acc = (test_preds == y_test).float().mean().item()

        # Save
        test_accs.append(test_acc)
    model_str = model_name.split('/')[-1]
    with open(f'/home/echeng/encoding-models/results/{model_str}/ckpt_{checkpoint}_{task}.pkl', 'wb') as f:
        pickle.dump(test_accs, f)