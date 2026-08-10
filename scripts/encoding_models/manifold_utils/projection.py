import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pdb
from tqdm import tqdm
from sklearn.decomposition import PCA
from pydiffmap import diffusion_map as dm
import numpy as np

def squish(X, layer_order, d=1000):
    # X is layers x N x d
    ipcas = []
    
    # init
    squished_i = X[layer_order[0]]  # N x d0
    # squished_i = (squished_i - squished_i.mean(0)) / (squished_i.std(0) + 1e-8)
    
    # iterate over the rest
    for j in tqdm(range(1, len(layer_order)), desc='iterating pca'):
        layer_idx = layer_order[j]
        layer_i = X[layer_idx]  # N x d_i
        # layer_i = (layer_i - layer_i.mean(0)) / (layer_i.std(0) + 1e-8) # normalize
    
        # combine current squished representation and new layer
        combined = np.concatenate([squished_i, layer_i], axis=1)  # N x (d_prev + d_i)
    
        # update PCA basis
        ipca = PCA(n_components=d)
        ipca.fit(combined)
    
        # project combined manifold into shared d-dimensional subspace
        squished_i = ipca.transform(combined)

        ipcas.append(ipca)

    return squished_i, ipcas

def squish_test(X_test, ipcas, layer_order):
    """
    Project test representations into the same shared subspace 
    learned from training via PCA.

    X_test: np.ndarray of shape (num_layers, N_test, d_i)
    ipca: fitted IncrementalPCA object from training
    layer_order: list of layer indices used during training
    """
    
    # initialize with the first layer
    squished_i = X_test[layer_order[0]]
    squished_i = (squished_i - squished_i.mean(0)) / (squished_i.std(0) + 1e-8)
    
    for j in tqdm(range(1, len(layer_order)), desc='projecting test'):
        layer_idx = layer_order[j]
        layer_i = X_test[layer_idx]
        layer_i = (layer_i - layer_i.mean(0)) / (layer_i.std(0) + 1e-8)
        
        # combine the current squished representation with the new layer
        combined = np.concatenate([squished_i, layer_i], axis=1)
        
        # project into the learned PCA subspace (no fitting)
        ipca = ipcas[j-1]
        squished_i = ipca.transform(combined)

    return squished_i  # shape: (N_test, d)

class IdentityProjection:
    def __init__(self):
        pass

    def fit(self, x):
        return x
    
    def transform(self, x):
        return x
    
    def inverse_transform(self, x):
        return x
        

def get_up_projection_map(args, My_train, Rresp, Presp, project_type='pca', projection_map_y=None):
    """
    Returns a function that up-projects a vector in the projected space back to the original space.
    """
    if project_type == 'pca':
        return projection_map_y.inverse_transform
    elif project_type == 'I':
        return projection_map_y.inverse_transform
    elif project_type == 'dm':
        input_dim = My_train.shape[-1]
        output_dim = Rresp.shape[-1]  

        My_train = torch.Tensor(My_train).to('cuda')
        Rresp = torch.Tensor(Rresp).to('cuda')
        train_dataset = TensorDataset(My_train, Rresp)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        My_test = projection_map_y.transform(Presp)
        My_test = torch.Tensor(My_test).to('cuda')
        Presp = torch.Tensor(Presp).to('cuda')
        test_dataset = TensorDataset(My_test, Presp)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = UpProjection(input_dim, output_dim)
        train_model(model, train_dataloader, test_dataloader, epochs=args.autoencoder_epochs, lr=args.autoencoder_lr)
        return model

def down_project(x, project_type='pca', n_evecs=0.97):
    """
    Down-projects x to n_evecs using PCA or DM
    """
    if project_type == 'pca':
        if type(n_evecs) == float:
            if n_evecs.is_integer(): n_evecs = int(n_evecs)

        pca = PCA(n_components=n_evecs, whiten=True)
        # project x onto the n eigenvectors set by n_evecs
        x_projected = pca.fit_transform(x)
        print('Explained var: ', np.sum(pca.explained_variance_ratio_))
        print('Rank: ', pca.n_components_)
        return x_projected, pca
    
    elif project_type == 'dm':
        n_evecs = 1000 if n_evecs == 'auto' else n_evecs
        dmap = dm.DiffusionMap.from_sklearn(n_evecs=int(n_evecs), epsilon='bgh', k=300)
        return dmap.fit_transform(x), dmap
    
    elif project_type == 'I':
        return x, IdentityProjection()
    else:
        raise ValueError(f"Invalid projection type: {project_type}")

def get_up_projections_torch(My_train_hat, Rresp, up_projection_map_y):
    """
    This is for memory issues when using gpu.
    """
    eval_dataset = TensorDataset(My_train_hat, torch.Tensor(Rresp).to('cuda'))
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    preds = []

    up_projection_map_y.eval()
    for x_batch, _ in eval_dataloader:
        with torch.no_grad():
            pred = up_projection_map_y(x_batch).detach().cpu().numpy()
            preds.append(pred)

    preds = np.vstack(preds)
    return preds


class UpProjection(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[500, 1000, 10000]):
        super(UpProjection, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on the last layer
                layers.append(nn.GELU())

        self.model = nn.Sequential(*layers)
        self.model = self.model.to('cuda')

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_dataloader, val_dataloader, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        train_bs = x_batch.shape[0]

        if epoch % 10 == 0:
            # Evaluate
            model.eval()
            val_loss = []
            with torch.no_grad():
                for x_batch, y_batch in val_dataloader:
                    y_pred = model(x_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss.append(loss.item())
            test_bs = x_batch.shape[0]
            val_loss = sum(val_loss) / (test_bs * len(val_loss))
            model.train()

            print(f"Epoch {epoch}, Train Loss: {loss.item() / train_bs:.6f}")
            print(f"Epoch {epoch}, Test Loss: {val_loss:.6f}")

