from dadapy import Data
import numpy as np 
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def pick_two(features, seed_layer: int):
    layers = build_all_data_parallel(features)
    seed = layers[seed_layer]
    L_layers = len(layers)

    def compute_idcorr(l):
        if l == seed_layer:
            return l, np.inf
        return l, idCorr(seed, layers[l])

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_idcorr, l) for l in range(L_layers)]
        for f in tqdm(as_completed(futures), total=L_layers, desc="Computing idCorr"):
            results.append(f.result())

    best_layer, min_id_corr = min(results, key=lambda x: x[1])
    print(f'best layer: {best_layer}')
    return seed_layer, best_layer

def build_all_data_parallel(features):
    L_layers = features.shape[1]
    layers = [None] * L_layers

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(build_data, features, l) for l in range(L_layers)]
        for f in tqdm(as_completed(futures), total=L_layers, desc="Building Data"):
            l, data_obj = f.result()
            layers[l] = data_obj

    return layers

def build_data(features, l):
    return l, Data(features[:, l, :])


def idCorr(data_X: Data, data_Y: Data):
    idX = GRIDE(data_X)
    idY = GRIDE(data_Y)

    data_XY = Data(np.concat([data_X.coordinates, data_Y.coordinates], axis=1))

    idXY = GRIDE(data_XY)
    return (idX + idY - idXY) / max(idX, idY)

def GRIDE(data_X: Data):
    ie = data_X
    ids_scaling, ids_scaling_err, rs_scaling = ie.return_id_scaling_gride(range_max = 512)

    gride_scale = index_of_flattest_point_with_boundaries(ids_scaling)

    return ids_scaling[gride_scale]

def index_of_flattest_point_with_boundaries(arr):
    arr = np.asarray(arr)
    n = len(arr)
    derivatives = []

    for i in range(n):
        if i == 0:
            derivative = arr[1] - arr[0]
        elif i == n - 1:
            derivative = arr[n - 1] - arr[n - 2]
        else:
            derivative = (arr[i + 1] - arr[i - 1]) / 2

        derivatives.append(derivative)

    # Find index where absolute derivative is minimized
    min_index = int(np.argmin(np.abs(derivatives)))
    return min_index, derivatives[min_index] 
    
