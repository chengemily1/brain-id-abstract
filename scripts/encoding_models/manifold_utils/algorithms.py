# import umap
# import umap.parametric_umap
import numpy as np
import matplotlib.pyplot as plt

def UMAP_regression(X_train, Y_train, X_test, Y_test):
    """_summary_

    Args:
        X_train (_type_): _description_
        Y_train (_type_): _description_
        X_test (_type_): _description_
        Y_test (_type_): _description_
    """
    # Define a parametric UMAP model
    parametric_mapper = umap.parametric_umap.ParametricUMAP(n_components=2, random_state=42)

    # Train the model
    parametric_mapper.fit(X_train, y=Y_train)

    # Transform the training and test data
    X_train_umap = parametric_mapper.transform(X_train)
    X_test_umap = parametric_mapper.transform(X_test)

