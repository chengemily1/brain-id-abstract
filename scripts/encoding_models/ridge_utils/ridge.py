#import scipy
import numpy as np
import logging
from ridge_utils.utils import mult_diag, counter
import random
import itertools as itools
import joblib
import pdb
from manifold_utils.projection import down_project, get_up_projection_map, get_up_projections_torch

zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

logging.basicConfig(level=logging.INFO)
ridge_logger = logging.getLogger("ridge_corr")

def ridge(stim, resp, alpha, singcutoff=1e-10, normalpha=False, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [stim] that approximates
    [resp]. The regularization parameter is [alpha].

    Parameters
    ----------
    stim : array_like, shape (T, N)
        Stimuli with T time points and N features.
    resp : array_like, shape (T, M)
        Responses with T time points and M separate responses.
    alpha : float or array_like, shape (M,)
        Regularization parameter. Can be given as a single value (which is applied to
        all M responses) or separate values for each response.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value of stim. Good for
        comparing models with different numbers of parameters.

    Returns
    -------
    wt : array_like, shape (N, M)
        Linear regression weights.
    """
    try:
        U,S,Vh = np.linalg.svd(stim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(stim, full_matrices=False)

    UR = np.dot(U.T, np.nan_to_num(resp))
    
    # Expand alpha to a collection if it's just a single value
    if isinstance(alpha, (float,int)):
        alpha = np.ones(resp.shape[1]) * alpha
    
    # Normalize alpha by the LSV norm
    norm = S[0]
    if normalpha:
        nalphas = alpha * norm
    else:
        nalphas = alpha

    # Compute weights for each alpha
    ualphas = np.unique(nalphas)
    wt = np.zeros((stim.shape[1], resp.shape[1]))
    for ua in ualphas:
        selvox = np.nonzero(nalphas==ua)[0]
        #awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
        awt = Vh.T.dot(np.diag(S/(S**2+ua**2))).dot(UR[:,selvox])
        wt[:,selvox] = awt

    return wt

def ridge_projected(stim, resp, alpha, 
                    up_projection_map_y,
                    # up_projection_components, 
                    projection_map_y, 
                   singcutoff=1e-10, normalpha=False, logger=ridge_logger):
    """
    Performs ridge regression in projected Y space with a single alpha value.
    
    Parameters
    ----------
    stim : array_like, shape (T, N)
        Stimulus matrix with T timepoints and N features
    resp : array_like, shape (T, M)  
        Response matrix with T timepoints and M responses
    alpha : float
        Single regularization parameter applied uniformly in projected space
    up_projection_components : array_like, shape (n_components, M)
        Matrix to project from reduced space back to original response space
    projection_map_y : sklearn transformer
        Fitted projection (e.g., PCA) that transforms resp to reduced space
    singcutoff : float, default 1e-10
        Cutoff for removing small singular values
    normalpha : bool, default False
        Whether to normalize alpha by largest singular value
    
    Returns
    -------
    wt : array_like, shape (N, M)
        Ridge regression weights in original response space
    """
    # Ensure alpha is a single value
    if not isinstance(alpha, (float, int)):
        raise ValueError("alpha must be a single float/int value")
    
    # Project responses to reduced space
    resp_projected = projection_map_y.transform(resp)
    n_components = resp_projected.shape[1]
    
    # Compute SVD of stimulus
    try:
        U, S, Vh = np.linalg.svd(stim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U, S, Vh = svd_dgesvd(stim, full_matrices=False)
    
    # Remove tiny singular values
    good_s_mask = S > singcutoff
    U = U[:, good_s_mask]
    S = S[good_s_mask]
    Vh = Vh[good_s_mask]
    
    # Compute UR for projected responses
    UR = np.dot(U.T, np.nan_to_num(resp_projected))
    
    # Apply normalization if requested
    norm = S[0]
    normalized_alpha = alpha * norm if normalpha else alpha
    
    logger.info(f"Using alpha={alpha:.6f} (normalized: {normalized_alpha:.6f}) "
                f"for {n_components} components")
    
    # Compute ridge weights in projected space
    D = S / (S**2 + normalized_alpha**2)
    wt_projected = Vh.T.dot(np.diag(D)).dot(UR)
    
    # Project weights back to original space
    # wt = wt_projected @ up_projection_components
    wt = up_projection_map_y(wt_projected)
    
    return wt
    

def ridge_corr_pred(Rstim, Pstim, Rresp, Presp, valphas, normalpha=False,
                    singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. Returns the correlation 
    between predicted and actual [Presp], without ever computing the regression weights.
    This function assumes that each voxel is assigned a separate alpha in [valphas].

    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    valphas : list or array_like, shape (M,)
        Ridge parameter for each voxel.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    corr : array_like, shape (M,)
        The correlation between each predicted response and each column of Presp.
    
    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    joblib.dump(S, "singvals.jbl")
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    ## Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: %0.03f"%norm)
    if normalpha:
        nalphas = valphas * norm
    else:
        nalphas = valphas

    ## Precompute some products for speed
    UR = np.dot(U.T, Rresp) ## Precompute this matrix product for speed
    PVh = np.dot(Pstim, Vh.T) ## Precompute this matrix product for speed
    
    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(Presp)
    #Prespvar = Presp.var(0)
    Prespvar_actual = Presp.var(0)
    Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (Prespvar_actual - Prespvar).mean())

    ualphas = np.unique(nalphas)
    corr = np.zeros((Rresp.shape[1],))
    for ua in ualphas:
        selvox = np.nonzero(nalphas==ua)[0]
        alpha_pred = PVh.dot(np.diag(S/(S**2+ua**2))).dot(UR[:,selvox])

        if use_corr:
            corr[selvox] = (zPresp[:,selvox] * zs(alpha_pred)).mean(0)
        else:
            resvar = (Presp[:,selvox] - alpha_pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            corr[selvox] = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)

    return corr


def ridge_corr(Rstim, Pstim, Rresp, Presp, alphas, normalpha=False, corrmin=0.2,
               singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. The regression weights are NOT returned, because
    computing the correlations without computing regression weights is much, MUCH faster.

    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for each alpha.
    
    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    #joblib.dump((Rstim, U,S,Vh), "/hdd/singvals.jbl")
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    ## Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: %0.03f"%norm)
    if normalpha:
        nalphas = alphas * norm
    else:
        nalphas = alphas

    ## Precompute some products for speed
    UR = np.dot(U.T, Rresp) ## Precompute this matrix product for speed
    PVh = np.dot(Pstim, Vh.T) ## Precompute this matrix product for speed
    
    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(Presp)
    #Prespvar = Presp.var(0)
    Prespvar_actual = Presp.var(0)
    Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (Prespvar_actual - Prespvar).mean())
    Rcorrs = [] ## Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        #D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter 
        D = S / (S ** 2 + na ** 2) ## Reweight singular vectors by the (normalized?) ridge parameter
        
        pred = np.dot(mult_diag(D, PVh, left=False), UR) ## Best (1.75 seconds to prediction in test)
        # pred = np.dot(mult_diag(D, np.dot(Pstim, Vh.T), left=False), UR) ## Better (2.0 seconds to prediction in test)
        
        # pvhd = reduce(np.dot, [Pstim, Vh.T, D]) ## Pretty good (2.4 seconds to prediction in test)
        # pred = np.dot(pvhd, UR)
        
        # wt = reduce(np.dot, [Vh.T, D, UR]).astype(dtype) ## Bad (14.2 seconds to prediction in test)
        # wt = reduce(np.dot, [Vh.T, D, U.T, Rresp]).astype(dtype) ## Worst
        # pred = np.dot(Pstim, wt) ## Predict test responses

        if use_corr:
            #prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
            #Rcorr = np.array([np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1] for ii in range(Presp.shape[1])]) ## Slowly compute correlations
            #Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()/(prednorms*Prespnorms) ## Efficiently compute correlations
            Rcorr = (zPresp * zs(pred)).mean(0)
        else:
            ## Compute variance explained
            resvar = (Presp - pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            Rcorr = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)
            
        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
        
        log_template = "Training: alpha=%0.3f, mean corr=%0.5f, max corr=%0.5f, over-under(%0.2f)=%d"
        log_msg = log_template % (a,
                                  np.mean(Rcorr),
                                  np.max(Rcorr),
                                  corrmin,
                                  (Rcorr>corrmin).sum()-(-Rcorr>corrmin).sum())
        logger.info(log_msg)
    
    return Rcorrs

def ridge_corr_with_projection(Rstim, Pstim, Rresp, Presp, alphas, 
                            #   up_projection_components, 
                              up_projection_map_y,
                              projection_map_y,
                              normalpha=False, use_corr=True, 
                              singcutoff=1e-10, logger=ridge_logger):
    """
    Evaluate ridge regression performance with projection, correctly evaluating 
    performance in the original (up-projected) space.
    
    Key difference from original: This function trains in projected space but 
    evaluates performance in the ORIGINAL space after up-projection.
    
    Parameters
    ----------
    Rstim, Pstim : array_like
        Training and test stimuli
    Rresp, Presp : array_like  
        Training and test responses
    alphas : array_like
        Alpha values to test
    up_projection_components : array_like, shape (n_components, M)
        Matrix to project from reduced space back to original response space
    projection_map_y : sklearn transformer
        Fitted projection transformer
    normalpha : bool
        Whether to normalize alphas
    use_corr : bool
        Whether to use correlation (True) or R² (False) as metric
    
    Returns
    -------
    Rcorrs : list
        Correlation/R² values for each alpha, evaluated in ORIGINAL space
    """
    # Precompute z-scored test responses for correlation calculation
    if use_corr:
        zPresp = zs(Presp)
        Prespvar = None
    else:
        zPresp = None
        Prespvar = Presp.var(0)
        # Handle zero variance case
        Prespvar[Prespvar == 0] = 1.0
    
    Rcorrs = []
    
    for alpha in alphas:
        # Train ridge regression in projected space
        wt = ridge_projected(Rstim, Rresp, alpha, 
                            # up_projection_components, 
                           up_projection_map_y,
                           projection_map_y, singcutoff=singcutoff, 
                           normalpha=normalpha, logger=logger)
        
        # Predict in ORIGINAL space (this is the key correction)
        pred = Pstim @ wt
        
        # Evaluate performance in ORIGINAL space
        if use_corr:
            # Correlation metric
            Rcorr = (zPresp * zs(pred)).mean(0)
        else:
            # R-squared metric
            resvar = (Presp - pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            Rcorr = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)
        
        # Handle NaN values
        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
        
        # Log progress
        mean_corr = np.mean(Rcorr)
        max_corr = np.max(Rcorr)
        logger.info(f"Alpha {alpha:.6f}: mean_corr={mean_corr:.5f}, max_corr={max_corr:.5f}")
    
    return Rcorrs

def bootstrap_ridge(Rstim, Rresp, Pstim, Presp, alphas, nboots, chunklen, nchunks,
                    corrmin=0.2, joined=None, singcutoff=1e-10, normalpha=False, single_alpha=False,
                    use_corr=True, return_wt=True, logger=ridge_logger):
    """Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.
    [nchunks] random chunks of length [chunklen] will be taken from [Rstim] and [Rresp] for each regression
    run.  [nboots] total regression runs will be performed.  The best alpha value for each response will be
    averaged across the bootstraps to estimate the best alpha for that response.
    
    If [joined] is given, it should be a list of lists where the STRFs for all the voxels in each sublist 
    will be given the same regularization parameter (the one that is the best on average).
    
    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M different responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M different responses. Each response should be Z-scored across
        time.
    alphas : list or array_like, shape (A,)
        Ridge parameters that will be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
    chunklen : int
        On each sample, the training data is broken into chunks of this length. This should be a few times 
        longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of length 10.
    nchunks : int
        The number of training chunks held out to test ridge parameters for each bootstrap sample. The product
        of nchunks and chunklen is the total number of training samples held out for each sample, and this 
        product should be about 20 percent of the total length of the training data.
    corrmin : float in [0..1], default 0.2
        Purely for display purposes. After each alpha is tested for each bootstrap sample, the number of 
        responses with correlation greater than this value will be printed. For long-running regressions this
        can give a rough sense of how well the model works before it's done.
    joined : None or list of array_like indices, default None
        If you want the STRFs for two (or more) responses to be directly comparable, you need to ensure that
        the regularization parameter that they use is the same. To do that, supply a list of the response sets
        that should use the same ridge parameter here. For example, if you have four responses, joined could
        be [np.array([0,1]), np.array([2,3])], in which case responses 0 and 1 will use the same ridge parameter
        (which will be parameter that is best on average for those two), and likewise for responses 2 and 3.
    singcutoff : float, default 1e-10
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    normalpha : boolean, default False
        Whether ridge parameters (alphas) should be normalized by the largest singular value (LSV)
        norm of Rstim. Good for rigorously comparing models with different numbers of parameters.
    single_alpha : boolean, default False
        Whether to use a single alpha for all responses. Good for identification/decoding.
    use_corr : boolean, default True
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    return_wt : boolean, default True
        If True, this function will compute and return the regression weights after finding the best
        alpha parameter for each voxel. However, for very large models this can lead to memory issues.
        If false, this function will _not_ compute weights, but will still compute prediction performance
        on the prediction dataset (Pstim, Presp).
    
    Returns
    -------
    wt : array_like, shape (N, M)
        If [return_wt] is True, regression weights for N features and M responses. If [return_wt] is False, [].
    corrs : array_like, shape (M,)
        Validation set correlations. Predicted responses for the validation set are obtained using the regression
        weights: pred = np.dot(Pstim, wt), and then the correlation between each predicted response and each 
        column in Presp is found.
    alphas : array_like, shape (M,)
        The regularization coefficient (alpha) selected for each voxel using bootstrap cross-validation.
    bootstrap_corrs : array_like, shape (A, M, B)
        Correlation between predicted and actual responses on randomly held out portions of the training set,
        for each of A alphas, M voxels, and B bootstrap samples.
    valinds : array_like, shape (TH, B)
        The indices of the training data that were used as "validation" for each bootstrap sample.
    """
    nresp, nvox = Rresp.shape
    valinds = [] # Will hold the indices into the validation data for each bootstrap
    
    Rcmats = []
    for bi in counter(range(nboots), countevery=1, total=nboots):
        logger.info("Selecting held-out test set..")
        allinds = range(nresp)
        indchunks = list(zip(*[iter(allinds)]*chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds)-set(heldinds))
        valinds.append(heldinds)
        
        RRstim = Rstim[notheldinds,:]
        PRstim = Rstim[heldinds,:]
        RRresp = Rresp[notheldinds,:]
        PRresp = Rresp[heldinds,:]
        
        # Run ridge regression using this test set
        Rcmat = ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
                           corrmin=corrmin, singcutoff=singcutoff,
                           normalpha=normalpha, use_corr=use_corr,
                           logger=logger)
        
        Rcmats.append(Rcmat)
    
    # Find best alphas
    if nboots>0:
        allRcorrs = np.dstack(Rcmats)
    else:
        allRcorrs = None
    
    if not single_alpha:
        if nboots==0:
            raise ValueError("You must run at least one cross-validation step to assign "
                             "different alphas to each response.")
        
        logger.info("Finding best alpha for each voxel..")
        if joined is None:
            # Find best alpha for each voxel
            meanbootcorrs = allRcorrs.mean(2)
            bestalphainds = np.argmax(meanbootcorrs, 0)
            valphas = alphas[bestalphainds]
        else:
            # Find best alpha for each group of voxels
            valphas = np.zeros((nvox,))
            for jl in joined:
                # Mean across voxels in the set, then mean across bootstraps
                jcorrs = allRcorrs[:,jl,:].mean(1).mean(1)
                bestalpha = np.argmax(jcorrs)
                valphas[jl] = alphas[bestalpha]
    else:
        logger.info("Finding single best alpha..")
        if nboots==0:
            if len(alphas)==1:
                bestalphaind = 0
                bestalpha = alphas[0]
            else:
                raise ValueError("You must run at least one cross-validation step "
                                 "to choose best overall alpha, or only supply one"
                                 "possible alpha value.")
        else:
            meanbootcorr = allRcorrs.mean(2).mean(1)
            bestalphaind = np.argmax(meanbootcorr)
            bestalpha = alphas[bestalphaind]
        
        valphas = np.array([bestalpha]*nvox)
        logger.info("Best alpha = %0.3f"%bestalpha)

    if return_wt:
        # Find weights
        logger.info("Computing weights for each response using entire training set..")
        wt = ridge(Rstim, Rresp, valphas, singcutoff=singcutoff, normalpha=normalpha)

        # Predict responses on prediction set
        logger.info("Predicting responses for predictions set..")
        pred = np.dot(Pstim, wt)

        # Find prediction correlations
        nnpred = np.nan_to_num(pred)
        if use_corr:
            corrs = np.nan_to_num(np.array([np.corrcoef(Presp[:,ii], nnpred[:,ii].ravel())[0,1]
                                            for ii in range(Presp.shape[1])]))
        else:
            resvar = (Presp-pred).var(0)
            Rsqs = 1 - (resvar / Presp.var(0))
            corrs = np.sqrt(np.abs(Rsqs)) * np.sign(Rsqs)

        return wt, corrs, valphas, allRcorrs, valinds
    else:
        # get correlations for prediction dataset directly
        corrs = ridge_corr_pred(Rstim, Pstim, Rresp, Presp, valphas, 
                                normalpha=normalpha, use_corr=use_corr,
                                logger=logger, singcutoff=singcutoff)

        return [], corrs, valphas, allRcorrs, valinds


def bootstrap_ridge_with_y_projection(Rstim, Rresp, Pstim, Presp, alphas, 
                                     nboots, chunklen, nchunks,
                                     up_projection_map_y, 
                                     projection_map_y,
                                     corrmin=0.2, singcutoff=1e-10, 
                                     normalpha=False, use_corr=True, 
                                     return_wt=True, y_projection='pca',
                                     logger=ridge_logger):
    """
    Bootstrap ridge regression with Y projection, correctly optimized.
    
    This function finds the single best alpha by:
    1. For each bootstrap sample and each alpha:
       - Train ridge in projected space with that alpha
       - Predict and evaluate performance in ORIGINAL space
    2. Select alpha with best average performance in original space
    3. Train final model with selected alpha
    
    Parameters
    ----------
    Rstim, Rresp : array_like
        Training stimuli and responses
    Pstim, Presp : array_like
        Test stimuli and responses  
    alphas : array_like
        Alpha values to test via cross-validation
    nboots : int
        Number of bootstrap samples
    chunklen, nchunks : int
        Bootstrap sampling parameters
    up_projection_components : array_like, shape (n_components, M)
        Matrix to project from reduced space back to original response space
    projection_map_y : sklearn transformer
        Fitted projection transformer
    corrmin : float
        Minimum correlation for progress reporting
    singcutoff : float
        SVD singular value cutoff
    normalpha : bool
        Whether to normalize alpha values
    use_corr : bool
        Whether to use correlation (True) or R² (False)
    return_wt : bool
        Whether to compute and return final weights
    y_projection : str, default 'pca'
        Type of Y projection. If 'I', should call standard bootstrap_ridge instead.
        
    Returns
    -------
    wt : array_like or []
        Final regression weights (if return_wt=True)
    corrs : array_like
        Final correlations on test set
    best_alpha : float
        Selected best alpha value
    bootstrap_corrs : array_like
        Bootstrap correlation results, shape (n_alphas, n_voxels, n_boots)
    valinds : list
        Validation indices for each bootstrap
    """
    nresp, nvox = Rresp.shape
    
    # Check if identity projection was requested
    if y_projection == 'I':
        print('Calling original ridge function')
        return bootstrap_ridge(Rstim, 
                               Rresp, 
                               Pstim, 
                               Presp, 
                               alphas, 
                               nboots, 
                               chunklen, 
                               nchunks)

    logger.info(f"Starting bootstrap with {nboots} boots, {len(alphas)} alphas")
    logger.info(f"Data: {nresp} timepoints, {nvox} voxels")
    logger.info(f"Projection: {projection_map_y.n_components} components, type: {y_projection}")
    
    # Bootstrap cross-validation
    valinds = []
    bootstrap_corrs = []  # Will be shape (n_alphas, n_voxels, n_boots)
    
    for bi in range(nboots):
        logger.info(f"Bootstrap {bi+1}/{nboots}")
        
        # Create train/validation split
        allinds = list(range(nresp))
        indchunks = list(zip(*[iter(allinds)] * chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds) - set(heldinds))
        valinds.append(heldinds)
        
        # Split data
        RRstim, PRstim = Rstim[notheldinds, :], Rstim[heldinds, :]
        RRresp, PRresp = Rresp[notheldinds, :], Rresp[heldinds, :]
        
        # Test each alpha and evaluate in ORIGINAL space
        boot_corrs = ridge_corr_with_projection(
            RRstim, PRstim, RRresp, PRresp, alphas,
            # up_projection_components,
            up_projection_map_y, 
            projection_map_y,
            normalpha=normalpha, use_corr=use_corr, 
            singcutoff=singcutoff, logger=logger
        )
        
        bootstrap_corrs.append(boot_corrs)
    
    # Convert to array: (n_boots, n_alphas, n_voxels) -> (n_alphas, n_voxels, n_boots)
    bootstrap_corrs = np.array(bootstrap_corrs)  # (n_boots, n_alphas, n_voxels)
    bootstrap_corrs = np.transpose(bootstrap_corrs, (1, 2, 0))  # (n_alphas, n_voxels, n_boots)
    
    # Find best alpha based on mean performance across voxels and boots
    mean_corr_per_alpha = bootstrap_corrs.mean(axis=(1, 2))  # Average over voxels and boots
    best_alpha_idx = np.argmax(mean_corr_per_alpha)
    best_alpha = alphas[best_alpha_idx]
    
    logger.info(f"Best alpha selected: {best_alpha:.6f}")
    logger.info(f"Best alpha mean correlation: {mean_corr_per_alpha[best_alpha_idx]:.5f}")
    
    # Log all alpha performances
    for i, (alpha, mean_corr) in enumerate(zip(alphas, mean_corr_per_alpha)):
        marker = " <-- BEST" if i == best_alpha_idx else ""
        logger.info(f"Alpha {alpha:.6f}: mean_corr={mean_corr:.5f}{marker}")
    
    if return_wt:
        # Train final model with best alpha on full training set
        logger.info("Training final model with best alpha...")
        wt = ridge_projected(Rstim, Rresp, best_alpha, 
                        #    up_projection_components,
                           up_projection_map_y,
                           projection_map_y, singcutoff=singcutoff, 
                           normalpha=normalpha, logger=logger)
        
        # Final prediction and evaluation
        logger.info("Computing final test set performance...")
        pred = Pstim @ wt
        
        if use_corr:
            corrs = (zs(Presp) * zs(pred)).mean(0)
        else:
            resvar = (Presp - pred).var(0)
            Presp_var = Presp.var(0)
            Presp_var[Presp_var == 0] = 1.0  # Avoid division by zero
            Rsqs = 1 - (resvar / Presp_var)
            corrs = np.sqrt(np.abs(Rsqs)) * np.sign(Rsqs)
        
        corrs[np.isnan(corrs)] = 0
        
        logger.info(f"Final test performance: mean_corr={np.mean(corrs):.5f}, "
                   f"max_corr={np.max(corrs):.5f}")
        
        return wt, corrs, best_alpha, bootstrap_corrs, valinds
    
    else:
        # Just compute correlations without storing weights
        logger.info("Computing test correlations without storing weights...")
        
        # We need to recompute the prediction for the test set
        wt = ridge_projected(Rstim, Rresp, best_alpha, up_projection_map_y,
        # up_projection_components,
                           projection_map_y, singcutoff=singcutoff, 
                           normalpha=normalpha, logger=logger)
        pred = Pstim @ wt
        
        if use_corr:
            corrs = (zs(Presp) * zs(pred)).mean(0)
        else:
            resvar = (Presp - pred).var(0)
            Presp_var = Presp.var(0)
            Presp_var[Presp_var == 0] = 1.0
            Rsqs = 1 - (resvar / Presp_var)
            corrs = np.sqrt(np.abs(Rsqs)) * np.sign(Rsqs)
        
        corrs[np.isnan(corrs)] = 0
        
        return [], corrs, best_alpha, bootstrap_corrs, valinds

