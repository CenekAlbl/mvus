# various RANSAC algorithms
import numpy as np
import warnings
from scipy.optimize import least_squares
def vanillaRansac(estimateFn, verifyFn, data, minSamples, threshold, maxIter, param=None, verbose=0):
    """A vanilla implementation of RANSAC with fixed number of iterations

    Runs fixed number of iterations of RANSAC and outputs the model that has the most inliers. Model is represented as a set of parameters in a (n,1) numpy array where n is the number of the model parameters. Input data is a shape (m,k) numpy array where m is the dimension of one sample and k is the number of samples. 

    E.g. line fitting in two dimension would take data = np.array((2,n)) where n is the number of samples and data[i,:] = np.array([x_i,y_i]) and produce a result["model"] containing the parameters of the line (a,b). result["inliers"] contains the indices of inliers from the input data.


    Parameters
    ----------
    estimateFn : function handle
        a function that estimates the model, i.e. returns a list of possible hypotheses from a given minSamples of data.
    verifyFn : function handle
        function that takes a single model M and data and computes the error on this data
    data : numpy.array((m,k)) 
        Input data where m is the size of one sample and k is the number of samples
    minSamples : int
        number of samples needed to produce a model by estimateFn
    threshold : float 
        maximum error for data point to be considered an inlier
    maxIter : int
        number of iterations
    param: dict
        optional parameters or settings that can be used by estimateFn and estimateFn
    verbose : bool, optional
        switch to display warnings

    Returns
    -------
    result
        a dictionary where the optimal model is res["model"] and the inlier indices are res["inliers"] 

    """
    nsamples = data.shape[1]
    nInliersMax = 0
    idxs = np.arange(nsamples)
    result = {}
    for i in range(0,maxIter):
        sampleIdxs = np.random.choice(idxs, size=minSamples)
        M = estimateFn(data[:,sampleIdxs],param)
        if len(M) is not 0:
            if len(M.shape)==1:
                M = np.expand_dims(M,axis=0)
            for Mi in M:
                err = verifyFn(Mi,data,param)
                if(len(err.shape)>1):
                    err = np.sum(err,0)
                inliers = idxs[err<threshold]
                if np.sum(err[sampleIdxs])>1e-4 and verbose:
                    warnings.warn('Error on selected points too large!')
                if len(inliers) > nInliersMax:
                    result["model"] = Mi
                    result["inliers"] = inliers
                    nInliersMax = len(inliers)
                    if verbose:
                        print("Iteration %d, inliers: %d" % (i,nInliersMax))
    if not result and verbose:
        warnings.warn('Model not found! (something is wrong)')
    return result

def f(x, y):
    print(x, y)

def loRansacSimple(estimateFn, verifyFn, data, n, threshold, maxIter, param=None, optimizeFn=None, optimizeThr=None, verbose=0):
    """An implementation of simple version of LO-RANSAC as in [1] with fixed number of iterations

    Runs fixed number of iterations of LO-RANSAC in the simple version from [1] and outputs the model that has the most inliers. Model is represented as a set of parameters in a (n,1) numpy array where n is the number of the model parameters. Input data is a shape (m,k) numpy array where m is the dimension of one sample and k is the number of samples. 

    E.g. line fitting in two dimension would take data = np.array((2,n)) where n is the number of samples and data[i,:] = np.array([x_i,y_i]) and produce a result["model"] containing the parameters of the line (a,b). result["inliers"] contains the indices of inliers from the input data.


    Parameters
    ----------
    estimateFn : function handle
        a function that estimates the model, i.e. returns a list of possible hypotheses from a given minSamples of data.
    verifyFn : function handle
        function that takes a single model M and data and computes the error on this data
    data : numpy.array((m,k)) 
        Input data where m is the size of one sample and k is the number of samples
    minSamples : int
        number of samples needed to produce a model by estimateFn
    threshold : float 
        maximum error for data point to be considered an inlier
    maxIter : int
        number of iterations
    param: dict
        optional parameters or settings that can be used by estimateFn and estimateFn
    optimizeFn : function handle, optional
        function that takes data and model as input and computes error on each datapoint. This one is used in the optimization part, therefore the error computed by this function will be minimized. By default, verifyFn is used, but this parameter allows to define a different function to be optimized than the one used to compute the error of the model.
    optimizeThr : float, optional
        threshold to be used for filtering inliers from the output of optimizeFn. By default, threshold is used but this allows for having different criteria for inliers for the estimation and optimization part.
    verbose : bool
        switch to display warnings

    Returns
    -------
    result
        a dictionary where the optimal model is res["model"] and the inlier indices are res["inliers"] 
    
    [1] Chum O., Matas J., Kittler J. (2003) Locally Optimized RANSAC. In: Michaelis B., Krell G. (eds) Pattern Recognition. DAGM 2003. Lecture Notes in Computer Science, vol 2781. Springer, Berlin, Heidelberg
    
    """
    if optimizeFn is None:
        optimizeFn = verifyFn
    if optimizeThr is None:
        optimizeThr = threshold
    nsamples = data.shape[1]
    nInliersMax = 0
    idxs = np.arange(nsamples)
    result = {}
    for i in range(0,maxIter):
        sampleIdxs = np.random.choice(idxs, size=n)
        M = estimateFn(data[:,sampleIdxs],param)
        if len(M) is not 0:
            if len(M.shape)==1:
                M = np.expand_dims(M,axis=0)
            for Mi in M:
                #Mi = M[:,j]
                err = verifyFn(Mi,data,param)
                if(len(err.shape)>1):
                    err = np.sum(err,0)
                inliers = idxs[err<threshold]
                if np.sum(err[sampleIdxs])>1e-4 and verbose:
                    warnings.warn('Error on selected points too large!')
                if len(inliers) > nInliersMax:
                    result["model"] = Mi
                    result["inliers"] = inliers
                    nInliersMax = len(inliers)
                    if verbose:
                        print("Iteration %d, inliers: %d" % (i,nInliersMax))
                    # Do local optimization on inliers
                    fn = lambda x: optimizeFn(x,data[:,inliers],param)
                    res = least_squares(fn,Mi.ravel())
                    Mo = res["x"]
                    err = verifyFn(Mo,data,param)
                    inliers = idxs[err<optimizeThr]
                    if len(inliers) >= nInliersMax:
                        result["model"] = Mo
                        result["inliers"] = inliers
                        nInliersMax = len(inliers)
                        if verbose:
                            print("Iteration %d, inliers after LO: %d"% (i,nInliersMax))
                    else:
                        warnings.warn("Found smaller set after optimization")
    if not result and verbose:
        warnings.warn('Model not found! (something is wrong)')
    return result
