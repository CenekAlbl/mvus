import numpy as np
import pickle
import cv2
from scipy import interpolate, linalg
from reconstruction import epipolar
from tools import util


def sync_iter(fps1, fps2, detect1, detect2, frame1, frame2):


    def correspondence(beta_estimate, d=0):

        if d:
            ts2_d, _ = util.sampling(detect2[0]+d, int2)
            ts2 = ts2_d-d 
            ts1 = ts2 + beta_estimate
            ts1, idx = util.sampling(ts1, int1)
            ts2, ts2_d = ts2[idx], ts2_d[idx]

            x1 = np.asarray(interpolate.splev(ts1, spline1))
            x2 = np.asarray(interpolate.splev(ts2, spline2))
            x2_d = np.asarray(interpolate.splev(ts2_d, spline2))
            xd = x2_d - x2
            return np.concatenate((x1,x2,xd),axis=0)
            # return x1, x2, xd
        else:
            ts1, idx = util.sampling(detect2[0]+beta_estimate, int1)
            ts2 = detect2[0,idx]
            try:
                x1 = np.asarray(interpolate.splev(ts1, spline1))
                x2 = np.asarray(interpolate.splev(ts2, spline2))
                return x1, x2
            except:
                return None        


    def solver(data,param):
        '''
        This function reads data of 9 points and return a list of possible solution for Beta and F
        '''

        assert data.shape[1]==9, 'Number of input points must be 9'
        assert data.shape[0]==6, 'Input data must have 6 rows'

        # decompose input data
        s1 = data[:2,:]
        s2 = data[2:4,:]
        ds = data[4:,:]

        # Create design matrices A1, A2
        A1 = np.array([s1[0]*s2[0],s1[0]*s2[1],s1[0], s1[1]*s2[0],s1[1]*s2[1],s1[1], s2[0],s2[1],np.ones(9)]).T
        A2 = np.array([s1[0]*ds[0],s1[0]*ds[1],np.zeros(9), s1[1]*ds[0],s1[1]*ds[1],np.zeros(9), ds[0],ds[1],np.zeros(9)]).T

        # Compute eigenvalue
        w = -linalg.eigvals(A1,A2)
        w[np.iscomplex(w)] = np.inf
        betas = w.real[np.isfinite(w)]

        # compute fundamental matrix F using shifted data
        M = np.empty((0,10))
        for beta in betas:
            x2 = s2 + beta*ds
            # F = ep.compute_fundamental(s1,x2)
            F, mask = cv2.findFundamentalMat(s1[:2].T,x2[:2].T,method=cv2.FM_8POINT)

            if len(np.ravel(F)) == 9:
                M_i = np.append(np.ravel(F),np.array([beta*param['d']]))
                M = np.vstack((M, M_i))

        return M


    def error(M,data,param):
        cor = correspondence(beta_prior+M[-1])
        if cor is not None:
            x1, x2 = cor[0], cor[1]
            return epipolar.Sampson_error(util.homogeneous(x1), util.homogeneous(x2), M[:9].reshape((3,3)))
        else:
            return None


    def vanillaRansac(estimateFn, verifyFn, data, minSamples, threshold, maxIter, param=None, verbose=0):

        nsamples = data.shape[1]
        nInliersMax = 0
        idxs = np.arange(nsamples)
        result = []
        for i in range(0,maxIter):
            sampleIdxs = np.random.choice(idxs, size=minSamples, replace=False)
            M = estimateFn(data[:,sampleIdxs],param)
            if len(M) is not 0:
                if len(M.shape)==1:
                    M = np.expand_dims(M,axis=0)
                for Mi in M:
                    err = verifyFn(Mi,data,param)
                    if err is not None:
                        numInliers = sum(err<threshold)
                        if numInliers > nInliersMax:
                            result = Mi
                            nInliersMax = numInliers
        return result, nInliersMax


    # Pre-processing
    alpha = fps1 / fps2
    beta_prior = frame1/alpha - frame2
    detect1[0] /= alpha

    int1 = util.find_intervals(detect1[0])
    int2 = util.find_intervals(detect2[0])

    spline1, _ = interpolate.splprep(detect1[1:], u=detect1[0], s=0, k=3)
    spline2, _ = interpolate.splprep(detect2[1:], u=detect2[0], s=0, k=3)

    # Iterativ algorithm
    skip, maxInlier, k, k_max = 0, 0, 0, 20
    p_min, p_max = 0, 6
    d, p, beta = 2**p_min, p_min, beta_prior

    while k < k_max:

        # Ransac with d
        d = abs(d)
        data = correspondence(beta, d)
        param = {'d':d}
        model1, numInlier1 = vanillaRansac(solver,error,data,9,5,200,param)

        # Ransac with -d
        d = -d
        data = correspondence(beta, d)
        param = {'d':d}
        model2, numInlier2 = vanillaRansac(solver,error,data,9,5,200,param)

        # Select the better one
        model = model1 if numInlier1 >= numInlier2 else model2
        numInlier = numInlier1 if numInlier1 >= numInlier2 else numInlier2

        print('d:{}, beta:{:.3f}, maxInlier:{}, numInlier:{}'.format(abs(d), beta, maxInlier, numInlier))

        if numInlier < maxInlier:
            if p < p_max:
                p += 1
            else:
                p = 0
            d = 2**p
            skip += 1

            if skip >= p_max:
                break
        else:
            beta -= model[-1]
            maxInlier = numInlier
            skip = 0
            k += 1


    return beta*alpha


def sync_bf(fps1, fps2, detect1, detect2, frame1, frame2, r=10):

    # Pre-processing
    alpha = fps1 / fps2
    detect1[0] /= alpha
    beta_prior = frame1/alpha - frame2


    def search(beta_list, thres=8):
        maxInlier = 0
        beta_est = 0
        for beta in beta_list:
            detect2_b = np.vstack((detect2[0]+beta,detect2[1:]))
            pts1, pts2 = util.match_overlap(detect1, detect2_b)

            F, mask = cv2.findFundamentalMat(pts1[1:].T, pts2[1:].T, method=cv2.FM_RANSAC, ransacReprojThreshold=thres)
            inlier = sum(mask.reshape(-1,)) #/ len(pts1[0])

            if inlier > maxInlier:
                maxInlier = inlier
                beta_est = beta

        return beta_est, maxInlier


    beta_coarse = np.arange(beta_prior-r*fps2, beta_prior+r*fps2, fps2)
    beta_est, _ = search(beta_coarse)

    beta_fine = np.arange(beta_est-fps2/2, beta_est+fps2/2, fps2/20)
    beta_est, numInlier  = search(beta_fine)

    return beta_est * alpha, numInlier/fps1


if __name__ == "__main__":

    path = './data/fixposition/trajectory/flight_5_25_20.pkl'
    with open(path, 'rb') as file:
        flight = pickle.load(file)

    fps1, fps2 = flight.cameras[0].fps, flight.cameras[2].fps
    detect1, detect2 = flight.detections[0], flight.detections[2]
    frame1, frame2 = 0, 409.606

    # beta = sync_iter(fps1, fps2, detect1, detect2, frame1, frame2)
    beta, overlap = sync_bf(fps1, fps2, detect1, detect2, frame1, frame2)

    print('Finished!')