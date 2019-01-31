import numpy as np
from matplotlib import pyplot as plt
import struct  # Binary writing

from scipy.signal import medfilt, medfilt2d
import scipy.ndimage

import sklearn.decomposition as skdec
import sklearn.linear_model as sklin

from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut


# Regression
def regress(features, score):
    pred = []
    # Leave one out split
    loo = LeaveOneOut()	
    logo = LeaveOneGroupOut()
    for trainidx, testidx in loo.split(features):
        # Indices
        X_train, X_test = features[trainidx], features[testidx]
        X_test -= X_train.mean(0)
        X_train -= X_train.mean(0)

        Y_train, Y_test = score[trainidx], score[testidx]		
        # Linear regression
        regr = sklin.Ridge(alpha=1, normalize=True, random_state=42)
        regr.fit(X_train, Y_train)
        # Predicted score
        pred.append(regr.predict(X_test))

    return np.array(pred), regr.coef_


def regress_group(features, score, groups=None):
    pred = []
    if groups is None:
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 
                           9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 
                           15, 16, 16, 17, 17, 18, 19, 19])
    # Leave one out split
    loo = LeaveOneOut()	
    logo = LeaveOneGroupOut()
    logo.get_n_splits(features, score, groups)
    logo.get_n_splits(groups=groups)  # 'groups' is always required
    
    for trainidx, testidx in logo.split(features, score, groups):
        # Indices
        X_train, X_test = features[trainidx], features[testidx]
        X_test -= X_train.mean(0)
        X_train -= X_train.mean(0)

        Y_train, Y_test = score[trainidx], score[testidx]		
        # Linear regression
        regr = sklin.Ridge(alpha=1, normalize=True, random_state=42)
        regr.fit(X_train, Y_train)
        # Predicted score
        pred.append(regr.predict(X_test))
        
    predflat = []
    for group in pred:
        for p in group:
            predflat.append(p)

    return np.array(predflat), regr.coef_


def regress_new(features, sgrades):
    # Evaluate surface
    loo_surf = LeaveOneOut()
    loo_surf.get_n_splits(features)
    surfp = []
    for train_idx, test_idx in loo_surf.split(features):
        # Train split
        f = features[train_idx]-features.mean(0)
        g = sgrades[train_idx]

        # Linear regression
        Rmodel = sklin.Ridge(alpha=1, normalize=True, random_state=42)
        Rmodel.fit(f, g.reshape(-1, 1))

        # Evaluate on test sample
        p = Rmodel.predict((features[test_idx]-features.mean(0)).reshape(1, -1))
        surfp.append(p)
    return np.array(surfp).squeeze(), Rmodel.coef_


# Logistic regression
def logreg(features, score):
    pred = []
    # Leave one out split
    loo = LeaveOneOut()	
    for trainidx, testidx in loo.split(features):
        # Indices
        X_train, X_test = features[trainidx], features[testidx]
        X_test -= X_train.mean(0)
        X_train -= X_train.mean(0)

        Y_train, Y_test = score[trainidx], score[testidx]		
        # Linear regression
        regr = sklin.LogisticRegression(solver='newton-cg', max_iter=1000)
        regr.fit(X_train, Y_train)
        # Predicted score
        P = regr.predict_proba(X_test)
        pred.append(P)

    pred = np.array(pred)
    pred = pred[:, :, 1]
    return pred.flatten()


def logreg_group(features, score, groups=None):
    pred = []
    if groups is None:
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 
                           9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 
                           15, 16, 16, 17, 17, 18, 19, 19])
    # Leave one out split
    loo = LeaveOneOut()	
    logo = LeaveOneGroupOut()
    logo.get_n_splits(features, score, groups)
    logo.get_n_splits(groups=groups)  # 'groups' is always required
    
    for trainidx, testidx in logo.split(features, score, groups):
        # Indices
        X_train, X_test = features[trainidx], features[testidx]
        X_test -= X_train.mean(0)
        X_train -= X_train.mean(0)

        Y_train, Y_test = score[trainidx], score[testidx]		
        # Linear regression
        regr = sklin.LogisticRegression(solver='newton-cg', max_iter=1000)
        regr.fit(X_train, Y_train)
        # Predicted score
        P = regr.predict_proba(X_test)
        pred.append(P)

    # pred = np.array(pred)
    # pred = pred[:,:,1]
    
    predflat = []
    for group in pred:
        for p in group:
            predflat.append(p)
            
    return np.array(predflat)[:, 1]


# Bilinear interpolation (new)
def imbilinear(im, col, x, row, y):
    """Takes bilinear interpolation from image.
    Starts from coordinates [y,x], ends at row,col.
    See Wikipedia article for bilinear interpolation."""
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y))
    q11 = im[y1:y1+row, x1:x1+col]
    q21 = im[y1:y1+row, x2:x2+col]
    q12 = im[y2:y2+row, x1:x1+col]
    q22 = im[y2:y2+row, x2:x2+col]
    r1 = ((x2-x)/(x2-x1+1e-12))*q11+((x-x1)/(x2-x1+1e-12))*q21
    r2 = ((x2-x)/(x2-x1+1e-12))*q12+((x-x1)/(x2-x1+1e-12))*q22
    p = ((y2-y)/(y2-y1+1e-12))*r1+((y-y1)/(y2-y1+1e-12))*r2
    return p


def MRELBP(im, parameters):
    """ Takes Median Robust Extended Local Binary Pattern from image im
    Uses n neighbours from radii r_large and r_small, r_large must be larger than r_small
    Median filter uses kernel sizes weight_center for center pixels, w_r[0] for larger radius and w_r[1]
    #or smaller radius
    Grayscale values are centered at their mean and scales with global standad deviation
    """

    n = parameters['N']
    r_large = parameters['R']
    r_small = parameters['r']
    weight_center = parameters['wc']
    weight_large = parameters['wl']
    weight_small = parameters['ws']

    # Mean grayscale value and std
    mean_image = im.mean()
    std_image = im.std()

    # Centering and scaling with std
    image = (im-mean_image)/std_image

    # Median filtering
    image_center = medfilt(image, weight_center)
    # Center pixels
    dist = round(r_large+(weight_large-1)/2)
    image_center = image_center[dist:-dist, dist:-dist]
    # Subtracting the mean pixel value from center pixels
    image_center = image_center-image_center.mean()
    # Binning center pixels
    center_hist = np.zeros((1, 2))
    center_hist[0, 0] = np.sum(image >= 0)
    center_hist[0, 1] = np.sum(image_center < 0)

    # --------------- #
    # center_hist[0,0] = np.sum(image_center>=-1e-06)
    # center_hist[0,1] = np.sum(image_center<-1e-06)
    # --------------- #
    
    # Median filtered images for large and small radius
    image_large = medfilt(image, weight_large)
    image_small = medfilt2d(image, weight_small)

    # Neighbours
    pi = np.pi
    # Empty arrays for the neighbours
    row, col = np.shape(image_center)
    n_large = np.zeros((row, col, n))
    n_small = np.zeros((row, col, n))
    
    for k in range(n):
        # Angle to the neighbour
        theta = k * (-1 * 2 * pi / n)
        # Large neighbourhood
        x = dist + r_large * np.cos(theta)
        y = dist + r_large * np.sin(theta)
        if abs(x-round(x)) < 1e-06 and abs(y-round(y)) < 1e-06:
            x = int(round(x))
            y = int(round(y))
            p = image_large[y:y+row, x:x+col]
        else:
            p = imbilinear(image_large, col, x, row, y)
        n_large[:, :, k] = p
        # Small neighbourhood
        x = dist+r_small*np.cos(theta)
        y = dist+r_small*np.sin(theta)
        if abs(x-round(x)) < 1e-06 and abs(y-round(y)) < 1e-06:
            x = int(round(x))
            y = int(round(y))
            p = image_small[y:y+row, x:x+col]
        else:
            p = imbilinear(image_small, col, x, row, y)
        n_small[:, :, k] = p

    # Thresholding radial neighbourhood
    n_radial = n_large-n_small

    # Subtraction of means
    mean_large = n_large.mean(axis=2)
    mean_small = n_small.mean(axis=2)
    for k in range(n):
        n_large[:, :, k] = n_large[:, :, k]-mean_large
        n_small[:, :, k] = n_small[:, :, k]-mean_small

    # Converting to binary images and taking the lbp values

    # Initialization of arrays
    lbp_large = np.zeros((row, col))
    lbp_small = np.zeros((row, col))
    lbp_radial = np.zeros((row, col))

    for k in range(n):
        lbp_large = lbp_large+(n_large[:, :, k] >= 0)*2**k  # NOTE ACCURACY FOR THRESHOLDING!!!
        lbp_small = lbp_small+(n_small[:, :, k] >= 0)*2**k
        lbp_radial = lbp_radial+(n_radial[:, :, k] >= 0)*2**k
        # --------------- #
        # lbp_large = lbp_large+(n_large[:,:,k]>=-1e-06)*2**k  # NOTE ACCURACY FOR THRESHOLDING!!!
        # lbp_small = lbp_small+(n_small[:,:,k]>=-1e-06)*2**k
        # lbp_radial = lbp_radial+(n_radial[:,:,k]>=-1e-06)*2**k
        # --------------- #

    # Binning
    large_hist = np.zeros((1, 2**n))
    small_hist = np.zeros((1, 2**n))
    radial_hist = np.zeros((1, 2**n))
    for k in range(2**n):
        large_hist[0, k] = np.sum(lbp_large == k)
        small_hist[0, k] = np.sum(lbp_small == k)
        radial_hist[0, k] = np.sum(lbp_radial == k)

    # Mapping
    mapping = getmapping(n)
    large_hist = maplbp(large_hist, mapping)
    small_hist = maplbp(small_hist, mapping)
    radial_hist = maplbp(radial_hist, mapping)
    
    hist = np.concatenate((center_hist, large_hist, small_hist, radial_hist), 1)
    
    return hist.T, (lbp_large, lbp_small, lbp_radial)


def getmapping(N):
    # Defines rotation invariant uniform mapping for lbp of N neighbours
    newMax = N + 2
    table = np.zeros((1, 2**N))
    for k in range(2**N):
        # Binary representation of bin number
        binrep = np.binary_repr(k, N)
        # Convert string to list of digits
        i_bin = np.zeros((1, len(binrep)))
        for ii in range(len(binrep)):
            i_bin[0, ii] = int(float(binrep[ii]))
        # Rotation
        j_bin = np.roll(i_bin, -1)
        # uniformity
        numt = np.sum(i_bin != j_bin)
        # Binning
        if numt <= 2:
            b = np.binary_repr(k, N)
            c = 0
            for ii in range(len(b)):
                c = c+int(float(b[ii]))
            table[0, k] = c
        else:
            table[0, k] = N+1
    #num = newMax
    return table


# Apply mapping to lbp
def maplbp(bin, mapping):
    # Applies mapping to lbp bin
    # Number of bins in output
    N = int(np.max(mapping))
    # Empty array
    outbin = np.zeros((1, N+1))
    for k in range(N+1):
        # RIU indices
        M = mapping == k
        # Extract indices from original bin to new bin
        outbin[0, k] = np.sum(M*bin)
    return outbin


# Image padding
def impadding(im, padlength):
    row, col = np.shape(im)
    im_pad = np.zeros((row + 2 * padlength, col + 2 * padlength))

    # Center
    im_pad[padlength:-padlength, padlength:-padlength] = im
    plt.imshow(im_pad)
    plt.show()
    return im_pad


# Scikit PCA
def ScikitPCA(features, ncomp, whitening=False, solver='full'):
    pca = skdec.PCA(n_components=ncomp, svd_solver=solver, whiten=whitening, random_state=42)
    #pca = skdec.PCA(n_components=ncomp, svd_solver='full', random_state=42)
    score = pca.fit(features).transform(features)
    return pca, score

##Principal component analysis
#def PCA(features,ncomp):	
#    #Feature dimension, x=num variables,N=num observations
#    x,N = np.shape(features)
#    #Mean feature
#    mean_f = np.mean(features,axis=1)
#    #Centering
#    centrd = np.zeros((x,N))
#    for k in range(N):
#        centrd[:,k] = features[:,k]-mean_f
#
#    #PCs from covariance matrix if N>=x, svd otherwise
#    if False:
#        #Covariance matrix
#        Cov = np.zeros((x,x))
#        f = np.zeros((x,1))
#        for k in range(N):		
#            f[:,0] = centrd[:,k]
#            Cov = Cov+1/N*np.matmul(f,f.T)
#
#        #Eigen values
#        E,V = np.linalg.eig(Cov)		
#        #Sort eigenvalues and vectors to descending order
#        idx = np.argsort(E)[::-1]
#        V = np.matrix(V[:,idx])
#        E = E[idx]
#
#        for k in range(ncomp):						
#            s = np.matmul(V[:,k].T,centrd).T			
#            try:
#                score = np.concatenate((score,s),axis=1)
#            except NameError:
#                score = s
#            p = V[:,k]
#            try:
#                pcomp = np.concatenate((pcomp,p),axis=1)
#            except NameError:
#                pcomp = p
#    else:
#        #PCA with SVD
#        u,s,v = np.linalg.svd(centrd,compute_uv=1)
#        pcomp = v[:,:ncomp]
#        # Save results
#        writer = pd.ExcelWriter(r'C:\Users\sarytky\Desktop\trials' + r'\PCA_test.xlsx')
#        df1 = pd.DataFrame(centrd)
#        df1.to_excel(writer, sheet_name='dataAdjust')
#        df2 = pd.DataFrame(u)
#        df2.to_excel(writer, sheet_name='u')
#        df3 = pd.DataFrame(s)
#        df3.to_excel(writer, sheet_name='s')
#        df4 = pd.DataFrame(v)
#        df4.to_excel(writer, sheet_name='v')        
#        writer.save()
#        np.savetxt(r'C:\Users\sarytky\Desktop\trials' + '\\''dataAdjust_python.csv', centrd, delimiter=',')
#
#        score = np.matmul(u,s).T[:,1:ncomp]
#    return pcomp,score


# Local grayscale standardization
def localstandard(im, parameters):
    """Centers grayscales with Gaussian weighted mean"""
    # Unpack parameters
    w1 = parameters['ks1']
    w2 = parameters['ks2']
    sigma1 = parameters['sigma1']
    sigma2 = parameters['sigma2']

    # Gaussian kernels
    kernel1 = Gauss2D(w1, sigma1)
    kernel2 = Gauss2D(w2, sigma2)
    # Blurring
    blurred1 = scipy.ndimage.convolve(im, kernel1)
    blurred2 = scipy.ndimage.convolve(im, kernel2)
    # Centering grayscale values
    centered = im-blurred1
    # Standardization
    std = (scipy.ndimage.convolve(centered**2, kernel2))**0.5
    new_im = centered/(std+1e-09)
    return new_im


# Gaussian kernel
def Gauss2D(w, sigma):
    # Generates 2d gaussian kernel
    kernel = np.zeros((w, w))
    # Constant for centering
    r = (w-1)/2
    for ii in range(w):
        for jj in range(w):
            x = -((ii-r)**2+(jj-r)**2)/(2*sigma**2)
            kernel[ii, jj] = np.exp(x)
    # Normalizing the kernel
    kernel = 1/np.sum(kernel)*kernel
    return kernel


def loadbinary(path, datatype = np.int32):
    if datatype == np.float64:
        bytesarray = np.fromfile(path, dtype=np.int64)  # read everything as int32
    else:
        bytesarray = np.fromfile(path, dtype=np.int32)  # read everything as int32
    w = bytesarray[0]
    l = int((bytesarray.size - 1) / w)
    with open(path, "rb") as f:  # open to read binary file
        if datatype == np.float64:
            f.seek(8)  # skip first integer (width)
        else:
            f.seek(4)  # skip first integer (width)
        features = np.zeros((w, l))
        for i in range(w):
            for j in range(l):
                if datatype == np.int32:
                    features[i, j] = struct.unpack('<i', f.read(4))[0]  
                    # when reading byte by byte (struct), 
                    # data type can be defined with every byte
                elif datatype == np.float32:
                    features[i, j] = struct.unpack('<f', f.read(4))[0]  
                elif datatype == np.float64:
                    features[i, j] = struct.unpack('<d', f.read(8))[0]  
        return features


def loadbinaryweights(path):
    bytesarray64 = np.fromfile(path, dtype=np.int64)  # read everything as int64
    bytesarray32 = np.fromfile(path, dtype=np.int32)  # read everything as int32
    w = bytesarray32[0]
    ncomp = bytesarray32[1]
    with open(path, "rb") as f:  # open to read binary file
        f.seek(8)  # skip first two integers (width)
        eigenvec = np.zeros((w, ncomp))
        for i in range(w):
            for j in range(ncomp):
                eigenvec[i, j] = struct.unpack('<f', f.read(4))[0]  
        singularvalues = np.zeros(ncomp)
        for i in range(ncomp):
            singularvalues[i] = struct.unpack('<f', f.read(4))[0]  
        weights = np.zeros(ncomp)
        for i in range(ncomp):
            weights[i] = struct.unpack('<d', f.read(8))[0]
        mean = np.zeros(w)
        for i in range(w):
            mean[i] = struct.unpack('<d', f.read(8))[0]
        return w, ncomp, eigenvec, singularvalues, weights, mean


def writebinaryweights(path, ncomp, eigenvectors, singularvalues, weights, mean):
    # Input eigenvectors in shape: components, features
    with open(path, "wb") as f:
        f.write(struct.pack('<i', eigenvectors.shape[1]))  # Width
        f.write(struct.pack('<i', ncomp))  # Number of components
        # Eigenvectors 
        for j in range(eigenvectors.shape[1]):
            for i in range(eigenvectors.shape[0]):  # Write row by row, component at a time
                f.write(struct.pack('<f', eigenvectors[i, j]))
        # Singular values
        for i in range(singularvalues.shape[0]):
            f.write(struct.pack('<f', singularvalues[i]))
        # Weights
        for i in range(weights.shape[0]):
            f.write(struct.pack('<d', weights[i]))
        for i in range(mean.shape[0]):
            f.write(struct.pack('<d', mean[i]))
    return True


def writebinaryimage(path, image, dtype):
    with open(path, "wb") as f:
        f.write(struct.pack('<i', image.shape[0]))  # Width
        # Image values as float
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if dtype == 'float':
                    f.write(struct.pack('<f', image[i, j]))
                if dtype == 'double':
                    f.write(struct.pack('<d', image[i, j]))
                if dtype == 'int':
                    f.write(struct.pack('<i', image[i, j]))                    
    return True
