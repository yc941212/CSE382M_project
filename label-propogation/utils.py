import numpy as np
import os, sys
from numpy import random, linalg

import scipy as sp
import os, sys
from timeit import default_timer as timer

import cupy as cp
import cupyx
import cupyx.scipy.linalg as csl
import cupyx.scipy.sparse.linalg as cssl
import scipy.linalg as sl

import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def mixture(d, n, a):
    """
    generate gaussian mixtures
    """
    mu = np.zeros(d, dtype='float64')
    mu[:] = a
    num = n //2

    random.seed(0)
    x1 = random.normal(mu, size = (num,d))
    random.seed(100)
    x2 = random.normal(-mu, size = (num,d))
    x = np.concatenate((x1,x2))
    y = np.zeros(n, dtype='int')
    y[:num] = 0
    y[num:] = 1

    return x, y

def labeling(y, l, s1 = 30, s2 = 40):
    nl = l//2
    yc = np.where(y==0)[0]
    random.seed(s1)
    idx0 = random.choice(yc, nl, replace = False)
    yc =np.where(y==1)[0]
    random.seed(s2)
    idx1 = random.choice(yc, nl, replace = False)

    return np.concatenate((idx0,idx1))


def update_labeled_samples(dataset, class_indices, pseudocound=1, n_samples=None, adjust_X = True, poolname = None):
    """ updates the labeled data set with the new samples, re-orders unknowns.
    """
    # if select
    print("# of selected samples: ", len(class_indices))
    if 'sel' in dataset:
        s = 'new samples selected: '
        for k in class_indices:
            i = dataset['idx'][k]
            dataset['sel'].append(i)
            s += '(' + str(k) +','+str(i)+')  '
        s += '\n'
        print(s)

    # update labeled set
    n_new = 0
    dataset['n_dupl'] = 0
    if isinstance(class_indices, dict):
        for c in range(len(class_indices)):
            dataset['y'][class_indices[c]] = dataset['y_true'][class_indices[c]];
            n_new += len(class_indices[c]);
    else:
        if n_samples is not None:
            already_labeled = 0;
            for idx in class_indices:
                if dataset['y'][idx] != -1:
                    already_labeled += 1;
                    continue;
                else:
                    dataset['y'][idx] = dataset['y_true'][idx];
                    n_new += 1
                if n_new >= n_samples:
                    print("sucessfully added {} new data points; skipped {} data points (reason: already labeled)".format(n_new, already_labeled))
                    dataset['n_dupl'] = already_labeled
                    break
        else:
            dataset['y'][class_indices] = dataset['y_true'][class_indices];
            n_new += len(class_indices);
        # print(" .. class {} with {}".format(dataset['y_true'][class_indices], class_indices));

    # re-order
    sorter = np.argsort(-dataset['y'])
    if poolname in dataset:
        s = np.argsort(sorter)
        ss = np.searchsorted(sorter, dataset[poolname], sorter=s)
        dataset[poolname] = s[ss]
        print("re-orderin pool set")
    if adjust_X:
        print("re-ordering X")
        dataset['X'] = np.copy(dataset['X'][sorter])
    dataset['y'] = np.copy(dataset['y'][sorter])
    dataset['y_true'] = np.copy(dataset['y_true'][sorter])
    if 'X_NN' in dataset:
        dataset['X_NN'] = dataset['X_NN'][sorter]
        #dataset['X_NN'] = np.copy(dataset['X_NN'][sorter])
        print("re-ordering NN features X_NN")
    if dataset['n_l'] + n_new != np.sum(dataset['y']!=-1):
        print("WARNING: some points have already been labeled, nl = {}".format(np.sum(dataset['y']==-1)))


    # add :=======
    if 'idx' in dataset:
        dataset['idx'] = np.copy(dataset['idx'][sorter])
    #=====
    if 'p' in dataset:
        dataset['p'] = np.copy(dataset['p'][sorter])

    dataset['n_l'] = np.sum(dataset['y']!=-1)
    # print("n_new: {}, n_l: {}, n_u: {}".format(n_new, dataset['n_l'], np.sum(dataset['y']==-1)))
    # estimate class proportions q
    for c in range(dataset['n_classes']):
        dataset['q'][c] = (np.sum(np.where(dataset['y']==c, 1, 0)) + pseudocound) / (dataset['n_l'] + pseudocound * dataset['n_classes'])
    if 'classes' in dataset:
        for c in range(dataset['n_classes']):
            dataset['classes'][c] = np.where(dataset['y_true'] == c)[0]
    return dataset, sorter

def check_(a,b):

    a = a.astype('int')
    b = b.astype('int')
    print("check:", np.allclose(a,b))

    return None



def get_Laplacian(X, n_neighbors, lgc = False):

    n_samples = X.shape[0]
    n = n_samples * n_neighbors
    n1 = n_samples+1
    print(" start get Laplacian")
    import cuml.neighbors as clnn
    with cp.cuda.Device(2):
        neigh = clnn.NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        _, indices = neigh.kneighbors(X)

        indices = cp.asarray(indices).astype('int')
        weights = cp.ones((n_samples, n_neighbors), dtype = 'float32')
        weights[:,0]=0.
        weights = weights.get().reshape((n,))
        indices = indices.get().reshape((n,)).astype('int')

    indptr = np.linspace(0,n,num=n1, dtype ='int')
    W = sp.sparse.csr_matrix((weights, indices, indptr), shape=(n_samples,n_samples))
    W = W.tolil()
    rows, cols = W.nonzero()
    W[cols, rows] = W[rows, cols]
    W = W.tocsr()
    D = W.sum(axis=0).A.ravel()
    D = np.power(D, -0.5)
    D = sp.sparse.diags(D)
    W = D.dot(W).dot(D)
    if lgc:
        print("lgc")
        return W

    I = sp.sparse.eye(n_samples, dtype = 'float32')
    Delta = I - W
    print(" finish get Laplacian")

    return Delta

def evaluate(y_est, y, N_l):
    acc = accuracy_score(y_est, y)
    acc_l = accuracy_score(y_est[:N_l], y[:N_l])
    acc_u = accuracy_score(y_est[N_l:], y[N_l:])
    return acc, acc_l, acc_u

def loss_onehot(f, *args):
    #10 args
    N_l, N_u, n_classes, y_l, loss_l, l2, form, lambda_1, lambda_2, Delta, option = args

    y_l = y_l.astype('float64')
    N = N_l + N_u
    f = f.reshape(n_classes, N)
    f_l = f[:, :N_l]
    f_u = f[:, N_l:]

    def softmax(x):
        x -= np.max(x, axis=0)
        P = np.exp(x) / np.sum(np.exp(x), axis = 0)
        return P

    # labeled:
    if loss_l == 'SE':
        L_l = 0.5 * (np.linalg.norm(f_l - y_l) **2.)
    if loss_l == 'CE':
        if form == 'A':
            P = np.copy(f_l)
        if form == 'B':
            P = softmax(f_l)
        L_l = y_l * np.log(P)
        L_l = -L_l.sum()

    # unlabeled:
    L_u = 0.5* f.dot(Delta.dot(f.T))
    L_u = lambda_1 * np.trace(L_u)
    if l2: # if include l2 of unlabeled data
        L_u += 0.5 * (np.linalg.norm(f_u) **2.)
    if lambda_2 > 1.e-10: # if include entropy
        if form == 'A':
            P = np.copy(f_u)
        if form == 'B':
            P = softmax(f_u)
        L_u2 = P * np.log(P)
        L_u -= (lambda_2 * L_u2.sum())

    # total:
    L = L_l + L_u
    if option == 'out':
        return {'l': L_l, 'u': L_u, 't': L}

    return L
def grad_onehot(f, *args):
    #10 args
    N_l, N_u, n_classes, y_l, loss_l, l2, form, lambda_1, lambda_2, Delta, option = args

    y_l = y_l.astype('float64')
    N = N_l + N_u
    f = f.reshape(n_classes, N)
    f_l = f[:, :N_l]
    f_u = f[:, N_l:]

    def softmax(x):
        x -= np.max(x, axis=0)
        P = np.exp(x) / np.sum(np.exp(x), axis = 0)
        return P

    # labeled:
    if loss_l == 'SE':
        g_l = f_l - y_l
    if loss_l == 'CE':
        if form == 'A':
            g_l = - (y_l/f_l)
        if form == 'B':
            P = softmax(f_l)
            g_l = P - y_l

    # unlabeled:
    laplacian = (Delta.dot(f.T)).T
    g_u = lambda_1 * laplacian[:, N_l:]
    g_l +=  (lambda_1 * laplacian[:, :N_l])
    if l2:
        g_u += f_u
    if lambda_2 > 1.e-10:
        if form == 'A':
            g_u -= lambda_2 * (1. + np.log(f_u))
        if form == 'B':
            P = softmax(f_u)
            PlogP = P * (1. + np.log(P))
            g_u2 = P * np.sum(PlogP, axis = 0)
            g_u2 -= PlogP
            g_u += (lambda_2 * g_u2)

    # total:
    g = np.concatenate((g_l, g_u), axis = 1).flatten()
    if option =='out':
        return {'l': np.linalg.norm(g_l), 'u': np.linalg.norm(g_u), 't': np.linalg.norm(g)}

    return g






