import numpy as np
import scipy as sp
import time
import os, argparse, sys
from absl import app
from absl import flags
from absl import logging

from utils import *

from scipy.optimize import Bounds, minimize

import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

FLAGS = flags.FLAGS

def main(argv):


    #parse input
    N, N_l, d = FLAGS.N, FLAGS.N_l, FLAGS.d
    a, tol, lambda_1, lambda_2 = FLAGS.a,FLAGS.tol, FLAGS.lambda_1, FLAGS.lambda_2

    method, loss_l = FLAGS.method, FLAGS.loss_l
    l2 = FLAGS.l2 
    form = 'B'
    s1, s2 = 30, 40 # random seed
    bound = FLAGS.bound

    # form data
    N_u = N - N_l
    n_classes = 2
    x,y = mixture(d, N, a)
    idx = labeling(y, N_l, s1 = s1, s2 = s2)

    dataset = {'name':'mixGuassian',
                'n_classes':2,
                'n_samples': N,
                'X': x,
                'y': np.full(N, -1.),
                'y_true': y,
                'n_l':0,
                'q' : np.zeros((2,))
                }

    dataset['y'][idx] = y[idx]
    dataset,sorter = update_labeled_samples(dataset, idx)
    check_(dataset['y'][:N_l], dataset['y_true'][:N_l])
    y = dataset['y_true'].astype('int')
    
    #get laplacian
    k = 200
    Delta = get_Laplacian(dataset['X'], k)

    #optimization
    y_label = dataset['y_true'][:N_l]
    y_label = y_label.astype('int')
    y_l = np.zeros((n_classes, N_l), dtype ='float32')
    for i in range(N_l):
        y_l[y_label[i],i] = 1.0

    if bound:
        b_lower = np.zeros(n_classes * N, dtype= 'float64')
        b_lower[:]=0.0001
        b_upper = np.ones(n_classes * N, dtype = 'float64')
        b_upper[:] = 0.99999
        bounds = Bounds(b_lower, b_upper)
    option='optimize'
    args = (N_l, N_u, n_classes, y_l, loss_l,  l2, form, lambda_1, lambda_2, Delta, option)
    logging.info("\n d, N_l, N, loss_l, l2, form, lambda_1, lambda_2, bound, option \n {:d}, {:d}, {:d}, {:s}, {:d}, {:s}, {:f}, {:f}, {:d}, {:s}".format(
                d, N_l, N, loss_l, l2, form, lambda_1, lambda_2, bound, option))
    x0 = 0.5 * np.ones(n_classes * N, dtype = 'float64')
    if bound:
        res = minimize(loss_onehot, x0, args = args, method = method, jac = grad_onehot,tol = tol, options = {'disp': True}, bounds = bounds)
    else:
        res = minimize(loss_onehot, x0, args = args, method = method, jac = grad_onehot, tol = tol, options = {'disp': True})
    #evaluation
    f = res.x
    a = list(args)
    a[10] = 'out'
    args = tuple(a)
    loss = loss_onehot(f, *args)
    grad_norm = grad_onehot(f, *args)
    f = f.reshape((n_classes, N))
    y_est = np.argmax(f, axis = 0)
    acc, acc_l,acc_u = evaluate(y_est,y,N_l)

    print("loss:",loss)
    print("grad_norm:", grad_norm)
    logging.info("\n\n\n ACCURACY-total, labeled, unlabeled: {:f}, {:f}, {:f} \n\n\n".format(acc, acc_l, acc_u))




if __name__ == '__main__':
    flags.DEFINE_integer('N', 10000, 'total number.')
    flags.DEFINE_integer('d', 50, 'dimension.')
    flags.DEFINE_float('a', 0.23, 'mean of one gaussian.')
    flags.DEFINE_integer('N_l', 50, 'labeled budget.')

    flags.DEFINE_string('method','L-BFGS-B', 'method for optimizataion.')
    flags.DEFINE_float('tol', 1.e-10, 'tolerance for convergence.')
    flags.DEFINE_string('loss_l', 'SE', 'loss labeled.')
    flags.DEFINE_integer('l2', 0, 'include l2 regularization.')

    flags.DEFINE_float('lambda_1', 1.e0, 'lambda_1.')
    flags.DEFINE_float('lambda_2', 1.e-12, 'lambda_2.')
    flags.DEFINE_integer('bound', 0, 'bound')



    app.run(main)



