import numpy as np
import skimage.filters as filt
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import sobel
from skimage import transform
from skimage import data, io, filters
from skimage import img_as_ubyte

def parse_TILT_arguments(**kwargs):

    my_args = kwargs

    if 'initial_tfm_matrix' not in my_args:
        my_args['initial_tfm_matrix'] = np.eye(3)

    if 'outer_tol' not in my_args:
        my_args['outer_tol'] = 1e-4

    if 'outer_max_iter' not in my_args:
        my_args['outer_max_iter'] = 50

    if 'outer_display_period' not in my_args:
        my_args['outer_display_period'] = 1

    if 'inner_tol' not in my_args:
        my_args['inner_tol'] = 1e-4

    if 'inner_c' not in my_args:
        my_args['inner_c'] = 1

    if 'inner_mu' not in my_args:
        my_args['inner_mu'] = []  #questionable

    if 'inner_display_period' not in my_args:
        my_args['inner_display_period'] = 100

    if 'inner_max_iter' not in my_args:
        my_args['inner_max_iter'] = np.inf

    if 'blur' not in my_args:
        my_args['blur'] = 1

    if 'branch' not in my_args:
        my_args['branch'] = 1

    if 'pyramid' not in my_args:
        my_args['pyramid'] = 1

    if 'focus_threshold' not in my_args:  #when doing pyramid the smallest focus_edge we can tolerate.
        my_args['focus_threshold'] = 50

    if 'outer_tol_step' not in my_args:  #as resolution goes high how relaxed should the outer_tol be
        my_args['outer_tol_step'] = 10

    if 'blur_kernel_size_k' not in my_args:  # neighbourhood scalar for the size of the blur kernel.
        my_args['blur_kernel_size_k'] = 2

    if 'blur_kernel_sigma_k' not in my_args: # standard derivation scalar for blur kernel.
        my_args['blur_kernel_sigma_k'] = 2

    if 'pyramid_max_level' not in my_args: # number of pyramid levels we want to act on.
        my_args['pyramid_max_level'] = 2

    if 'branch_max_iter' not in my_args: # in each branch, how much iteration we take.
        my_args['branch_max_iter'] = 10

    if 'branch_accuracy' not in my_args: # higher means smaller step-width.
        my_args['branch_accuracy'] = 5

    if 'branch_max_rotation' not in my_args:
        my_args['branch_max_rotation'] = np.pi / 6

    if 'branch_max_skew' not in my_args:
        my_args['branch_max_skew'] = 1

    if 'display_result' not in my_args:
        my_args['display_result'] = 1

    if 'focus_size' not in my_args:
        my_args['focus_size'] = []

    if 'save_path' not in my_args:
        my_args['save_path'] = []

    return my_args


def constraints(tau, XData, YData, mode):
    """constraints() will get the linearize constraints of tau according to mode.
    -----------------------------input--------------------------------------
    tau:          p-by-1 real vector.
    mode:         one of 'euclidean', 'euclidean_notranslation', 'affine', 'affine_notranslation', 'homography',
    'homography_notranslation'.
    ----------------------------output--------------------------------------
    linearized constraints on tau.

    """

    if mode == 'euclidean':
        S = np.zeros((2,1))

    elif mode == 'homography':
        S = np.zeros((1, 8))
        temp = tau.reshape(4, 2).T

        X= temp[0, :]
        Y = temp[1, :]

        e1 = np.vstack((X[2]-X[0], Y[2]-Y[0]))
        e2 = np.vstack((X[3]-X[1], Y[3]-Y[1]))

        norm_e1 = np.sum(e1 * e1)
        norm_e2 = np.sum(e2 * e2)
        e1e2 = np.sum(e1*e2)

        N = 2 * np.sqrt(norm_e1 * norm_e2 - e1e2**2)
        S[0, 0] = 1 / N * (2 * (X[2]-X[0])*(-1)*norm_e2 - 2*e1e2*(-1)*(X[3]-X[1]))
        S[0, 1] = 1 / N * (2*(Y[2]-Y[0])*(-1)*norm_e2 - 2*e1e2*(-1)*(Y[3]-Y[1]))
        S[0, 2] = 1 / N * (2*(X[3]-X[1])*(-1)*norm_e1 - 2*e1e2*(-1)*(X[2]-X[0]))
        S[0, 3] = 1 / N * (2*(Y[3]-Y[1])*(-1)*norm_e1 - 2*e1e2*(-1)*(Y[2]-Y[0]))
        S[0, 4] = 1 / N * (2*(X[2]-X[0])*norm_e2 - 2*e1e2*(X[3]-X[1]))
        S[0, 5] = 1 / N * (2*(Y[2]-Y[0])*norm_e2 - 2*e1e2*(Y[3]-Y[1]))
        S[0, 6] = 1 / N * (2*(X[3]-X[1])*norm_e1 - 2*e1e2*(X[2]-X[0]))
        S[0, 7] = 1 / N * (2*(Y[3]-Y[1])*norm_e1 - 2*e1e2*(Y[2]-Y[0]))
    return S

'''def para2tfm(tau, XData, YData, mode):
    """para2tfm will turn tau to tfm_matrix according to mode.
       ----------------------------input---------------------------------------
       tau:      p-by-1 vector
       mode:     one of 'euclidean', 'euclidean_notranslation', 'affine', 'affine_notranslation', 'homography',
                'homography_notranslation'
       ----------------------------output--------------------------------------
       tfm_matrix:   3-by-3 transform matrix.
    """
    tfm_matrix = np.eye(3)
    if mode == 'euclidean':
        tfm_matrix[0:2,0:2] = [[np.cos(tau[0]), -np.sin(tau[0])],
                               [np.sin(tau[0]), np.cos(tau[0])]]

    elif mode == 'affine':
        tfm_matrix[0,0:2] = tau[0:2].H
        tfm_matrix[1,0:2] = tau[2:].H

    else:
        print 'no param'

    return tfm_matrix
'''


def para2tfm(tau, XData, YData, mode):
    tfm_matrix = np.eye(3)

    if mode == 'euclidean':
        tfm_matrix[:2,:2]=np.array([[np.cos(tau[0]),-np.sin(tau[0])],[np.sin(tau[0]),np.cos(tau[0])]]).reshape(2,2)

    if mode == 'affine':
        tfm_matrix[0,:2] = np.transpose(tau[:2])
        tfm_matrix[1,:2] = np.transpose(tau[2:])
    if mode == 'homography':
        X = np.array([XData[0], XData[1], XData[1], XData[0]])
        Y = np.array([YData[0], YData[0], YData[1], YData[1]])

        temp = tau.reshape(8,1).reshape(4,2).T

        U = temp[0,:]
        V = temp[1,:]
        A = np.zeros((8, 8))
        b = np.zeros((8, 1))
        insert_A = np.zeros((2,8))
        insert_b = np.zeros((2,1))
        for i in range(4):
            insert_A[0,:]=[0, 0, 0, - X[i], - Y[i], - 1, V[i] * X[i], V[i] * Y[i]]
            insert_b[0] = -V[i]
            insert_A[1,:]=[X[i], Y[i], 1, 0, 0, 0, - U[i] * X[i], - U[i] * Y[i]]
            insert_b[1] = U[i]
            A[2 * i:2 * (i+1),:] = insert_A
            b[2 * i:2 * (i+1)] = insert_b

        solution = np.linalg.solve(A, b)
        #    solution = np.random.random(8).reshape(8, 1)
        tfm_matrix = np.reshape(np.vstack((solution, 1)),(3,3))

    return tfm_matrix


def tfm2para(tfm_matrix, XData, YData, mode):
    """tfm2para will transpose tfm_matrix to its corresponding parameter.
     -------------------------input------------------------------------------
     tfm_matrix:       3-by-3 matrix.
     mode:             one of 'euclidean', 'euclidean_notranslation', 'affine', 'affine_notranslation',
                       'homography', 'homography_notranslation'
     -------------------------output-----------------------------------------
     tau:              p-by-1 real vector.
    """
    if mode == 'euclidean':
        tau = np.arccos(tfm_matrix[0, 0])
        if tfm_matrix[1, 0] < 0:
            tau *= -1.
        tau = np.array(tau)

    elif mode =='affine':
        tau = np.zeros((4, 1))
        tau[0:2] = tfm_matrix[0,0:2].H
        tau[2:] = tfm_matrix[1,0:2].H

    elif mode == 'homography':
        X = np.array([XData[0], XData[1], XData[1], XData[0]]).reshape(1,4)
        Y = np.array([YData[0], YData[0], YData[1], YData[1]]).reshape(1,4)

        pt= np.vstack((X, Y, np.ones((1, 4))))
        tfm_pt=tfm_matrix.dot(pt)

        tfm_pt[0, :] = tfm_pt[0, :] / tfm_pt[2, :]
        tfm_pt[1, :] = tfm_pt[1, :] / tfm_pt[2, :]

        tau = tfm_pt[0:2, :].T.reshape(8, 1)

    return tau


def transform_point(input_pt, tfm_matrix):
    input_pt = input_pt.reshape(2)

    print input_pt
    pt = np.hstack((input_pt, 1))

    output_pt = tfm_matrix.dot(pt)
    output_pt /= float(output_pt[2])

    output_pt = output_pt[:2]

    return output_pt
