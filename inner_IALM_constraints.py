from utils import *


def inner_IALM_constraints(Dotau, J, S_J, tol=1e-4, c=1., mu=-5., max_iter=500):
    # well shit then I guess I need to check stuff mu=1.25/np.linalg.norm(Dotau)
    """
    inner_IALM_constraints will solve the programming:
    min ||A||_*+lambda*||E||_1   s.t.     Dotau+J*delta_tau=A+E and
                                          S*delta_tau=0
    via the Inexact ALM method.
    ---------------------------------input----------------------------------
    Dotau:            m-by-n image matrix.
    J:                m-by-n-by-p tenso
    S_J:              c-by-p matrix.
    inner_para:       parameters.
    --------------------------------output----------------------------------
    A:                m-by-n matrix, low-rank part.
    E:                m-by-n matrix, error part.
    delta_tau:        step of tau.
    f:                objective funtion value.
    """

    #prep data
    #if mu == -5.:
    mu=1.25/np.linalg.norm(Dotau, 2)

    m, n = Dotau.shape
    E = np.zeros((m, n))
    A = np.zeros((m, n))
    p = J.shape[2]
    delta_tau = np.zeros((p,1))

    J_vec = np.reshape(J, (m*n,p))
    Jo = J_vec

    J_vec = np.vstack((J_vec, S_J))

    pinv_J_vec = np.linalg.pinv(J_vec)

    inner_round = 0
    rho = 1.25
    lmbda = c / np.sqrt(m)

    Y_1 = Dotau

    norm_two = np.linalg.norm(Y_1, 2)
    norm_inf = np.linalg.norm(Y_1.reshape(m*n, 1), np.inf)/lmbda
    dual_norm = max(norm_two, norm_inf)
    Y_1 = Y_1 / dual_norm;
    Y_2 = np.zeros((S_J.shape[0], 1))
    d_norm = np.linalg.norm(Dotau, 'fro')
    error_sign = 0

    first_f = np.sum(np.linalg.svd(Dotau)[1])

    stop_criterion = 10*tol
    while (stop_criterion > tol) and (inner_round < max_iter):

        inner_round += 1
        temp_0 = Dotau + np.reshape(np.dot(Jo, delta_tau), (m,n)) + Y_1 / mu

        temp_1 = temp_0 - E
        U, S, V = np.linalg.svd(temp_1, full_matrices=False)
        S = np.diag(S)
        shrinkage_S =(S > 1/mu).astype(int) * (S - 1/mu)
        A = U.dot(shrinkage_S).dot(V)

        temp_2 = temp_0 - A
        E = (temp_2 > lmbda/mu).astype(int) * (temp_2 - lmbda/mu) + (temp_2 < -lmbda/mu).astype(int) * (temp_2 + lmbda/mu)
        f = np.sum(np.sum(np.abs(shrinkage_S)) + lmbda * np.sum(np.sum(np.abs(E))))
        temp_3 = A + E - Dotau - Y_1 / mu
        temp_3 = np.reshape(temp_3, (m*n,1))
        temp_3 = np.vstack((temp_3, -Y_2 / mu))

        delta_tau = pinv_J_vec.dot(temp_3)
        derivative_Y_1 = Dotau - A - E + np.reshape(Jo.dot(delta_tau), (m, n))
        derivative_Y_2 = S_J.dot(delta_tau)
        Y_1 = Y_1 + derivative_Y_1 * mu
        Y_2 = Y_2 + derivative_Y_2 * mu

        stop_criterion=np.sqrt(np.linalg.norm(derivative_Y_1, 'fro')**2 + np.linalg.norm(derivative_Y_2, 2)**2)/d_norm
        mu = mu * rho

    return A, E, delta_tau, f
