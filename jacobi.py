from utils import *


def jacobi(du, dv, XData, YData, tau, mode):
    m, n = du.shape
    [X0, Y0] = np.meshgrid(range(XData[0], XData[1]), range(YData[0], YData[1]))

    if mode == 'euclidean':
        tau = [tau]
        J = np.zeros((m, n, 1))
        J[:, :, 0] = du * (- X0 * np.sin(tau[0]) - Y0 * np.cos(tau[0])) + dv * (X0 * np.cos(tau[0]) - Y0 * np.sin(tau[0]))
    if mode == 'affine':
        J = np.zeros((m, n, 4))
        J[:, :, 0] = X0 * du
        J[:, :, 1] = Y0 * du
        J[:, :, 2] = X0 * dv
        J[:, :, 3] = Y0 * dv

    if mode == 'homography':
        H = para2tfm(tau, XData, YData, mode)
        N1 = H[0,0] * X0 + H[0,1] * Y0 + H[0,2]
        N2 = H[1,0] * X0 + H[1,1] * Y0 + H[1,2]
        N = H[2,0] * X0 + H[2,1] * Y0 + 1

        N = N.astype(float)
        N1 = N1.astype(float)
        N2 = N2.astype(float)

        dIdH = np.zeros((m, n, 8))
        dIdH[:, :, 0] = du * X0 /N
        dIdH[:, :, 1] = du * Y0 / N
        dIdH[:, :, 2] = du / N
        dIdH[:, :, 3] = dv * X0 / N
        dIdH[:, :, 4] = dv * Y0 / N
        dIdH[:, :, 5] = dv / N
        dIdH[:, :, 6] = du * (-N1 / (N * N) * X0) + dv * (-N2 / (N * N) * X0)
        dIdH[:, :, 7] = du * (-N1 / (N * N) * Y0) + dv * (-N2 / (N * N) * Y0)

        dPdH = np.zeros((8, 8))
        X = np.array([XData[0], XData[1], XData[1], XData[0]])
        Y = np.array([YData[0], YData[0], YData[1], YData[1]])
        N1 = X * H[0,0] + Y * H[0,1] + H[0,2]
        N2 = X * H[1,0] + Y * H[1,1] + H[1,2]
        N = X * H[2,0] + Y * H[2,1] + 1

        N = N.astype(float)
        N1 = N1.astype(float)
        N2 = N2.astype(float)

        for i in range(4):
            dPdH[2 * i, 0] = X[i] / N[i]
            dPdH[2 * i, 1] = Y[i] / N[i]
            dPdH[2 * i, 2] = 1 / N[i]
            dPdH[2 * i, 6] = -N1[i] / (N[i] ** 2) * X[i]
            dPdH[2 * i, 7] = -N1[i] / (N[i] ** 2) * Y[i]
            dPdH[2 * i+1, 3] = X[i] / N[i]
            dPdH[2 * i+1, 4] = Y[i] / N[i]
            dPdH[2 * i+1, 5] = 1 / N[i]
            dPdH[2 * i+1, 6] = -N2[i] / (N[i] ** 2) * X[i]
            dPdH[2 * i+1, 7] = -N2[i] / (N[i] ** 2) * Y[i]

        dHdP = np.linalg.inv(dPdH)

        J = np.zeros((m, n, 8))
        for i in range(8):

            for j in range(8):
                J[:, :, i]=J[:, :, i] + dIdH[:, :, j]*dHdP[j, i]

    return J