import numpy as np
def tfm2para(tfm_matrix, XData, YData, mode):
    if mode == 'euclidean':
        tau = np.acos(tfm_matrix[0,0])
        if tfm_matrix[1,0] < 0:
            tau = -tau
    if mode == 'affine':
        tau = np.zeros((4,1))
        tau[:2] = np.transpose(tfm_matrix[0,:2])
        tau[2:] = np.transpose(tfm_matrix[1,:2])
    if mode == 'homography':
        X = [XData[0], XData[1], XData[1], XData[0]]
        Y = [YData[0], YData[0], YData[1], YData[1]]
        pt = [X,Y,np.ones((1, 4))]
        tfm_pt = np.dot(tfm_matrix,pt) ## not clear
        tfm_pt[0,:]=tfm_pt[0,:]/ tfm_pt[2,:]
        tfm_pt[1,:]=tfm_pt[1,:]/ tfm_pt[2,:]
        tau = np.reshape((tfm_pt[:2,:],(8,1)))
    return tau
    