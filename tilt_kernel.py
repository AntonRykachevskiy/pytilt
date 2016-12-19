from jacobi import *
from inner_IALM_constraints import *
from utils import *
import cv2


def polina_transform(input_image, tfm_matrix, UData = None,
                     VData = None, XData = None, YData = None,
                     inv_flag = True, cut_flag = True):

    final = np.eye(3)

    #first, translate to a new center:
    u_trans = np.eye(3)
    UV_scale = np.eye(3)
    if not UData is None:
        u_trans[0,2] = UData[0]
        u_trans[1,2] = VData[0]

        UV_scale[0,0] = (UData[1] - UData[0]) / float(input_image.shape[1])
        UV_scale[1,1] = (VData[1] - VData[0]) / float(input_image.shape[0])

    # then apply the transform
    M = np.eye(3)
    if inv_flag:
        M = np.linalg.inv(tfm_matrix)
    else:
        M = tfm_matrix

    # then do a transform according to XData and YData
    x_trans = np.eye(3)
    if not XData is None:
        x_trans[0,2] = -XData[0]
        x_trans[1,2] = -YData[0]

    final = x_trans.dot(M).dot(u_trans).dot(UV_scale)

    if not XData is None and cut_flag:
        destination_size = (np.sum(np.abs(XData)).astype(int),np.sum(np.abs(YData)).astype(int))
    else:
        destination_size = (int(input_image.shape[1] * tfm_matrix[0, 0]), int(input_image.shape[0] *tfm_matrix[1, 1]))
        #print destination_size

    img_1 =  cv2.warpPerspective(input_image, final, destination_size)

    return img_1

def tilt_kernel(input_image, mode, center, focus_size, initial_tfm_matrix, para):

    outer_tol = 5e-5
    outer_max_iter = 500
    outer_display_period = 1

    if input_image.shape[2] > 1:
        input_image = input_image[:, :, 0]*0.299 + input_image[:, :, 1]*0.587 + input_image[:, :, 2]*0.144

    input_image = input_image.astype(float)

    image_center = np.floor(center)
    print 'im_c', image_center
    focus_size = np.floor(focus_size)
    print 'fs', focus_size
    image_size = input_image.shape

    print image_size
    focus_center = np.zeros((2,1))
    focus_center[0] = np.floor((focus_size[1])/2)
    focus_center[1] =np.floor((focus_size[0])/2)
    A_scale = 1

    UData = [1-image_center[0], image_size[1]-image_center[0]-1]
    VData = [1-image_center[1], image_size[0]-image_center[1]-1]

    # Polina Switched
    XData = [1-focus_center[0], focus_size[1]-focus_center[0]+1]

    YData = [1-focus_center[1], focus_size[0]-focus_center[1]+1]
    #YData = [1-focus_center[0], focus_size[1]-focus_center[0]+1]
    #XData = [1-focus_center[1], focus_size[0]-focus_center[1]+1]

    #inp_im = np.hstack((np.zeros((input_image.shape[0], 1)),input_image, np.zeros((input_image.shape[0],1))))
    #inp_im = np.vstack((np.zeros((1, input_image.shape[1] +2 )),inp_im, np.zeros((1, input_image.shape[1]  + 2))))
    #input_image = input_image.astype(np.uint8)

    #input_du = (inp_im[2:,:] - inp_im[:-2,:])[:,1:-1]
    #input_dv = (inp_im[:,2:] - inp_im[:,:-2])[1:-1, :]
    input_du = sobel(input_image, 1)
    input_dv = sobel(input_image, 0)

    Dotau_series = []

    tfm_matrix=initial_tfm_matrix

    Dotau = polina_transform(input_image, tfm_matrix, UData, VData, XData, YData)

    Dotau_series.append(Dotau)

    du = polina_transform(input_du, tfm_matrix, UData, VData, XData, YData)
    dv = polina_transform(input_dv, tfm_matrix, UData, VData, XData, YData)

    du = du / np.linalg.norm(Dotau, 'fro') - (sum(sum(Dotau*du))) / (np.linalg.norm(Dotau, 'fro')**3) * Dotau
    dv = dv / np.linalg.norm(Dotau, 'fro') - (sum(sum(Dotau*dv))) / (np.linalg.norm(Dotau, 'fro')**3) * Dotau

    A_scale = np.linalg.norm(Dotau, 'fro')
    Dotau = Dotau.astype(float) / np.linalg.norm(Dotau, 'fro')

    tau = tfm2para(tfm_matrix, XData, YData, mode)

    J = jacobi(du, dv, XData, YData, tau, mode)
    S = constraints(tau, XData, YData, mode)

    outer_round = 0
    pre_f = 0

    while 1:
        outer_round += 1
        A, E, delta_tau, f = inner_IALM_constraints(Dotau, J, S)

        tau = tau + delta_tau

        tfm_matrix = para2tfm(tau, XData, YData, mode)

        Dotau = polina_transform(input_image, tfm_matrix, UData, VData, XData, YData)

        Dotau_series.append(Dotau)
        # judge convergence
        if (outer_round >= outer_max_iter) or (np.abs(f - pre_f) < outer_tol):
            break

        pre_f = f
        du = polina_transform(input_du, tfm_matrix, UData, VData, XData, YData)
        dv = polina_transform(input_dv, tfm_matrix, UData, VData, XData, YData)

        du = du / np.linalg.norm(Dotau, 'fro') - (sum(sum(Dotau*du))) / (np.linalg.norm(Dotau, 'fro')**3) * Dotau
        dv = dv / np.linalg.norm(Dotau, 'fro') - (sum(sum(Dotau*dv))) / (np.linalg.norm(Dotau, 'fro')**3) * Dotau
        A_scale = np.linalg.norm(Dotau, 'fro')
        Dotau /= np.linalg.norm(Dotau, 'fro')

        J = jacobi(du, dv, XData, YData, tau, mode)
        S = constraints(tau, XData, YData, mode)

    return Dotau, A, E, tfm_matrix, UData, VData, XData, YData, A_scale, Dotau_series
