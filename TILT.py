from cv2 import GaussianBlur
from tilt_kernel import *
from utils import *
import os


def TILT(input_image, mode, init_points, **kwargs):
    args = parse_TILT_arguments(**kwargs)
    args['mode'] = mode
    args['input_image'] = input_image
    args['initial_points'] = np.floor(init_points)
    args['focus_size'] = np.asarray([init_points[1, 1]-args['initial_points'][1, 0], args['initial_points'][0, 1]-args['initial_points'][0, 0]])
    args['center'] = np.floor(np.mean(init_points, axis=1))
    args['focus_center'] = np.floor(np.asarray([args['focus_size'][1],  args['focus_size'][0]])/2.)
    focus_center = args['focus_center']
    XData = np.asarray([1-focus_center[0], args['focus_size'][1]-focus_center[0]])
    YData = np.asarray([1-focus_center[1], args['focus_size'][0]-focus_center[1]])

    image_center = args['center']
    image_size = input_image.shape
    UData = [1-image_center[0], image_size[1]-image_center[0]-1]
    VData = [1-image_center[1], image_size[0]-image_center[1]-1]

    X_ = XData
    Y_ = YData
    U_ = UData
    V_ = VData

    original_args = args

    # creating boundaries of the image
    expand_rate = 0.8
    initial_points = args['initial_points']
    left_bound = np.ceil(max(initial_points[0,0] - expand_rate * (initial_points[0,1] - initial_points[0,0]), 0))
    right_bound = np.floor(min(initial_points[0,1] + expand_rate * (initial_points[0,1] - initial_points[0,0]), input_image.shape[1]));
    top_bound = np.ceil(max(initial_points[1,0] - expand_rate * (initial_points[1,1] - initial_points[1,0]), 0))
    bottom_bound = np.floor(min(initial_points[1,1] + expand_rate*(initial_points[1,1] - initial_points[1,0]), input_image.shape[0]));
    new_image = np.zeros((bottom_bound - top_bound , right_bound - left_bound , input_image.shape[2]))

    for c in range(input_image.shape[2]):
        new_image[:, :, c] = input_image[top_bound:bottom_bound, left_bound:right_bound, c] #maybe miss one pixel? but whatever

    args['input_image'] = new_image

    args['center'] = args['center'] + np.asarray([1-left_bound, 1-top_bound])

    pre_scale_matrix=np.eye(3)

    min_length = original_args['focus_size']

    initial_tfm_matrix = args['initial_tfm_matrix']
    initial_tfm_matrix = np.linalg.inv(pre_scale_matrix).dot(initial_tfm_matrix).dot(pre_scale_matrix)
    args['initial_tfm_matrix'] = initial_tfm_matrix
    args['focus_size'] = np.around(args['focus_size']/pre_scale_matrix[0,0])
    args['center'] = args['center']/pre_scale_matrix[0,0]
    parent_path='./'

    if args['branch'] == 1:
        total_scale = 1  # whatever it means change to args
        downsample_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
        scale_matrix = np.linalg.matrix_power(downsample_matrix, total_scale)
        # if args['blur'] == 1:
        #    input_image = GaussianBlur(args['input_image'], (5,5), 4)
        # else:
        #    input_image = args['input_image']

        input_image = polina_transform(args['input_image'], scale_matrix, inv_flag=False)  # check

        # np.shape(input_image)[1]
        if input_image.shape[2] > 1:
            input_image = input_image[:, :, 0] * 0.144 + input_image[:, :, 1] * 0.587 + input_image[:, :, 2] * 0.299

        input_image = input_image.astype(float)

        initial_tfm_matrix = np.dot(np.dot(scale_matrix, args['initial_tfm_matrix']), np.linalg.inv(scale_matrix))
        center = np.floor(transform_point(args['center'], scale_matrix))  #### mad crazy functions
        focus_size = np.floor(args['focus_size'] / (2 ** total_scale))
        f_branch = np.zeros((3, 2 * args['branch_accuracy'] + 1))
        Dotau_branch = np.empty((3, 2 * args['branch_accuracy'] + 1, 3, 3))
        result_tfm_matrix = np.empty((3, 2 * args['branch_accuracy'] + 1, 3, 3))

        if args['mode'] == 'euclidean':
            max_rotation = args['branch_max_rotation']
            level = 1
            candidate_matrix = np.empty((1, 2 * args['branch_accuracy'] + 1, 3, 3))
            for i in range(2 * args['branch_accuracy'] + 1):
                candidate_matrix[0, i, :, :] = np.eye(3)
                theta = - max_rotation + (i) * max_rotation / float(args['branch_accuracy'])
                candidate_matrix[0, i, :2, :2] = np.array(
                    [[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        if args['mode'] == 'affine' or args['mode'] == 'homography':
            max_rotation = args['branch_max_rotation']
            max_skew = args['branch_max_skew']
            level = 3
            candidate_matrix = np.empty((3, 2 * args['branch_accuracy'] + 1, 3, 3))
            for i in range(2 * args['branch_accuracy'] + 1):
                candidate_matrix[0, i, :, :] = np.eye(3)
                theta = -max_rotation + (i) * max_rotation / float(args['branch_accuracy'])
                candidate_matrix[0, i, :2, :2] = np.array(
                    [[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                candidate_matrix[1, i, :, :] = np.eye(3)
                candidate_matrix[1, i, 0, 1] = -max_skew + (i) * max_skew / float(args['branch_accuracy'])
                candidate_matrix[2, i, :, :] = np.eye(3)
                candidate_matrix[2, i, 1, 0] = -max_skew + (i) * max_skew / float(args['branch_accuracy'])

        gap = 5
        BLACK_MATRIX = np.zeros((focus_size[0] * level + gap * (level - 1),
                                 focus_size[1] * (2 * args['branch_accuracy'] + 1) + gap * 2 * args['branch_accuracy']))

        normal_outer_max_iter = args['outer_max_iter']
        normal_display_inter = args['display_result']
        args['outer_max_iter'] = 1  # for debug, set it to 1;
        args['display_result'] = 0
        for i in range(level):
            for j in range(2 * args['branch_accuracy'] + 1):
                tfm_matrix = np.linalg.inv(
                    candidate_matrix[i, j, :, :].dot(np.linalg.inv(initial_tfm_matrix)))  ### don't want mess with this
                args['figure_no'] = (i - 1) * level + j
                args['save_path'] = []

                image_size = np.shape(input_image)
                image_center = np.floor(center)
                focus_center = np.zeros((2, 1))
                focus_center[0] = np.floor((1 + focus_size[1]) / 2)
                focus_center[1] = np.floor((1 + focus_size[0]) / 2)
                UData = np.array([1 - image_center[0], image_size[1] - image_center[0]])
                VData = np.array([1 - image_center[1], image_size[0] - image_center[1]])
                XData = np.array([1 - focus_center[0], focus_size[1] - focus_center[0]])
                YData = np.array([1 - focus_center[1], focus_size[0] - focus_center[1]])

                Dotau = polina_transform(input_image, tfm_matrix, UData, VData, XData, YData)

                Dotau = Dotau / np.linalg.norm(Dotau, 'fro')
                U, S, V = np.linalg.svd(Dotau)
                f = np.sum(S)
                start = [(focus_size[0] + gap) * (i) + 1, (focus_size[1] + gap) * (j) + 1]
                f_branch[i, j] = f
                result_tfm_matrix[i, j, :, :] = tfm_matrix
            index = np.argmin(f_branch[i, :])
            value = np.amin(f_branch[i, :])
            initial_tfm_matrix = result_tfm_matrix[i, index, :, :]

        initial_tfm_matrix = np.linalg.inv(scale_matrix).dot(initial_tfm_matrix).dot(scale_matrix)

    if args['pyramid'] == 1:
        downsample_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
        upsample_matrix = np.linalg.inv(downsample_matrix)
        total_scale = np.ceil(np.max(np.log2(np.min(args['focus_size'])/args['focus_threshold']), 0));

        for scale in range(int(total_scale),-1,-1):
            # begin each level of the pyramid
            if total_scale - scale >= args['pyramid_max_level']:
                break

            # Blur if required
            #BULA HERNIA
            if args['blur'] == 1 and scale == 0:
                input_image = GaussianBlur(args['input_image'], (5,5), 4) ##GIVE KERNEL
                #input_image=imfilter(args.input_image, fspecial('gaussian', ceil(args.blur_kernel_size_k*2^scale), ceil(args.blur_kernel_sigma_k*2^scale)));
            else:
                input_image = args['input_image']


            # prepare image and initial tfm_matrix
            scale_matrix = np.linalg.matrix_power(downsample_matrix, scale)
            #tfm = maketform('projective', scale_matrix');
            #input_image=imtransform(input_image, tfm, 'bicubic');

            input_image = polina_transform(input_image, scale_matrix, inv_flag = False)

            tfm_matrix = scale_matrix.dot(initial_tfm_matrix).dot(np.linalg.inv(scale_matrix))

            center = np.floor(transform_point(args['center'], scale_matrix));

            focus_size = np.floor(args['focus_size']/(2 ** scale))

            #args['save_path'] = fullfile(parent_path, ['pyramid', num2str(scale)]);
            #args.figure_no=100+total_scale-scale+1;
            Dotau, A, E, tfm_matrix, UData, VData, XData, YData, A_scale, Dotau_series = tilt_kernel(input_image, args['mode'], center, focus_size, tfm_matrix, args)
            # update tfm_matrix of the highest-resolution level.
            initial_tfm_matrix = np.linalg.inv(scale_matrix).dot(tfm_matrix).dot(scale_matrix)
            args['outer_tol']=args['outer_tol']*args['outer_tol_step']

        tfm_matrix=initial_tfm_matrix

    else:
        if args['blur'] == 1:
            img_size=np.shape(args['input_image'])
            img_size=img_size[:2]
            img_size=img_size[:2] # POLINA question: what is the point of this?
            gauss_kernel = np.ceil(args['blur_kernel_size_k']*max(img_size)/50).astype(int)
            if np.mod(gauss_kernel, 2) == 0:
                gauss_kernel += 1
            #if gauss_kernel == 1:
            #    gauss_kernel += 2   # Because GaussBlur with kernel size = 1 does not make sense
            # POLINA: I commented above because maybe we should forgo gauss for small images

            gauss_sigma = np.ceil(args['blur_kernel_sigma_k']*max(img_size)/50).astype(int)

            # POLINA: hopefully a more appropriate gaussian blur
            input_image = GaussianBlur(args['input_image'], (gauss_kernel, gauss_kernel), gauss_sigma)
        else:
            input_image = args['input_image']

        args['figure_no'] = 101
        args['save_path'] = os.path.join(parent_path, 'some_name')
        # POLINA: below seems to be a mistake
        Dotau, A, E, tfm_matrix, UData, VData, XData, YData, A_scale, Dotau_series = tilt_kernel(input_image,
                                                                                                 args['mode'],
                                                                                                 args['center'],
                                                                                                 args['focus_size'],
                                                                                                 initial_tfm_matrix,
                                                                                                 args)
    args = original_args
    tfm_matrix = np.dot(pre_scale_matrix,np.dot(tfm_matrix,np.linalg.inv(pre_scale_matrix)))

    focus_size = args['focus_size']
    image_size = np.shape(args['input_image'])
    image_size = image_size[:2]
    image_center = args['center']
    focus_center = np.zeros((2, 1))
    focus_center[0] = np.floor((1+args['focus_size'][1])/2)
    focus_center[1] = np.floor((1+args['focus_size'][0])/2)

    UData = np.array([1-image_center[0], image_size[1]-image_center[0]])
    VData = np.array([1-image_center[1], image_size[0]-image_center[1]])
    XData = np.array([1-focus_center[0], args['focus_size'][1]-focus_center[0]])
    YData = np.array([1-focus_center[1], args['focus_size'][0]-focus_center[1]])

    print 'focus_size', focus_size, 'focus_center', focus_center

    top_left = np.array([XData[0], YData[0]])
    top_right = np.array([XData[1], YData[0]])
    bottom_left = np.array([XData[0], YData[1]])
    bottom_right = np.array([XData[1], YData[1]])

    # Calculate borders for transformed frame
    tf_top_left = np.round(transform_point(top_left, tfm_matrix) + image_center).astype(int)
    tf_top_right = np.round(transform_point(top_right, tfm_matrix) + image_center).astype(int)
    tf_bottom_left = np.round(transform_point(bottom_left, tfm_matrix) + image_center).astype(int)
    tf_bottom_right = np.round(transform_point(bottom_right, tfm_matrix) + image_center).astype(int)

    # Borders of initial frame
    ac_top_left = np.round(top_left.T + image_center).flatten().astype(int)
    ac_top_right = np.round(top_right.T + image_center).flatten().astype(int)
    ac_bottom_left = np.round(bottom_left.T + image_center).flatten().astype(int)
    ac_bottom_right = np.round(bottom_right.T + image_center).flatten().astype(int)

    fig, ax = plt.subplots(3, 1)
    args['input_image'] = 255 - args['input_image']

    # draw frame of input selection on image
    cv2.line(args['input_image'], (ac_top_left[0], ac_top_left[1]), (ac_top_right[0], ac_top_right[1]),
                         (100, 250, 100))
    cv2.line(args['input_image'], (ac_top_left[0], ac_top_left[1]), (ac_bottom_left[0], ac_bottom_left[1]),
                         (100, 250, 100))
    cv2.line(args['input_image'], (ac_bottom_right[0], ac_bottom_right[1]), (ac_top_right[0], ac_top_right[1]),
                         (100, 250, 100))
    cv2.line(args['input_image'], (ac_bottom_right[0], ac_bottom_right[1]), (ac_bottom_left[0], ac_bottom_left[1]),
                         (100, 250, 100))

    # draw frame of output transform on image
    cv2.line(args['input_image'], (tf_top_left[0], tf_top_left[1]), (tf_top_right[0], tf_top_right[1]),
                         (100,100,250))
    cv2.line(args['input_image'], (tf_top_left[0], tf_top_left[1]), (tf_bottom_left[0], tf_bottom_left[1]),
                         (100, 100, 250))
    cv2.line(args['input_image'], (tf_top_right[0], tf_top_right[1]), (tf_bottom_right[0], tf_bottom_right[1]),
                         (100, 100, 250))
    cv2.line(args['input_image'], (tf_bottom_left[0], tf_bottom_left[1]), (tf_bottom_right[0], tf_bottom_right[1]),
                         (100, 100, 250))

    ax[0].imshow(args['input_image'])
    ax[1].imshow(Dotau_series[0], cmap='gray')
    ax[2].imshow(Dotau_series[-1], cmap='gray')

    plt.show()

    return Dotau_series, Dotau, A, E, tfm_matrix, U_, V_, X_, Y_ #UData, VData, XData, YData
