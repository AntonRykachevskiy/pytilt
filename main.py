import matplotlib.pyplot as plt
import numpy as np
import cv2


from TILT import TILT
from tilt_kernel import  polina_transform


class Img():
    def __init__(self, filename):
        self.fname = filename
        self.img = cv2.imread(self.fname)
        cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
        self.point = ()

    def getCoord(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(self.img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.point

    def __onclick__(self, click):
        self.point = (click.xdata, click.ydata)
        return self.point

    def getImg(self):
        return self.img

if __name__ == '__main__':

    inputt = Img('building1.jpg')

    img = inputt.getImg()
    top_left = np.around(inputt.getCoord()).astype(int)
    bottom_right = np.around(inputt.getCoord()).astype(int)

    init_points = np.zeros((2, 2))
    init_points[0, 0] = top_left[0]
    init_points[0, 1] = bottom_right[0]
    init_points[1, 0] = top_left[1]
    init_points[1, 1] = bottom_right[1]

    init_points = init_points.astype(int)

    #check = cv2.imread('building_.jpg')
    # init_points = np.asarray([[0, check.shape[1]], [0, check.shape[0]]])
    #init_points = np.asarray([[30, 100], [40, 80]])

    plt.imshow(img[init_points[1][0]: init_points[1][1], init_points[0][0]: init_points[0][1]])
    plt.show()

    Ds, Dotau, A, E, tfm_matrix, UData, VData, XData, YData = TILT(img, 'homography', init_points, blur=0, pyramid=1, branch=1)

    plt.imshow(A)
    plt.show()

    #plt.imshow(polina_transform(img, tfm_matrix, UData, VData, XData, YData, cut_flag=False))
    #plt.show()

    print np.sum(E)
