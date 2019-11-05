import numpy as np
from PIL import Image
from scipy import signal
from helper import rgb2gray
from corner_detector import corner_detector
from anms import anms
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_name = "test.jpg"
    # read in RGB matrix
    I = np.array(Image.open(file_name).convert('RGB'))
    print(I.shape)
    # convert the RGB into gray image
    im_gray = rgb2gray(I)
    # find the corner matrix
    cimg = corner_detector(im_gray)

    # print(cimg)
    # print(np.max(cimg[np.where(cimg > 0)]))
    # cimg = cv2.dilate(cimg,None)
    I1 = I.copy()
    I1[cimg>0]=[0,0,255]
    thresh1_len = np.sum(I1[cimg>0])
    plt.imshow(I1)
    plt.show()

    # ANMS
    max_pts = thresh1_len/10
    x,y,rmax = anms(cimg, max_pts)
    I2 = I.copy()
    I2[y,x]=[0,0,255]
    plt.imshow(I2)
    plt.show()

    # max_pts = int(thresh1_len/1000)
    max_pts = 4
    print(max_pts)
    x,y,rmax = anms(cimg, max_pts)
    I2 = I.copy()
    I2[y,x]=[0,0,255]
    plt.imshow(I2)
    plt.show()
