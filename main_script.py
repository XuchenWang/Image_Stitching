import numpy as np
from PIL import Image
from scipy import signal
from helper import rgb2gray
from corner_detector import corner_detector
from anms import anms
import cv2
import matplotlib.pyplot as plt
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from mymosaic import mymosaic

if __name__ == '__main__':
    file_name1 = "test.jpg"
    file_name2 = "test1.jpg"
    # read in RGB matrix
    I1 = np.array(Image.open(file_name1).convert('RGB'))
    I2 = np.array(Image.open(file_name2).convert('RGB'))
    # convert the RGB into gray image
    im_gray1 = rgb2gray(I1)
    im_gray2 = rgb2gray(I2)
    # find the corner matrix
    cimg1 = corner_detector(im_gray1)
    cimg2 = corner_detector(im_gray2)

    #     # testing:printing image
    # I_copy = I1.copy()
    # I_copy[cimg1>0]=[0,0,255]
    # thresh1_len = np.sum(I_copy[cimg1>0])
    # plt.imshow(I_copy)
    # plt.show()
    # I_copy = I2.copy()
    # I_copy[cimg2>0]=[0,0,255]
    # # thresh1_len = np.sum(I_copy[cimg2>0])
    # plt.imshow(I_copy)
    # plt.show()

    # ANMS
    max_pts1 = 50 #need to be adjust to different images
    x1,y1,rmax1 = anms(cimg1, max_pts1)
    max_pts2 = 100
    x2,y2,rmax2 = anms(cimg2, max_pts2)

    #     # testing:printing image
    # I_copy = I1.copy()
    # I_copy[y1,x1]=[0,0,255]
    # plt.imshow(I_copy)
    # plt.show()
    # I_copy = I2.copy()
    # I_copy[y2,x2]=[0,0,255]
    # plt.imshow(I_copy)
    # plt.show()

    # feat_desc
    descs1 = feat_desc(im_gray1, x1.flatten(), y1.flatten())
    descs2 = feat_desc(im_gray2, x2.flatten(), y2.flatten())


    # feat_match
    final_match1 = []
    final_match2 = []
    match1 = feat_match(descs1,descs2)
    print(match1)
    match2 = feat_match(descs2,descs1)
    print(match2)
    for ind,value in enumerate(match1):
        if value!=-1 and match2[int(value)] == ind:
            final_match1.append(ind)
            final_match2.append(int(value))
    print(final_match1)
    print(final_match2)



    # ransac
    thresh = 0.5
    x1 = x1[final_match1]
    # print(type())
    y1 = y1[final_match1]
    x2 = x2[final_match2]
    y2 = y2[final_match2]
    H, inlier_ind = ransac_est_homography(x1, y1, x2, y2, thresh)
    print(H)
    print(inlier_ind)



    # mosaicing




