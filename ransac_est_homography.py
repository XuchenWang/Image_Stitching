'''
  File name: ransac_est_homography.py
  Author: Xuchen Wang
  Date created: Nov 3, 2019
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image.
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''

import numpy as np
import random
from math import sqrt
from est_homography import est_homography


def ransac_est_homography(x1, y1, x2, y2, thresh):  # x1, y1, x2, y2 should be numpy array
    k = 5000
    N = len(x1)
    H = np.zeros([3, 3])
    ones = np.ones([N, ])
    inlier_ind = np.zeros([N, ])
    for i in range(k):
        curr_inlier_ind = np.zeros([N, ])
        sample = random.sample(list(range(N)), 4)
        x = x1[sample]
        y = y1[sample]
        X = x2[sample]
        Y = y2[sample]
        curr_H = est_homography(x, y, X, Y)

        img1_coor_matrix = np.vstack((x1.flatten(), y1.flatten(), ones))
        img2_coor_matrix = np.vstack((x2.flatten(), y2.flatten()))

        img2_coor_matrix_hat = curr_H @ img1_coor_matrix
        img2_coor_matrix_hat = img2_coor_matrix_hat/img2_coor_matrix_hat[2,]
        Lambda = img2_coor_matrix_hat[0:2, :] - img2_coor_matrix
        Sigma = Lambda * Lambda
        d = np.sqrt(Sigma[0, :] + Sigma[1, :])
        curr_inlier_ind = (d - thresh) < 0

        if sum(curr_inlier_ind) > sum(inlier_ind):
            inlier_ind = curr_inlier_ind
            H = curr_H

    return H, inlier_ind


# if __name__ == '__main__':
#     x = np.array([1,2,3,4]).reshape(-1,1)
#     y = np.array([2,3,4,5]).reshape(-1,1)
#     X = np.array([1,2,3,4]).reshape(-1,1)
#     Y = np.array([2,3,4,5]).reshape(-1,1)
#     thresh = 0.5
#     H, inlier_ind = ransac_est_homography(x, y, X, Y, thresh)
#     print(H,inlier_ind)
