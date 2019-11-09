'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''
import numpy as np
from helper import rgb2gray, interp2
from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import scipy.misc
import cv2


def matching_plot(img_target, img_stitching, x_tar, y_tar, x_sti, y_sti, inlier_ind):
  I_copy1 = img_target.copy()
  I_copy2 = img_stitching.copy()
  x1_in = x_tar[inlier_ind]
  y1_in = y_tar[inlier_ind]
  x2_in = x_sti[inlier_ind]
  y2_in = y_sti[inlier_ind]
  x1_out = x_tar[inlier_ind == False]
  y1_out = y_tar[inlier_ind == False]
  x2_out = x_sti[inlier_ind == False]
  y2_out = y_sti[inlier_ind == False]

  I_copy1[y1_in, x1_in] = [255, 0, 0]
  I_copy2[y2_in, x2_in] = [255, 0, 0]
  I_copy1[y1_out, x1_out] = [0, 0, 255]
  I_copy2[y2_out, x2_out] = [0, 0, 255]

  H1, W1 = I_copy1.shape[:2]
  H2, W2 = I_copy2.shape[:2]
  matchPlot = np.zeros((max(H1, H2), W1 + W2, 3), np.uint8)
  matchPlot[:H1, :W1, :3] = I_copy1
  matchPlot[:H2, W1:W1 + W2, :3] = I_copy2
  x2_in = x2_in + W1

  points1 = list(zip(x1_in, y1_in))
  points2 = list(zip(x2_in, y2_in))
  for point1, point2 in list(zip(points1, points2)):
    matchPlot = cv2.line(matchPlot, point1, point2, [255, 0, 0], 1)
  plt.imshow(matchPlot)
  plt.show()

def feat_match_between_img(x1, y1, x2, y2, descs_target, descs_stitching,thresh):
  final_match1 = []
  final_match2 = []
  match1 = feat_match(descs_target, descs_stitching)
  match2 = feat_match(descs_stitching, descs_target)
  for ind, value in enumerate(match1):
    if value != -1 and match2[int(value)] == ind:
      final_match1.append(ind)
      final_match2.append(int(value))

  # ransac
  x1 = x1[final_match1]
  # print(type())
  y1 = y1[final_match1]
  x2 = x2[final_match2]
  y2 = y2[final_match2]
  H, inlier_ind = ransac_est_homography(x2, y2, x1, y1, thresh)

  return x1, y1, x2, y2, H, inlier_ind

def mymosaic(img_input):
  # Your Code Here
  img0 = img_input[0]
  img1 = img_input[1]
  img2 = img_input[2]
  # img_input = list(img_input)
  # N = len(img_input)
  # img1 = img_input[0]
  # for i in range(1,N):
  #   img2 = img_input[i]
  img_mosaic = oneMosaic(img0 = img0,img1 = img1,img2 = img2)

  scipy.misc.imsave('ImageStitchingResult.jpg', img_mosaic)
  return img_mosaic


def oneMosaic(img0, img1, img2): # put img2 onto img1 then put img3 onto img1
  # corner detector
  im_gray0 = rgb2gray(img0)
  im_gray1 = rgb2gray(img1)
  im_gray2 = rgb2gray(img2)
  cimg0 = corner_detector(im_gray0)
  cimg1 = corner_detector(im_gray1)
  cimg2 = corner_detector(im_gray2)

  # PLOTTING: CORNER HARRIS
  I_copy = img0.copy()
  I_copy[cimg0 > 0] = [255, 0, 0]
  plt.imshow(I_copy)
  plt.show()

  I_copy = img1.copy()
  I_copy[cimg1>0]=[255,0,0]
  plt.imshow(I_copy)
  plt.show()

  I_copy = img2.copy()
  I_copy[cimg2>0]=[255,0,0]
  plt.imshow(I_copy)
  plt.show()


  # anms
  max_pts0 = 200
  x0, y0, rmax0 = anms(cimg0, max_pts0)

  max_pts1 = 200 #need to be adjust to different images
  x1,y1,rmax1 = anms(cimg1, max_pts1)

  max_pts2 = 200
  x2,y2,rmax2 = anms(cimg2, max_pts2)



  # PLOTTING: ANMS
  I_copy = img0.copy()
  I_copy[y0, x0] = [255,0, 0]
  plt.imshow(I_copy)
  plt.show()

  I_copy = img1.copy()
  I_copy[y1,x1]=[255,0,0]
  plt.imshow(I_copy)
  plt.show()

  I_copy = img2.copy()
  I_copy[y2,x2]=[255,0,0]
  plt.imshow(I_copy)
  plt.show()

  # feat_desc
  descs0 = feat_desc(im_gray0, x0.flatten(), y0.flatten())
  descs1 = feat_desc(im_gray1, x1.flatten(), y1.flatten())
  descs2 = feat_desc(im_gray2, x2.flatten(), y2.flatten())


  ### Comupute H matrix
  ## RANSAC parameter
  thresh = 0.4

  # feat_match img0-->img1
  x10_aft_mat,y10_aft_mat,x0_aft_mat,y0_aft_mat,H0,inlier_ind0 = feat_match_between_img(x1,y1,x0,y0,descs1,descs0,thresh)
  # feat_match img2-->img1
  x12_aft_mat,y12_aft_mat,x2_aft_mat,y2_aft_mat,H1,inlier_ind1 = feat_match_between_img(x1,y1,x2,y2,descs1,descs2,thresh)


  # PLOTTING: MATCHING
  # matching_plot(img1, img0, x10_aft_mat, y10_aft_mat, x0_aft_mat, y0_aft_mat, inlier_ind0)
  matching_plot(img0, img1, x0_aft_mat, y0_aft_mat, x10_aft_mat, y10_aft_mat, inlier_ind0)
  matching_plot(img1, img2, x12_aft_mat, y12_aft_mat, x2_aft_mat, y2_aft_mat, inlier_ind1)

  # mosaicing img2 to img1
  def mosaicing_mapping(img_sti, H):
      Himg, Wimg, _ = img_sti.shape
      # indicate 4 augmented corner point in img2
      four_corner_1 = np.array([0,0,1])
      four_corner_2 = np.array([Wimg-1,0,1])
      four_corner_3 = np.array([0,Himg-1,1])
      four_corner_4 = np.array([Wimg-1,Himg-1,1])
      four_corner_aug_array = np.vstack((four_corner_1,four_corner_2,four_corner_3,four_corner_4)).T
      assert four_corner_aug_array.shape == (3,4)
      # get mapped 4 points in the img1(no pad yet)
      mapped_points = H @ four_corner_aug_array
      mapped_points_norm = np.round(mapped_points/mapped_points[2,]).astype(int)
      # mashgrid to get all the coor in a 4 sided area(no pad yet)
      x_min = np.min(mapped_points_norm[0,:])
      x_max = np.max(mapped_points_norm[0,:])
      y_min = np.min(mapped_points_norm[1,:])
      y_max = np.max(mapped_points_norm[1,:])

      x, y = np.meshgrid(np.arange(x_min,x_max+1), np.arange(y_min,y_max+1))
      square_points_matrix = np.vstack((x.flatten(), y.flatten())).T

      # construct triangulation
      four_corner_array = mapped_points_norm.T[:,0:2]
      Tri = Delaunay(four_corner_array)
      # index
      in_polygen_index = Tri.find_simplex(square_points_matrix)>=0
      in_polygen_points = square_points_matrix[in_polygen_index,:]
      # map back to the source img
      mapback_points = np.linalg.inv(H) @ \
                       np.hstack((in_polygen_points,np.ones((in_polygen_points.shape[0],1)))).T
      mapback_points_norm = mapback_points/mapback_points[2,]
      mapback_points_norm_x = mapback_points_norm[0,:].reshape(-1,1)
      mapback_points_norm_y = mapback_points_norm[1,:].reshape(-1,1)
      mapback_points_norm_x = np.round(mapback_points_norm_x).astype(int) #?
      mapback_points_norm_y = np.round(mapback_points_norm_y).astype(int) #?
      # mesh_mapback_x, mesh_mapback_y = np.meshgrid(mapback_points_norm_x, mapback_points_norm_y)

      # find the interp value for 3 channel
      interp_val0 = interp2(img_sti[:,:,0], mapback_points_norm_x, mapback_points_norm_y)
      interp_val1 = interp2(img_sti[:,:,1], mapback_points_norm_x, mapback_points_norm_y)
      interp_val2 = interp2(img_sti[:,:,2], mapback_points_norm_x, mapback_points_norm_y)

      return four_corner_array,in_polygen_points,interp_val0,interp_val1,interp_val2
  # attach the channel value
  # in_polygen_points_with_channel_value = \
    # np.hstack((in_polygen_points,interp_val0,interp_val1,interp_val2))

  img0_four_corner_array,img0_in_polygen_points,img0_interp_val0,img0_interp_val1,img0_interp_val2 = mosaicing_mapping(img0, H0)
  img2_four_corner_array,img2_in_polygen_points,img2_interp_val0,img2_interp_val1,img2_interp_val2 = mosaicing_mapping(img2, H1)

  # padding + plotting two images together
  Himg, Wimg, _ =  img1.shape
  pad_x = 20*Wimg
  pad_y = 20*Himg

  resultImage = img1.copy()
  resultImage = np.pad(resultImage, ((pad_y, pad_y), (pad_x, pad_x),(0,0)), 'constant')
  plt.imshow(resultImage)
  plt.show()

  img0_in_polygen_points_x = img0_in_polygen_points[:,0].flatten() + pad_x
  img0_in_polygen_points_y = img0_in_polygen_points[:,1].flatten() + pad_y

  resultImage[:,:,0][img0_in_polygen_points_y, img0_in_polygen_points_x] = img0_interp_val0.flatten()
  resultImage[:,:,1][img0_in_polygen_points_y, img0_in_polygen_points_x] = img0_interp_val1.flatten()
  resultImage[:,:,2][img0_in_polygen_points_y, img0_in_polygen_points_x] = img0_interp_val2.flatten()

  img2_in_polygen_points_x = img2_in_polygen_points[:,0].flatten() + pad_x
  img2_in_polygen_points_y = img2_in_polygen_points[:,1].flatten() + pad_y

  resultImage[:,:,0][img2_in_polygen_points_y, img2_in_polygen_points_x] = img2_interp_val0.flatten()
  resultImage[:,:,1][img2_in_polygen_points_y, img2_in_polygen_points_x] = img2_interp_val1.flatten()
  resultImage[:,:,2][img2_in_polygen_points_y, img2_in_polygen_points_x] = img2_interp_val2.flatten()

  # alpha-blending the overlapping parts
  # identify the overlapping area
  Himg1, Wimg1, _ = img1.shape
  # indicate 4 corner point in img1
  img1_four_corner_1 = np.array([0,0])
  img1_four_corner_2 = np.array([Wimg1-1,0])
  img1_four_corner_3 = np.array([0,Himg1-1])
  img1_four_corner_4 = np.array([Wimg1-1,Himg1-1])
  four_corner_aug_array = np.vstack((img1_four_corner_1,img1_four_corner_2,\
                                     img1_four_corner_3,img1_four_corner_4))

  img1_bound_x_min = np.min(four_corner_aug_array[:,0])
  img1_bound_x_max = np.max(four_corner_aug_array[:,0])
  img1_bound_y_min = np.min(four_corner_aug_array[:,1])
  img1_bound_y_max = np.max(four_corner_aug_array[:,1])


  logic_x = np.logical_and(img1_bound_x_min<=img0_in_polygen_points[:,0].flatten(),\
                         img0_in_polygen_points[:,0].flatten()<=img1_bound_x_max)

  logic_y = np.logical_and(img1_bound_y_min<=img0_in_polygen_points[:,1].flatten(),\
                         img0_in_polygen_points[:,1].flatten()<=img1_bound_y_max)
  logic = np.logical_and(logic_x,logic_y)
  index_overlap = np.array(np.where(logic)).flatten()


  # overlapping part -0.5 of img1 and img2_interp

  resultImage[:,:,0][img0_in_polygen_points_y[index_overlap],img0_in_polygen_points_x[index_overlap]] = \
    (np.array(resultImage[:,:,0][img0_in_polygen_points_y[index_overlap],img0_in_polygen_points_x[index_overlap]])*0.8).astype(np.uint8)
  resultImage[:,:,1][img0_in_polygen_points_y[index_overlap],img0_in_polygen_points_x[index_overlap]] = \
    (np.array(resultImage[:,:,1][img0_in_polygen_points_y[index_overlap],img0_in_polygen_points_x[index_overlap]])*0.8).astype(np.uint8)
  resultImage[:,:,2][img0_in_polygen_points_y[index_overlap],img0_in_polygen_points_x[index_overlap]] = \
    (np.array(resultImage[:,:,2][img0_in_polygen_points_y[index_overlap],img0_in_polygen_points_x[index_overlap]])*0.8).astype(np.uint8)


  logic_x = np.logical_and(img1_bound_x_min<=img2_in_polygen_points[:,0].flatten(),\
                         img2_in_polygen_points[:,0].flatten()<=img1_bound_x_max)

  logic_y = np.logical_and(img1_bound_y_min<=img2_in_polygen_points[:,1].flatten(),\
                         img2_in_polygen_points[:,1].flatten()<=img1_bound_y_max)
  logic = np.logical_and(logic_x,logic_y)
  index_overlap = np.array(np.where(logic)).flatten()


  # overlapping part -0.5 of img1 and img2_interp

  resultImage[:,:,0][img2_in_polygen_points_y[index_overlap],img2_in_polygen_points_x[index_overlap]] = \
    (np.array(resultImage[:,:,0][img2_in_polygen_points_y[index_overlap],img2_in_polygen_points_x[index_overlap]])*0.8).astype(np.uint8)
  resultImage[:,:,1][img2_in_polygen_points_y[index_overlap],img2_in_polygen_points_x[index_overlap]] = \
    (np.array(resultImage[:,:,1][img2_in_polygen_points_y[index_overlap],img2_in_polygen_points_x[index_overlap]])*0.8).astype(np.uint8)
  resultImage[:,:,2][img2_in_polygen_points_y[index_overlap],img2_in_polygen_points_x[index_overlap]] = \
    (np.array(resultImage[:,:,2][img2_in_polygen_points_y[index_overlap],img2_in_polygen_points_x[index_overlap]])*0.8).astype(np.uint8)



  plt.imshow(resultImage)
  plt.show()


  #crop image
  # find the square which contain the whole img

  Himg1, Wimg1, _ = img1.shape
  # indicate 4 corner point in img1
  img1_boundary_map_coor = four_corner_aug_array

  img0_boundary_map_coor = img0_four_corner_array
  img2_boundary_map_coor = img2_four_corner_array

  boundary_coor = np.vstack((img1_boundary_map_coor,img0_boundary_map_coor,img2_boundary_map_coor)).T
  boundary_coor[0,:] = boundary_coor[0,:]+pad_x
  boundary_coor[1,:] = boundary_coor[1,:]+pad_y
  boundary_x_min = np.min(boundary_coor[0,:])
  boundary_x_max = np.max(boundary_coor[0,:])
  boundary_y_min = np.min(boundary_coor[1,:])
  boundary_y_max = np.max(boundary_coor[1,:])

  resultImage = resultImage[boundary_y_min:boundary_y_max+1,\
                boundary_x_min:boundary_x_max+1,:]
  plt.imshow(resultImage)
  plt.show()
  return resultImage

