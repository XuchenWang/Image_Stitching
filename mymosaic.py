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
from PIL import Image
import matplotlib.pyplot as plt

def mymosaic(img_input):
  # Your Code Here
  img_input = list(img_input)
  N = len(img_input)
  img1 = img_input[0]
  for i in range(1,N):
    img2 = img_input[i]
    img1 = oneMosaic(img1,img2)

  img_mosaic = img1
  return img_mosaic


def oneMosaic(img1, img2): # input should be gray images
  im_gray1 = rgb2gray(img1)
  im_gray2 = rgb2gray(img2)
  cimg1 = corner_detector(im_gray1)
  cimg2 = corner_detector(im_gray2)

  max_pts1 = 200 #need to be adjust to different images
  x1,y1,rmax1 = anms(cimg1, max_pts1)
  max_pts2 = 200
  x2,y2,rmax2 = anms(cimg2, max_pts2)


        # testing:printing image
  I_copy = I1.copy()
  I_copy[y1,x1]=[0,0,255]
  plt.imshow(I_copy)
  plt.show()
  I_copy = I2.copy()
  I_copy[y2,x2]=[0,0,255]
  plt.imshow(I_copy)
  plt.show()


    # feat_desc
  descs1 = feat_desc(im_gray1, x1.flatten(), y1.flatten())
  descs2 = feat_desc(im_gray2, x2.flatten(), y2.flatten())

    # feat_match
  final_match1 = []
  final_match2 = []
  match1 = feat_match(descs1,descs2)
  match2 = feat_match(descs2,descs1)
  for ind,value in enumerate(match1):
      if value!=-1 and match2[int(value)] == ind:
          final_match1.append(ind)
          final_match2.append(int(value))

  print('final_match1',final_match1)
  print('final_match2',final_match2)


    # ransac
  thresh = 0.5
  x1 = x1[final_match1]
  # print(type())
  y1 = y1[final_match1]
  x2 = x2[final_match2]
  y2 = y2[final_match2]
  H, inlier_ind = ransac_est_homography(x2, y2, x1, y1, thresh)

# testing: printing image
  I_copy1 = I1.copy()
  I_copy2 = I2.copy()
  x1_in = x1[inlier_ind]
  y1_in = y1[inlier_ind]
  x2_in = x2[inlier_ind]
  y2_in = y2[inlier_ind]
  x1_out = x1[inlier_ind==False]
  y1_out = y1[inlier_ind==False]
  x2_out = x2[inlier_ind==False]
  y2_out = y2[inlier_ind==False]

  I_copy1[y1_in,x1_in]=[255,0,0]
  I_copy2[y2_in,x2_in]=[255,0,0]
  I_copy1[y1_out,x1_out]=[0,0,255]
  I_copy2[y2_out,x2_out]=[0,0,255]
  plt.imshow(I_copy1)
  plt.show()
  plt.imshow(I_copy2)
  plt.show()

  #mosaicing img2 to img1
  Himg, Wimg, _ = img2.shape
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

  interp_val0 = interp2(img2[:,:,0], mapback_points_norm_x, mapback_points_norm_y)
  interp_val1 = interp2(img2[:,:,1], mapback_points_norm_x, mapback_points_norm_y)
  interp_val2 = interp2(img2[:,:,2], mapback_points_norm_x, mapback_points_norm_y)

  # attach the channel value
  # in_polygen_points_with_channel_value = \
    # np.hstack((in_polygen_points,interp_val0,interp_val1,interp_val2))

  # padding + plotting two images together
  pad_x = 2*Wimg
  pad_y = 2*Himg

  resultImage = img1.copy()
  resultImage = np.pad(resultImage, ((pad_y, pad_y), (pad_x, pad_x),(0,0)), 'constant')
  plt.imshow(resultImage)
  plt.show()
  resultImage = resultImage
  plt.imshow(resultImage)
  plt.show()
  in_polygen_points_x = in_polygen_points[:,0].flatten() + pad_x
  in_polygen_points_y = in_polygen_points[:,1].flatten() + pad_y

  resultImage[:,:,0][in_polygen_points_y, in_polygen_points_x] = interp_val0.flatten()
  resultImage[:,:,1][in_polygen_points_y, in_polygen_points_x] = interp_val1.flatten()
  resultImage[:,:,2][in_polygen_points_y, in_polygen_points_x] = interp_val2.flatten()
  plt.imshow(resultImage)
  plt.show()


  #crop image
  # find the square which contain the whole img

  Himg1, Wimg1, _ = img1.shape
  # indicate 4 corner point in img1
  img1_four_corner_1 = np.array([0,0])
  img1_four_corner_2 = np.array([Wimg1-1,0])
  img1_four_corner_3 = np.array([0,Himg1-1])
  img1_four_corner_4 = np.array([Wimg1-1,Himg1-1])
  four_corner_aug_array = np.vstack((img1_four_corner_1,img1_four_corner_2,\
                                     img1_four_corner_3,img1_four_corner_4))

  img2_boundary_map_coor = four_corner_array

  boundary_coor = np.vstack((four_corner_aug_array,img2_boundary_map_coor)).T
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



if __name__ == '__main__':
  I1 = np.array(Image.open("left.jpg").convert('RGB'))
  I2 = np.array(Image.open("middle.jpg").convert('RGB'))

  input_img = [I1,I2]
  mymosaic(input_img)
