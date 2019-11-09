'''
  File name: corner_detector.py
  Author:Lishuo Pan
  Date created: Nov 3, 2019
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
from cv2 import cornerHarris
import numpy as np
def corner_detector(img):
  cimg = cornerHarris(img, 10, 3, 0.001)
  H,W = cimg.shape
  vect_cimg = cimg.flatten()
  vect_cimg[vect_cimg < np.max(vect_cimg)*0.01] = 0
  cimg = vect_cimg.reshape(H,W)

  return cimg
