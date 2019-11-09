import numpy as np
from PIL import Image
from mymosaic import mymosaic
import scipy.misc

if __name__ == '__main__':
    I0 = np.array(Image.open("Test1_left.jpg").convert('RGB'))
    I0 = scipy.misc.imresize(I0, [300, 400])
    I1 = np.array(Image.open("Test1_middle.jpg").convert('RGB'))
    I1 = scipy.misc.imresize(I1, [300, 400])
    I2 = np.array(Image.open("Test1_right.jpg").convert('RGB'))
    I2 = scipy.misc.imresize(I2, [300, 400])
    input_img = [I0, I1, I2]
    mymosaic(input_img)


