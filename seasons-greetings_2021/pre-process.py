#!/usr/bin/env python3

from PIL import Image
import numpy as np

import damask

im = np.fliplr(np.array(Image.open('star.png')).astype('int').T)
im = im.reshape(im.shape+(1,))

damask.Grid(im,im.shape).save('star')
