import cv2
import numpy as np

# read image
img = cv2.imread('exp_centered.png')
old_image_height, old_image_width, channels = img.shape

# create new image of desired size and color (blue) for padding
new_image_width = old_image_width
new_image_height = old_image_height * 2
color = (255,255,255)
result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

# compute center offset
x_center = (new_image_width - old_image_width) // 2
y_center = (new_image_height - old_image_height) // 2

# copy img image into center of result image
result[y_center:y_center+old_image_height, 
       x_center:x_center+old_image_width] = img

# view result
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save result
cv2.imwrite("exp_recentered.png", result)
