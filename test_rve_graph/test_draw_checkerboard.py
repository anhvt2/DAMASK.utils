from PIL import Image

h = 500
w = 500

# You can also easily set the number of squares per row
number_of_square_across = 10

# You can also easily set the colors
color_one = (255, 255, 255)
color_two = (0, 128, 128)

length_of_square = h/number_of_square_across
length_of_two_squares = h/number_of_square_across*2

img = Image.new("RGB", (h, w), (255, 0, 0))  # create a new 15x15 image
pixels = img.load()  # create the pixel map

for i in range(h):
    # for every 100 pixels out of the total 500 
    # if its the first 50 pixels
    if (i % length_of_two_squares) >= length_of_square:
        for j in range(w):
            if (j % length_of_two_squares) < length_of_square:
                pixels[i,j] = color_one
            else:
                pixels[i,j] = color_two

    # else its the second 50 pixels         
    else:
        for j in range(w):
            if (j % length_of_two_squares) >= length_of_square:
                pixels[i,j] = color_one
            else:
                pixels[i,j] = color_two

print("done")
img.show()
