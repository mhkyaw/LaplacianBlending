import sys
import cv2
import numpy as np
import project2_util

import laplacian_blend


def on_change(sigma):
    #since the slider goes from 0 to some number and sigma cant be 0, we add 1 to it.
    #we extract the low frequencies from the image
    A_lo = cv2.GaussianBlur(img1, (0,0), 2*(sigma+1))

    #We then extract the high frequences from the image by subtracting all the low frequencies
    A_hi = img1-A_lo

    #This then amplifies the high frequencies of the original image, "sharpening" it.
    sharpen = img1 + A_hi
    sharpen = np.clip(sharpen, 0, 255).astype(np.uint8)

    cv2.imshow("Image", sharpen)



def main():
    #Command land error handling
    if len(sys.argv) != 2:
        print('usage: python sharpen.py IMG1')
        sys.exit(1)

    #Read in the image
    global img1
    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR)

    cv2.imshow("Image", img1)

    #Turn image into float for arithmetic
    img1 = img1.astype(np.float32)

    #Trackbar for various sigma values
    cv2.resizeWindow('Resized Window', 300, 300)
    cv2.createTrackbar('Sigma value', "Image", 0, 50, on_change)

    #Press key to end program
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
