import sys
import cv2
import numpy as np
import project2_util

import laplacian_blend

def main():

    if len(sys.argv) != 6:
        print('usage: python hybrid.py IMG1 IMG2 SIGMA K RESULT')
        sys.exit(1)

    #Read in command line arguments and images
    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR)
    sigma = float(sys.argv[3])
    k = float(sys.argv[4])
    result_filename = sys.argv[5]

    #Convert to float for arithmetic
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    #Create low pass A image and high pass B image
    A_lo = cv2.GaussianBlur(img1, (0,0), 2*sigma)
    B_lo = cv2.GaussianBlur(img2, (0,0), sigma)
    B_hi = img2 - B_lo

    #Combine the low and high frequences of the respective images
    result = A_lo + k*B_hi
    result = np.clip(result, 0, 255).astype(np.uint8)
    cv2.imwrite(result_filename, result)

    #Visualize the laplacian pyramid of the hybrid result
    output1 = project2_util.visualize_pyramid(laplacian_blend.pyr_build(result))

    window = 'Hybrid image result'
    project2_util.setup_window(window)
    project2_util.label_and_wait_for_key(window, result,
                                         'Result (hit a key when done)')

    window = 'Laplacian Pyramid of Hybrid image result'
    project2_util.setup_window(window)
    project2_util.label_and_wait_for_key(window, output1,
                                         'Result (hit a key when done)')

if __name__ == '__main__':
    main()
