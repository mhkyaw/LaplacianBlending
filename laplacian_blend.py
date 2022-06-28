import sys
import cv2
import numpy as np
import project2_util


##########################################################################################
#This function takes in an image and constructs a laplacian pyramid

def pyr_build(image):
    #initialize a list to hold successive iterations of the pyramid
    laplace = []

    #set the first level to the input image
    Gi = image

    #loop over and create successive levels of the pyramid
    while(True):
        #End the loop once we have reached an appropriate maximum depth
        if(Gi.shape[0] < 16 or Gi.shape[1] < 16):
            laplace.append(Gi.astype(np.float32))
            break

        #Convolve Gi with a small Gaussian kernel
        Gi1 = cv2.pyrDown(Gi)

        #Get Gi dimensions to resize Gi1
        h, w = Gi.shape[0:2]

        #Add the current pryamid level in this iteration to the laplace list
        laplace.append(Gi.astype(np.float32) - cv2.pyrUp(Gi1 , dstsize = (w,h)).astype(np.float32))

        #Setup for next iteration
        Gi = Gi1

    return laplace

##########################################################################################
#This function takes in a laplacian pyramid and returns its respective image

def pyr_reconstruct(pyramid):
    n = len(pyramid)-1

    #initialize first reconstructed level
    rn = pyramid[n]

    #loop over all levels of the pyramid to reconstruct the original image
    while(n>0):

        #reconstruct previous level
        rnext = cv2.pyrUp(rn, dstsize = (pyramid[n-1].shape[1],pyramid[n-1].shape[0])) + pyramid[n-1]

        n = n-1

        #Setup for next iteration
        rn = rnext

    #Clip intensity values to 0,255 range then converting to uint8
    rn = np.clip(rn,0,255).astype(np.uint8)

    return rn

##########################################################################################
#This function takes in two laplacian pyramids and combines them with a given mask

def blend(pyramid1, pyramid2, mask):
    #initialize a list for the resulting pyramid
    blend = []

    #loop over all the levels in the respective pyramids and combine them with a given mask
    for i in range(len(pyramid1)):
        h, w = pyramid1[i].shape[0:2]
        mask_i = cv2.resize(mask, (w,h), interpolation = cv2.INTER_AREA)
        lp_i = project2_util.alpha_blend(pyramid1[i], pyramid2[i], mask_i)

        blend.append(lp_i)

    return blend

##########################################################################################
def main():

    if len(sys.argv) != 5:
        print('usage: python laplacian_blend.py IMG1 IMG2 MASK RESULT')
        sys.exit(1)

    #Read in command line arguments and images
    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR)
    mask = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)
    result_filename = sys.argv[4]

    #Build laplacian pyramids of respective images
    result1 = pyr_build(img1)
    result2 = pyr_build(img2)

    #Construct the blended image by reconstructing the blended laplacian pyramids
    recon = pyr_reconstruct(blend(result1, result2, mask))

    #Show the blended image
    cv2.imshow("Blended Image", recon)
    cv2.waitKey(1000000)

    #Show respective pyramids for the two images
    output1 = project2_util.visualize_pyramid(result1)
    output2 = project2_util.visualize_pyramid(result2)

    window = 'Laplacian pyramid'
    project2_util.setup_window(window)
    project2_util.label_and_wait_for_key(window, output1,
                                         'Result (hit a key when done)')

    window = 'Laplacian pyramid'
    project2_util.setup_window(window)
    project2_util.label_and_wait_for_key(window, output2,
                                         'Result (hit a key when done)')
    cv2.imwrite(result_filename, recon)

if __name__ == '__main__':
    main()
