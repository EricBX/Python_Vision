# Hybrid images based on opencv & numpy 
# Reference: Oliva A, Torralba A, Schyns P G. Hybrid images[J]. ACM Transactions on Graphics (TOG), 2006, 25(3): 527-532.
# Author: EricBX
# Date: 2020.4

import cv2
import numpy as np
import os

def blur(img, ksize, sigma_):
    tmp = cv2.GaussianBlur(img, ksize, sigma_)  # directly using the Gaussian blur function in OpenCV
    return tmp

if __name__=='__main__':
    # mkdir
    if not os.path.isdir('./results'):
        os.mkdir('./results')

    # read images
    img1 = cv2.imread('./images/cat.bmp', cv2.IMREAD_COLOR)  # low frequency, far away
    img2 = cv2.imread('./images/dog.bmp', cv2.IMREAD_COLOR)  # high frequency, close-up

    # set Gaussian Pyramid levels:
    sigma0 = 2
    for scale in [1.0, 0.5, 0.25]:
        # resize
        height = int(img1.shape[0] * scale)
        width = int(img1.shape[1] * scale)
        img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)

        for sigma in [sigma0, 2*sigma0, 4*sigma0, 8*sigma0]:
            sigma = int(sigma/scale)
            # parameter setting
            ks = sigma*4+1
            ksize1 = (ks, ks)
            sigma1 = sigma
            ksize2 = (ks, ks)
            sigma2 = sigma

            # processing
            img1_blur = blur(img1, ksize1, sigma1)
            img2_blur = blur(img2, ksize2, sigma2)
            img1_blur = np.array(img1_blur, dtype=np.int)
            img2_blur = np.array(img2_blur, dtype=np.int)
            img2_high = img2 - img2_blur
            hybrid_img = (img1_blur + img2_high)/2
            hybrid_img[np.where(hybrid_img < 0)] = 0
            hybrid_img[np.where(hybrid_img > 255)] = 255

            # showing
            cv2.imshow('img1', img1)
            cv2.imshow('img2', img2)
            cv2.imshow('img1_blur', img1_blur.astype(np.uint8))
            cv2.imshow('img2_blur', img2_blur.astype(np.uint8))
            cv2.imshow('img2 - img2_blur', img2_high.astype(np.uint8))
            cv2.imshow('hybrid_img', hybrid_img.astype(np.uint8))

            cv2.moveWindow('img1', 50, 50)
            cv2.moveWindow('img2', 50, 600)
            cv2.moveWindow('img1_blur', 600, 50)
            cv2.moveWindow('img2_blur', 600, 600)
            cv2.moveWindow('img2 - img2_blur', 1200, 600)
            cv2.moveWindow('hybrid_img', 1200, 50)

            # save results
            cv2.imwrite('./results/low_scale=%.2f_sigma=%d.png' % (scale, sigma), img1_blur.astype((np.uint8)))
            cv2.imwrite('./results/high_scale=%.2f_sigma=%d.png' % (scale, sigma), img2_high.astype((np.uint8)))
            cv2.imwrite('./results/hybrid_scale=%.2f_sigma=%d.png' % (scale, sigma), hybrid_img.astype((np.uint8)))

            cv2.waitKey(0)
            cv2.destroyAllWindows()