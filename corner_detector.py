# Harris corner detector based on opencv & numpy
# Author: EricBX
# Date: 2020.4

import os
import cv2
import numpy as np
import scipy.ndimage.filters
from skimage.feature import hog

def harris_detector(img, sigma=2, k=0.5, threshold=0.1):
    # 1. input image and convert it to gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img_gray',img_gray); cv2.waitKey(0);cv2.destroyAllWindows()

    # 2. compute image derivatives
    img_gray = np.asarray(img_gray, dtype=np.double)  # to avoid overflows
    [Ix, Iy] = np.gradient(img_gray)
    # cv2.imshow('Ix',Ix); cv2.imshow('Iy',Iy);cv2.waitKey(0);cv2.destroyAllWindows()

    # 3. compute M components as squares of derivates
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
    # cv2.imshow('Ixx',Ixx); cv2.imshow('Iyy',Iyy);cv2.imshow('Ixy',Ixy);cv2.waitKey(0);cv2.destroyAllWindows()

    # 4. Gaussian filter g() with sigma
    ksize = (sigma*4+1, sigma*4+1)
    GIxx = cv2.GaussianBlur(Ixx, ksize, sigma)
    GIyy = cv2.GaussianBlur(Iyy, ksize, sigma)
    GIxy = cv2.GaussianBlur(Ixy, ksize, sigma)
    # cv2.imshow('GIxx', GIxx);    cv2.imshow('GIyy', GIyy);    cv2.imshow('GIxy', GIxy);
    # cv2.waitKey(0);    cv2.destroyAllWindows()

    # 5. compute cornerness
    C = GIxx * GIyy - GIxy**2 - k * ((GIxx + GIyy)/2)**2

    # 6. threshold on C to pick high cornerness
    corner_threshold = C.max() * threshold

    # 7. non-maxima suppression to pick peaks
    corners_2D = np.zeros(C.shape)
    h = C.shape[0]
    w = C.shape[1]
    for i in range(1, h-2):
        for j in range(1, w-2):
            if C[i, j] > corner_threshold and C[i, j] == np.max(C[i-1:i+2, j-1:j+2]):
                corners_2D[i, j] = 1

    return corners_2D

def harris2list(corners_2D):
    corners_list = np.where(corners_2D==1)
    corners_list = np.vstack((corners_list[0],corners_list[1])).transpose()
    return corners_list

def merge_corners_and_image(img, corners):
    result = np.array(img)
    corners_blob = cv2.dilate(corners, None)
    result[np.where(corners_blob == 1)] = [0, 0, 255]
    return result

def estimate_scale(img, corners_2D, sigma_max):
    # bgr to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # to save the scale as radius of circle and plot it
    img_harris_scale = np.array(img)
    # get corners list
    corners_list = np.where(corners_2D==1)
    corners_list = np.vstack((corners_list[0],corners_list[1])).transpose()
    # compute the radius r for the patch at corner
    responses = np.zeros(len(corners_list))
    flags = np.zeros(len(corners_list))
    radii = np.zeros(len(corners_list))
    for sigma in range(1,sigma_max):
        # calculate the Laplacian of Gaussian
        LoG = scipy.ndimage.filters.gaussian_laplace(img_gray, sigma)
        for i in range(len(corners_list)):
            x = corners_list[i, 0]
            y = corners_list[i, 1]
            response = LoG[x, y]
            if response > responses[i]:
                responses[i] = response
            elif flags[i]==0:
                flags[i]=1
                radii[i] = sigma
                # print('response of corner [%d] = %d' % (i, sigma))
                cv2.circle(img_harris_scale,(y,x),sigma,[0,0,255])

    return img_harris_scale, radii

def estimate_orientation(img, img_harris_scale, radii, harris_list, k_scale=4):
    # k_scale = 4 # a fixed scale parameter
    img_HoG = np.array(img_harris_scale)
    # img_HoG2 = np.array(img)
    harris_orientations = np.zeros(len(harris_list))
    for k in range(len(harris_list)):
        c = harris_list[k]
        r = int(radii[k]) * k_scale
        patch = img[c[0]-r:c[0]+r+1, c[1]-r:c[1]+r+1]
        pixels = r*2+1
        if patch.shape[0]<pixels or patch.shape[1]<pixels:
            continue
        fd, hog_image = hog(patch, orientations=36, pixels_per_cell=(pixels, pixels), cells_per_block=(1, 1),
                                            visualize=True, multichannel=True)
        # visualization
        hog_image = hog_image * np.max(hog_image)**(-1)
        for i in range(r*2+1):
            for j in range(r*2+1):
                if hog_image[i,j]>(hog_image.max()/2):
                    img_HoG[c[0]-r+i, c[1]-r+j] = [0, hog_image[i,j]*255, 0]
                    harris_orientations[k] = np.arctan2(j-r, i-r)*360/np.pi
    return img_HoG, harris_orientations

if __name__ == '__main__':
    if not os.path.isdir('./results'):
        os.mkdir('./results')

    # prepare different versions of image
    img = cv2.imread('./images/plane.bmp')
    # translation version
    height, width = img.shape[:2]
    T = np.float32([[1, 0, width/4], [0, 1, height/4]])
    img_translation = cv2.warpAffine(img, T, (width, height))
    # cv2.imwrite('./images/img_translation.png', img_translation)
    # rotation version
    R = cv2.getRotationMatrix2D(center=(width/2, height/2), angle=45, scale=1.0)
    img_rotation = cv2.warpAffine(img, R, (width, height))
    # cv2.imwrite('./images/img_rotation.png', img_rotation)
    # scale version
    img_scale = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
    # cv2.imwrite('./images/img_scale.png', img_scale)
    # totally
    imgs = [img, img_translation, img_rotation, img_scale]
    imgs_name = ['origin', 'translation', 'rotation', 'scale']

    for i in range(4):
        img = imgs[i]
        cv2.imshow('img', img)
        # test harris corner detector
        harris = harris_detector(img,sigma=4, k=0.5, threshold=0.01)
        harris_list = harris2list(harris)
        print('\n=> totally %d corners detected by harris in img_%s' % (len(harris_list), imgs_name[i]), file=open('./exp4_log.txt', 'a'))
        cv2.imshow('harris', harris)
        cv2.imwrite('./results/harris_%s.png'%imgs_name[i], harris)
        # tmp = merge_corners_and_image(img, harris)
        # cv2.imshow('tmp', tmp)

        # test scale estimation
        img_harris_scale, radii = estimate_scale(img, harris, 20)
        # test orientation estimation
        img_HoG, orientations = estimate_orientation(img, img_harris_scale, radii, harris_list, k_scale=4)
        cv2.imshow('img_with_harris_corners+scales+orientations', img_HoG)
        cv2.imwrite('./results/result_%s.png'%imgs_name[i], img_HoG)

        # showing the list of neighborhoods with corresponding orientations
        print('corner \t radius \t orientation(degree)', file=open('./exp4_log.txt', 'a'))
        for j in range(len(harris_list)):
            print('(%d, %d) \t %d \t %.1f' % (harris_list[j,0], harris_list[j,1], radii[j], orientations[j]), file=open('./exp4_log.txt', 'a'))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ================================
    # separately testing

    # translation_harris = harris_detector(img_translation)
    # img_translation_harris = merge_corners_and_image(img_translation, translation_harris)
    # cv2.imshow('translation_harris', translation_harris)
    # cv2.imshow('img_translation_harris', img_translation_harris)
    # cv2.waitKey(0)
    #
    # rotation_harris = harris_detector(img_rotation)
    # img_rotation_harris = merge_corners_and_image(img_rotation, rotation_harris)
    # cv2.imshow('rotation_harris', rotation_harris)
    # cv2.imshow('img_rotation_harris', img_rotation_harris)
    # cv2.waitKey(0)
    #
    # scale_harris = harris_detector(img_scale)
    # img_scale_harris = merge_corners_and_image(img_scale, scale_harris)
    # cv2.imshow('scale_harris', scale_harris)
    # cv2.imshow('img_scale_harris', img_scale_harris)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
