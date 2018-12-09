import cv2 as cv

import numpy as np
from matplotlib import pyplot as plt

from utility import constants as ct
from utility import imageutil as im


def __find_min_and_max_depth__(points_4d):
    return min(points_4d[2]), max(points_4d[2])


def __choose_eq_distances__(points_4d):
    min_depth, max_depth = __find_min_and_max_depth__(points_4d)
    print "min: " + str(min_depth) + " max: " + str(max_depth)

    depths = np.linspace(min_depth, max_depth, 20, dtype=float)
    return depths


def __get_projection_mtx__(projection, n, d):
    bottom = np.hstack((n, d))
    p = np.vstack((projection, bottom))
    return p


def find_homography(p1, p2, points_4d, intrinsic_matrix):
    depths = __choose_eq_distances__(points_4d)
    n = np.array([0, 0, -1])
    print depths

    homography = []

    for i in depths:
        projection_2 = __get_projection_mtx__(p2, n, i)
        projection_1 = __get_projection_mtx__(p1, n, i)
        m = np.dot(projection_1, np.linalg.inv(projection_2))
        h = m[0:3, 0:3]
        # t2 = np.matmul(intrinsic_matrix, t1)
        # h = np.matmul(t1, np.linalg.inv(intrinsic_matrix))
        homography.append(h)

    return homography


def get_warped_images(homography, display=ct.DONT_DISPLAY_PLOT):
    image2 = im.read(ct.POSE_WRITE_PATH+"undistorted2.jpg")
    image1 = im.read(ct.POSE_WRITE_PATH+"undistorted1.jpg")

    for i, h in enumerate(homography):
        img = cv.warpPerspective(image2, h, None)
        if display:
            im.display("warped image", img)

        im.write(ct.POSE_WRITE_PATH+"warp/warped" + str(i) + ".jpg", img)




