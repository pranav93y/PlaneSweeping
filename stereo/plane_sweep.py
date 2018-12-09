import cv2 as cv
from PIL import Image

import numpy as np

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


    homography = []

    for i in depths:
        projection_2 = __get_projection_mtx__(p2, n, i)
        projection_1 = __get_projection_mtx__(p1, n, i)
        m = np.dot(projection_1, np.linalg.inv(projection_2))
        h = m[0:3, 0:3]

        homography.append(h)

    return homography, depths


def get_warped_images(homography, display=ct.DONT_DISPLAY_PLOT):
    image2 = im.read(ct.POSE_WRITE_PATH+"undistorted2.jpg", COLOR=0)
    images = []
    for i, h in enumerate(homography):
        img = cv.warpPerspective(image2, h, None)
        images.append(img)

        if display:
            im.display("warped image", img)

        im.write(ct.POSE_WRITE_PATH+"warp/warped" + str(i) + ".jpg", img)

    return images


def run_abs_diff_and_block_filter(projection_1, projection_2, points_4d, intrinsic_matrix, display=ct.DONT_DISPLAY_PLOT):
    homography, depths = find_homography(projection_1, projection_2, points_4d, intrinsic_matrix)

    warped_images = get_warped_images(homography, display)
    image1 = im.read(ct.POSE_WRITE_PATH + "undistorted1.jpg", COLOR=0)

    diff = []

    for image in warped_images:
        diff_image = cv.blur(cv.absdiff(image, image1), (22, 25))
        if display:
            im.display("diff", diff_image)

        diff.append(diff_image)

    return diff, depths


def compute_depth(projection_1, projection_2, points_4d, intrinsic_matrix, display=ct.DONT_DISPLAY_PLOT):

    warped_abs_diff, depths = run_abs_diff_and_block_filter(projection_1, projection_2, points_4d, intrinsic_matrix, display)
    image1 = im.read(ct.POSE_WRITE_PATH + "undistorted1.jpg", COLOR=0)

    shp = image1.shape

    image1 = image1.ravel()
    depth_matrix = np.zeros(image1.shape)

    ravel_diff = []

    for image in warped_abs_diff:
        img = image.ravel()
        ravel_diff.append(img)

    mat = np.array(ravel_diff)

    for pixel in range(len(mat[0])):

        minimum = np.argmin(mat[:, pixel])

        depth_matrix[pixel] = round(255 * (depths[minimum]/max(depths)))

    depth_matrix = depth_matrix.reshape(shp)
    img = Image.fromarray(depth_matrix)

    img.show()

    im.write(ct.STEREO_WRITE_PATH + "depth.png", np.array(img))





