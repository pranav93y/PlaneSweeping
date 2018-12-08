import cv2 as cv

import numpy as np
from matplotlib import pyplot as plt

from parameters import  intrinsic as it

from utility import constants as ct
from utility import imageutil as im


def get_fundamental_matrix(intrinsic_matrix, distortion_parameters, display=ct.DONT_DISPLAY_PLOT):
    pts1, pts2, img1, img2 = __get_matching_points__(intrinsic_matrix, distortion_parameters, display)
    F, pts1, pts2 = __compute_fundamental_matrix__(pts1, pts2)

    return F, pts1, pts2, img1, img2


def display_epipolar_lines():
    return


def __get_matching_points__(intrinsic_matrix, distortion_parameters, display=ct.DONT_DISPLAY_PLOT):

    images = im.load_images_from_folder(ct.POSE_READ_PATH, COLOR=0)

    img1, img2 = __preprocess_(images, intrinsic_matrix, distortion_parameters, display)

    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    return pts1, pts2, img1, img2


def __compute_fundamental_matrix__(pts1, pts2):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    return F, pts1, pts2


def drawlines(img1, img2, lines, pts1, pts2):
    """
    img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines
    """
    r, c = img1.shape

    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)

    return img1, img2


def draw_epipolar_lines(F, img1, img2, pts1, pts2, display=ct.DONT_DISPLAY_PLOT):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    im.write(ct.POSE_WRITE_PATH + "epipolar_lines1.jpg", img5)
    im.write(ct.POSE_WRITE_PATH + "epipolar_lines2.jpg", img3)

    if display:
        plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img3)
        plt.show()


def __preprocess_(images, intrinsic_matrix, distortion_parameters, display=ct.DONT_DISPLAY_PLOT):
    undistorted1 = it.undistort(cv.resize(images[0], (ct.IMAGE_X, ct.IMAGE_Y)), intrinsic_matrix, distortion_parameters, "undistorted1.jpg", display)
    undistorted2 = it.undistort(cv.resize(images[1], (ct.IMAGE_X, ct.IMAGE_Y)), intrinsic_matrix, distortion_parameters, "undistorted2.jpg", display)

    # img1 = cv.resize(undistorted1, (ct.IMAGE_X, ct.IMAGE_Y))
    # img2 = cv.resize(undistorted2, (ct.IMAGE_X, ct.IMAGE_Y))

    return undistorted1, undistorted2

def __get_rotation_and_translation__(pts1, pts2, intrinsic_matrix):
    essential_matrix, _ = cv.findEssentialMat(pts1, pts2, intrinsic_matrix)
    R1, R2, t = cv.decomposeEssentialMat(essential_matrix)
    return R1, R2, t

def __triangulate_points__(intrinsic_matrix, pts1, pts2, R1, R2, t):
    # R1, R2, t = __get_rotation_and_translation__(pts1, pts2, intrinsic_matrix)
    projection_1 = np.matmul(intrinsic_matrix, np.hstack((np.identity(3), t)))
    projection_2 = np.matmul(intrinsic_matrix, np.hstack((R1, -1*t)))

    pts1 = np.transpose(pts1)
    pts2 = np.transpose(pts2)

    points_4d = cv.triangulatePoints(np.array(projection_1, dtype=np.float), np.array(projection_2, dtype=np.float), np.array(pts1,dtype=np.float), np.array(pts2, dtype=np.float))

    i = 0
    for x in points_4d[3]:
        points_4d[0][i] = points_4d[0][i]/x
        points_4d[1][i] = points_4d[1][i]/x
        points_4d[2][i] = points_4d[2][i]/x
        points_4d[3][i] = points_4d[3][i]/x
        i += 1

    return projection_1, points_4d

def __project_points__(projection, points_4d):
    projected = np.matmul(projection, points_4d)
    i = 0
    for x in projected[2]:
        projected[0][i] = projected[0][i]/x
        projected[1][i] = projected[1][i]/x
        projected[2][i] = projected[2][i]/x
        i += 1

    x = projected[0]
    y = projected[1]


    im = plt.imread(ct.POSE_WRITE_PATH+"undistorted1.jpg",cv.IMREAD_UNCHANGED)
    rgb_img = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    implot = plt.imshow(rgb_img)
    plt.scatter(x, y)
    plt.show()

    print projected