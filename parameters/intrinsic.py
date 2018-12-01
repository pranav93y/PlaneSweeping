import numpy as np
import cv2 as cv

from utility import constants as ct
from utility import imageutil as im


def calibrate(display=ct.DONT_DISPLAY_PLOT):
    mtx, dist, rvecs, tvecs, objpoints, imgpoints = get_calibration_parameters(display)
    mean_error = calculate_reprojection_error(mtx, dist, rvecs, tvecs, objpoints, imgpoints)

    return mtx, dist, mean_error


def calculate_reprojection_error(mtx, dist, rvecs, tvecs, objpoints, imgpoints):
    tot_error = 0

    for i in range(0, len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        tot_error += error

    print("total error: ", tot_error / len(objpoints))

    return tot_error / len(objpoints)


def get_calibration_parameters(display=ct.DONT_DISPLAY_PLOT):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ct.X_COUNT_CHESS * ct.Y_COUNT_CHESS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ct.Y_COUNT_CHESS, 0:ct.X_COUNT_CHESS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = im.load_images_from_folder(ct.CALIBRATION_READ_PATH)
    i = 0

    for image in images:
        image = cv.resize(image, (ct.CALIB_IMAGE_X, ct.CALIB_IMAGE_Y))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        i = i + 1
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (ct.Y_COUNT_CHESS, ct.X_COUNT_CHESS), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            cv.drawChessboardCorners(image, (ct.Y_COUNT_CHESS, ct.X_COUNT_CHESS), corners, ret)

            im.write(ct.CALIBRATION_WRITE_PATH + "/calibration" + str(i) + ".png", image)

            if display:
                im.display('Image', image)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist, rvecs, tvecs, objpoints, imgpoints
