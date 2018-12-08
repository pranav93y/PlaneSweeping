from parameters import intrinsic as it
from pose import estimation as et
from utility import constants as ct

DISPLAY = ct.DONT_DISPLAY_PLOT

intrinsic_matrix, distortion_coefficients, mean_error = it.calibrate(DISPLAY)

et.match_and_project(intrinsic_matrix, distortion_coefficients, DISPLAY)
