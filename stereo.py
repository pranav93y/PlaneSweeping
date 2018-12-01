from parameters import intrinsic as it
from utility import constants as ct

DISPLAY = ct.DONT_DISPLAY_PLOT

intrinsic_matrix, distortion_coefficients, mean_error = it.calibrate(DISPLAY)
