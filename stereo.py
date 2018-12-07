from parameters import intrinsic as it
from pose import estimation as et
from utility import constants as ct

DISPLAY = ct.DONT_DISPLAY_PLOT

intrinsic_matrix, distortion_coefficients, mean_error = it.calibrate(DISPLAY)

pts1, pts2, img1, img2 = et.__get_matching_points__(intrinsic_matrix, distortion_coefficients, DISPLAY)
F, pts1, pts2 = et.__compute_fundamental_matrix__(pts1, pts2)
et.draw_epipolar_lines(F, img1, img2, pts1, pts2, DISPLAY)
