from parameters import intrinsic as it
from pose import estimation as et
from utility import constants as ct

DISPLAY = ct.DONT_DISPLAY_PLOT

intrinsic_matrix, distortion_coefficients, mean_error = it.calibrate(DISPLAY)

_pts1, _pts2, img1, img2 = et.__get_matching_points__(intrinsic_matrix, distortion_coefficients, DISPLAY)
F, pts1, pts2 = et.__compute_fundamental_matrix__(_pts1, _pts2)
et.draw_epipolar_lines(F, img1, img2, pts1, pts2, DISPLAY)
R1, R2, t = et.__get_rotation_and_translation__(pts1, pts2, intrinsic_matrix)
projetion, points_4d = et.__triangulate_points__(intrinsic_matrix, pts1, pts2, R1, R2, t)
et.__project_points__(projetion, points_4d)
