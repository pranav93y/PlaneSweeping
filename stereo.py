from parameters import intrinsic as it
from pose import estimation as et
from utility import constants as ct
from stereo import plane_sweep as ps
import numpy as np
DISPLAY = ct.DONT_DISPLAY_PLOT

intrinsic_matrix, distortion_coefficients, mean_error = it.calibrate(DISPLAY)

R, t, points_4d, projection_1, projection_2, intrinsic_matrix = et.match_and_project(intrinsic_matrix, distortion_coefficients, DISPLAY)
print "----------------------------------"
print "Distortion Coefficients: "
print distortion_coefficients
print "----------------------------------"
print "Intrinsic Matrix: "
print intrinsic_matrix
print "----------------------------------"
# homography = ps.find_homography(projection_1, projection_2, points_4d, intrinsic_matrix)
#
# ps.get_warped_images(homography, DISPLAY)

# ps.run_abs_diff_and_block_filter(projection_1, projection_2, points_4d, intrinsic_matrix, display=True)
ps.compute_depth(projection_1, projection_2, points_4d, intrinsic_matrix, display=False)

