from parameters import intrinsic as it
from pose import estimation as et
from utility import constants as ct
from stereo import plane_sweep as ps
import numpy as np
DISPLAY = ct.DONT_DISPLAY_PLOT

intrinsic_matrix, distortion_coefficients, mean_error = it.calibrate(DISPLAY)
print intrinsic_matrix
print "----------------------------------"
print distortion_coefficients
print "----------------------------------"

# intrinsic_matrix = np.array([[883.64406339, 0., 507.1981339], [0., 884.98856976, 382.46811708], [0., 0., 1.]])
# distortion_coefficients = np.array([3.69641561e-02,3.48490576e-01,-6.76000221e-04,1.95258318e-04,-1.13082536e+00])

R, t, points_4d, projection_1, projection_2 = et.match_and_project(intrinsic_matrix, distortion_coefficients, DISPLAY)

homography = ps.find_homography(projection_1, projection_2, points_4d, intrinsic_matrix)

ps.get_warped_images(homography, DISPLAY)




