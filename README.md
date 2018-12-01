# Plane Sweeping Stereo

## Prerequisites
- Python 3 
- Python packages: cv2, numpy, os, glob
- MAC OS or Windows 10 (Ubuntu Windows' Bash)

## Installation 
* Clone the repo:
``` 
git clone https://github.com/pranav93y/PlaneSweeping.git
```

* Navigate to `main.py`

* Run the program:
```bash
python3 main.py 
```

## Project Structure
* `main.py`: Main program
* `calibration.py`: Calibrates camera using the chessboard images
* `camera_pose`:
  
## Output
**Part 1**  
For the images that were taken to calibrate the camera we obtain the following:
* Intrinsic matrix K
* Radial distortion coefficients
* Reprojection mean square error
  
**Part 2**    
  
**Part 3**    
Computation for finding the camera relative pose:
* Few epipolar lines on the images.
* The matrix R<sup>R</sup><sub>L</sub> and **r**<sup>R</sup>
* Re-projected feature points on the first image

**Part 4**   
Creating a plane-sweeping stereo  
* Values for *d<sub>min</sub>* and *d<sub>max</sub>*
* N=20 warped second images
* Resulting depth image in grayscale