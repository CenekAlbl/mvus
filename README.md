# MultiViewUnsynch

<p align="center">
<img src="BA_pipeline.jpg" width="600" alt="Multi-view 3D trajectory reconstruction">
</p>

# Inputs

## 2D Detections
text files containing 2D detections and frame indicies in the form:
| x-dim | y-dim | frame-id|

## Camera Intrinsic Calibration files
.json files containing:
- K-matrix: 3*3 matrix of intrinsic camera parameters in the form:

<p align="left">
<img src="k_matrix.jpg" width="200" alt="Intrinsic camera parameter matrix">
</p>

    where:
        - (cx, cy) is a principal point that is usually at the image center
        - fx, fy are the focal lengths expressed in pixel units.
- distCoeff: a vector of [k1,k2,p1,p2[,k3]]
    
    where:
        - k1, k2, k3 are radial distortion coefficients. 
        - p1 and p2 are tangential distortion coefficients. 




# Outputs


## For developing
Tracking and trajectory reconstruction of moving objects using multiple unsynchronized cameras 

Folder and file structure:

./multiviewunsynch - main directory for the project, contains modules, please keep functionality in separate modules based on topic, i.e.

./thirdparty - this is where you put code that is not ours

./deprecated - previous codes that are not used anymore. This folder only temporarily keeps these codes and should be as clean as possible. For any on-going scripts, please keep them in a separate personal branch.

./imput_template.json - template file for loading input data. NOT FINISH YET.

GUIDELINES:

Main branch for developing is dev.

For each new functionality that you start proramming (e.g. epipolar geometry estimation) create a new branch. When your function is ready and tested, merge the branch to dev.

Don't merge to master.

Provide documentation for your code, see the example function x_() in ./multiviewunsynch/geometry.py


