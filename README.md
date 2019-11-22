# MultiViewUnsynch

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


