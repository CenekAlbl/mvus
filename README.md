# MultiViewUnsynch

Tracking and trajectory reconstruction of moving objects using multiple unsynchronized cameras 

Folder and file structure:

./multiviewunsynch - main directory for the project, contains modules, please keep functionality in separate modules based on topic, i.e.
./multiviewunsynch/geometry.py should contain functions that handle geometrical operations, e.g. rotation representation conversions etc.
If you have a misc function that you don't know where to put it, put it in:
./multiviewunsynch/utils.py

./thirdparty - this is where you put code that is not ours

./test - here will be unit tests for our functions

GUIDELINES:

Main branch for developing is dev.

For each new functionality that you start proramming (e.g. epipolar geometry estimation) create a new branch. When your function is ready and tested, merge the branch to dev.

Don't merge to master.

Provide documentation for your code, see the example function x_() in ./multiviewunsynch/geometry.py


