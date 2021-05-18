# README

## How to run the code?
1. Please run [imageSegmentation.py](imageSegmentation.py)- no arguments are required to be passed. The script will diligently ask your inputs and has accompanying instructions with it, kindly follow them.
2. Expected output - The output will show a window with the segmented image corresponding to the given configuration and a copy is also saved in the current directory with the name that you provided as input.

## File structure - 
- [ProjectReport.pdf](Dhruv_Rathi_CV2021_meanshift.pdf): Assignment report - contains interesting experiments, results and observations. Peek in!
- [utils.py](utils.py): Utility file to handle
    * Data Manipulation
    * Image Utils
    * Pre-processing Utils
    * Miscellaneous Utils

- [experiment_scripts.py](experiment_scripts.py): Holds all the ready to use experiments scripts, connects functions from [experiments.py](experimentss.py)
    * Experiment set 1 - 
        * pts.mat with vanilla meanshift algoritm
        * pts.mat with meanshift optimisation 1 algoritm
        * pts.mat with meanshift optimisation 2 algoritm.
    * Experiment set 2 - Experiment different images with second optimisation - without pre-processing.
    * Experiment set 3 - Experiment different images with second optimisation - with pre-processing - primarily image smoothing.

- [vanilla_algorithm.py](vannila_algorithm.py): Simplest implementation of find_peak and mean_shift.

- [optimisation_one.py](optimisation_one.py): 
    * Similar implementation of find_peak as vanilla
    * Mean Shift optimised by introducing concept of basin of attraction, thus saving iteartions and computation time.

- [optimisation_two.py](optimisation_two.py)
    * find_peak_opt_two() implements search path space and convergence of similar points in the space in a single peak, thus saving iterations and computation time.
    * Similar implementation of mean_shift_opt_2() as done in first optimisation.