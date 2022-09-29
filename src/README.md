# Codes
## Files in this directory:
- path.py: In this file, set the path to everything needed, depending on your environment. See below.
- run_*.py: Entry points for running computation. The results will be stored in ../results
- plot_*.py: Entry points for plotting the results. Load the reuslts from ../results and plot them.
- others: Classes and functions referenced from the main codes.
## path.py
Please set the following 3 paths.
- DIR_DATASET_ESC: The path to ESC-50 dataset. You can get the dataset at https://github.com/karolpiczak/ESC-50. It should contain directories named "audio" and "meta".
- DIR_DATASET_TIMIT: The path to TIMIT dataset. You can get the dataset at https://catalog.ldc.upenn.edu/LDC93s1. It should contain a directory named "timit" which in turn contains "TIMIT".
- DIR_REPO_NEUROPHYSIOLOGY: The path to [cascaded-am-tuning-for-sound-recognition](https://github.com/cycentum/cascaded-am-tuning-for-sound-recognition) repository. It should contain a directory named "cascaded-am-tuning-for-sound-recognition" which in turn contains python files and other directories.
