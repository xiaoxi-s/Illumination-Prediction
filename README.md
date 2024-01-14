# Illumination-Prediction

# Description

See this [page](https://xiaoxi-s.github.io/Illumination-Prediction/) for complete information.

# Command line

To patch dataset:

`
python3 patch_dataset.py -fp $SOURCE_PATH -dp $DEST_PATH
`


## Setup Tutorial

Note that the setup would also be applicable for the project. The steps would be generally the same. 

### Dependencies

 - conda
 - Python3.7
 - pytorch
 - numpy
 - opencv

### Steps for pytorch tutorial

  - See this [link](https://docs.anaconda.com/anaconda/install/windows/). Remeber to add Anaconda to my Path
  - Create virtual environment under the project directory using `conda create --prefix ./venv python=3.7`
  - Activate the virtual environment using `conda activate ./venv` under your project folder
  - Install pytorch using command line in this [document](https://pytorch.org/get-started/locally/). Check that the installation is successful.
  - Build your Convolutional Neural Network according to this [tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
  - Done! 

# Acknowledgment

  - General Idea: see this [website](https://lvsn.github.io/deepparametric/)
  - Dataset: [The Laval Indoor HDR Dataset](http://indoor.hdrdb.com/)
