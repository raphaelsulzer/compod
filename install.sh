#!/bin/bash
set -e

# Local variables
ENV_NAME=compod
PYTHON=3.10.10

# Installation script for Anaconda3 environments
echo "____________ Pick conda install _____________"
echo
# Recover the path to conda on your machine
CONDA_DIR=`realpath /opt/miniconda3`
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  CONDA_DIR=`realpath ~/anaconda3`
fi

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide you conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh
echo
echo

echo "________________ Installation _______________"
echo

# Check if the environment exists
if conda env list | awk '{print $1}' | grep -q "^$ENV_NAME$"; then
    read -p "Conda environment '$ENV_NAME' already exists. Do you want to remove and reinstall it? (yes/no): " answer

    if [[ "$answer" == "yes" || "$answer" == "y" ]]; then
        # Remove the environment
        conda env remove --name "$ENV_NAME" --yes > /dev/null 2>&1

        # Double-check removal
        if conda env list | awk '{print $1}' | grep -q "^$ENV_NAME$"; then
            echo "Failed to remove the environment '$ENV_NAME'."
            exit 1
        else
            echo "Conda environment '$ENV_NAME' removed successfully."
        fi

        ## Create a conda environment
        echo "Create conda environment '$ENV_NAME'."
        conda create -y --name $ENV_NAME python=$PYTHON > /dev/null 2>&1

    elif [[ "$answer" == "no" || "$answer" == "n" ]]; then
        echo "Installing in existing environment..."
    else
        echo "Invalid input. Please enter yes or no."
    fi
else
  ## Create a conda environment
  echo "Create conda environment '$ENV_NAME'."
  conda create -y --name $ENV_NAME python=$PYTHON > /dev/null 2>&1
fi


# Activate the env
echo "Activating ${ENV_NAME} conda environment."
source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate ${ENV_NAME}


echo "________________ Installation _______________"
echo

## install ubuntu dependencies
#sudo apt-get update && sudo apt-get install libgomp1 ffmpeg libsm6 libxext6 -y

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Dependencies
conda install -y conda-forge::gsl anaconda::libgomp anaconda::scipy conda-forge::shapely conda-forge::sage conda-forge::tqdm conda-forge::trimesh conda-forge::treelib conda-forge::colorlog pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

pip install open3d
pip install gco-wrapper

pip install .

echo
echo "Run 'conda activate ${ENV_NAME}' to activate the environment."